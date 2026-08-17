/* ----------------------------------------------------------------------
   Eternia paged nonbonded kernel for GROMACS -- DEVICE SIDE.

   Compiled by clang++-22 with -x cuda -DCLIO_GPU_YIELD_CORO=ON. Nothing in
   GROMACS includes this file; the only surface is eternia_nb.h.

   The kernel is the one verified standalone in eternia_nb_bench.cc, byte for
   byte. Only the driver differs: instead of generating a synthetic system it
   takes GROMACS's own pair list and coordinates through the boundary.
------------------------------------------------------------------------- */

#include "eternia_nb.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(ETERNIA_GMX_ENABLED)

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <string>
#include <vector>

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

// GROMACS nbnxm GPU layout constants (sc_gpuClusterSize etc. for CUDA).
constexpr int kClusterSize = 8;    // atoms per cluster
constexpr int kClustersPerSc = 8;  // clusters per supercluster
constexpr int kJGroupSize = 4;     // j-clusters per packed entry
constexpr int kAtomsPerSc = kClusterSize * kClustersPerSc;   // 64

/** One packed j-cluster group, mirroring nbnxm_cj_packed_t's payload. */
struct CjPacked {
  int cj[kJGroupSize];
  unsigned imask[2];
};
/**
 * Ints per packed entry in the paged vector: PADDED TO A POWER OF TWO.
 *
 * The payload is 6 ints (cj[4] + imask[2]). Storing it as 6 means the entry
 * stride does not divide the page size, so entries straddle page boundaries
 * -- and a straddling entry is one no single hold can read, because a hold
 * covers one page. The first version of this kernel silently DROPPED those
 * entries: they were excluded from the segment that started them and from
 * the next one, which the config sweep caught as a wrong answer at 4 KB
 * pages while 64 KB and 1 MB pages passed.
 *
 * Padding to 8 makes the stride divide every page size that is a power-of-two
 * multiple of 32 bytes, so an entry is always wholly inside one page. It
 * costs 25% more list bytes and removes a whole class of boundary handling.
 * Real nbnxm gets the same property from struct alignment.
 */
constexpr int kCjPackedInts = 8;   // 6 used, 2 padding

/** Pages recorded per pass-A window: 2048 bits = 64 u32 = 32 u64. */
constexpr int kPageBitmapBits = 2048;
constexpr int kPageBitmapWords = kPageBitmapBits / 32;

/** u64 slots of per-block global scratch: 2 for the page range, the touched
 *  bitmap, and room for kAtomsPerSc*3 staged float coordinates. */
constexpr int kScratchU64PerBlock =
    2 + kPageBitmapWords / 2 + (kAtomsPerSc * 3 + 1) / 2;

// Sci comes from eternia_nb.h -- one definition, shared by the kernel and
// the boundary, so the two cannot drift apart.
using eternia_gmx::Sci;

#if defined(CLIO_YIELD_CORO)
static constexpr u32 kYieldLaneBytes = 4096;
#else
static constexpr u32 kYieldLaneBytes = 256;
#endif

#if defined(CLIO_YIELD_CORO) && defined(__clang__) && defined(__CUDA__)
#define ETERNIA_NB_CORO 1
#endif

struct NbParams {
  float cutoffSq;
  float c6;
  float c12;
  int centralShift;   //!< see Config::centralShift
};

/** LJ force magnitude / r, and energy, matching the reference exactly. */
CTP_INLINE_CROSS_FUN void LjPair(float rsq, const NbParams &p, float *fscal,
                                 float *energy) {
  const float rinv2 = 1.0f / rsq;
  const float rinv6 = rinv2 * rinv2 * rinv2;
  *fscal = rinv6 * (12.0f * p.c12 * rinv6 - 6.0f * p.c6) * rinv2;
  *energy = rinv6 * (p.c12 * rinv6 - p.c6);
}

#if defined(ETERNIA_NB_CORO)

/**
 * One block per i-supercluster.
 *
 * The block holds its 64 i-atoms once, then streams its contiguous slice of
 * the packed pair list. Both the list and the coordinates are paged; the
 * force output is resident.
 */
__device__ gy::YCoroMain NbCoro(gv::DeviceVector<int> cjp,
                                gv::DeviceVector<float> xq, float *f,
                                const float *shiftVec,
                                const Sci *scis, int numSci, NbParams prm,
                                double *energy_out,
                                unsigned long long *pairs_out, u64 *scratch,
                                u32 block, u32 nblocks) {
  u64 run = 0;
  double e_local = 0.0;
  unsigned long long n_pairs = 0;
  // GLOBAL, not __shared__: a co_await can exit the kernel and have the
  // driver relaunch this block, at which point shared memory is whatever the
  // new launch got. Two u64 per block.
  // Per-block scratch: 2 u64 for the page range, then 64*3 floats for the
  // staged i-supercluster coordinates. GLOBAL, not __shared__, because a
  // co_await can exit the kernel and have this block relaunched.
  // DROP THIS BLOCK'S CACHES FIRST.
  //
  // Today this is a no-op: the GROMACS hook creates a fresh context for every
  // call, so no page is ever resident from a previous step. It is here
  // because that is a property of the CALLER, not of this kernel, and the
  // obvious optimisation -- cache the context across steps instead of
  // rebuilding it -- would silently start serving the previous step's
  // coordinates.
  //
  // That is not hypothetical. The LBANN integration cached its context, did
  // not drop, and served the first call's weights for the rest of training:
  // the forward pass was exact to 1.4e-07 on the first call and wrong by
  // 1.2e-03 by the second, while the objective still fell so it looked like
  // training was working.
  __syncthreads();
  if (threadIdx.x == 0) {
    cjp.DropAll();
    xq.DropAll();
  }
  __syncthreads();

  u64 *pg_lo_s = scratch + static_cast<u64>(block) * kScratchU64PerBlock;
  u64 *pg_hi_s = pg_lo_s + 1;
  u32 *touched = reinterpret_cast<u32 *>(pg_lo_s + 2);
  float *xi_s = reinterpret_cast<float *>(pg_lo_s + 2 + kPageBitmapWords / 2);

  for (int s = block; s < numSci; s += nblocks) {
    const Sci sc = scis[s];
    // The diagonal: a cluster pair at the self-image shift whose i- and
    // j-clusters coincide. nbnxm counts each unordered atom pair exactly
    // once, which for that case means the triangle j > i.
    const bool isCentral = (sc.shift == prm.centralShift);
    const u64 i0 = static_cast<u64>(sc.sci) * kAtomsPerSc;

    // STAGE the i-supercluster's coordinates OUT of the page cache.
    //
    // xi and the j-side views share one page table, so holding a j-page can
    // evict the i-page while the per-thread last_page_ still points at that
    // slot -- now refilled with another page. Reading an i-atom through it
    // then returns some other atom's coordinates.
    //
    // The config sweep is what exposed it: every configuration with a small
    // cache (slots 2-3) and many pages failed, while slots 8 and a 1 MB page
    // (one page for the whole array) passed. Copying the 64 atoms out once
    // per sci removes the aliasing entirely -- the same fix the LAMMPS pair
    // style needed, and the third time this hazard has appeared.
    {
      gv::DeviceVector<float> xi = xq;
      co_await xi.HoldPageCoro(i0 * 4, static_cast<u64>(kAtomsPerSc) * 4, &run);
      // The i-atoms carry this entry's periodic shift, exactly as nbnxm does
      // (xqbuf = xq[ai] + shift_vec[nb_sci.shift]). Staging is the right place
      // for it: applied once per atom rather than once per pair.
      const float sx = shiftVec ? shiftVec[sc.shift * 3 + 0] : 0.0f;
      const float sy = shiftVec ? shiftVec[sc.shift * 3 + 1] : 0.0f;
      const float sz = shiftVec ? shiftVec[sc.shift * 3 + 2] : 0.0f;
      for (u32 t = threadIdx.x; t < kAtomsPerSc * 3; t += blockDim.x) {
        const u32 d = t % 3;
        const float sh = (d == 0) ? sx : ((d == 1) ? sy : sz);
        xi_s[t] = xi.at(i0 * 4 + (t / 3) * 4 + d) + sh;
      }
      __syncthreads();
    }

    // Stream this sci's slice of the packed list, page by page.
    const u64 k0 = static_cast<u64>(sc.cjPackedBegin) * kCjPackedInts;
    const u64 k1 = static_cast<u64>(sc.cjPackedEnd) * kCjPackedInts;
    const u64 pe = cjp.h_->elems_per_page_;

    for (u64 off = k0; off < k1; ) {
      const u64 seg_end = ((off / pe) + 1) * pe < k1 ? ((off / pe) + 1) * pe : k1;
      co_await cjp.HoldPageCoro(off, seg_end - off, &run);

      // Entries are power-of-two sized (see kCjPackedInts), so every entry
      // overlapping this page lies WHOLLY inside it and this range is exact.
      const u64 first = off / kCjPackedInts;
      const u64 last = seg_end / kCjPackedInts;   // exclusive

      // PASS A: which coordinate pages do this segment's j-clusters touch?
      //
      // The j-side CANNOT be gathered with TryHoldRawConst. That call is
      // probe-only: it returns null on a miss and never faults, so with
      // nothing else touching the coordinate vector every probe would miss
      // and the pairs would be silently skipped. That exact mistake cost
      // three rounds on the LAMMPS pair style; the fix there and here is to
      // make the j-side page-major, so every hold is a real, block-collective
      // fault.
      const u64 xpe = xq.h_->elems_per_page_;
      if (threadIdx.x == 0) {
        pg_lo_s[0] = ~0ull;
        pg_hi_s[0] = 0ull;
      }
      __syncthreads();
      for (u64 entry = first + threadIdx.x; entry < last; entry += blockDim.x) {
        const u64 base = entry * kCjPackedInts;
        for (int jslot = 0; jslot < kJGroupSize; ++jslot) {
          const int cj = cjp.at(base + jslot);
          if (cj < 0) continue;
          const u64 ja0 = static_cast<u64>(cj) * kClusterSize;
          const u64 p0 = (ja0 * 4) / xpe;
          const u64 p1 = ((ja0 + kClusterSize - 1) * 4 + 3) / xpe;
          atomicMin(reinterpret_cast<unsigned long long *>(&pg_lo_s[0]),
                    static_cast<unsigned long long>(p0));
          atomicMax(reinterpret_cast<unsigned long long *>(&pg_hi_s[0]),
                    static_cast<unsigned long long>(p1));
        }
      }
      __syncthreads();
      const u64 pg_lo = pg_lo_s[0], pg_hi = pg_hi_s[0];

      // PASS B: hold each TOUCHED coordinate page once, block-collectively.
      //
      // Only the touched ones. An earlier version held every page in
      // [pg_lo, pg_hi], which for scattered j-clusters is a huge range that is
      // mostly empty: the coordinate cache thrashed completely (2,173,228
      // faults against 2,172,716 evictions on a 4.2M-atom run) and the kernel
      // took 411 s of a 417 s run. The bitmap is what makes the held set
      // proportional to the work rather than to the address span.
      if (pg_lo != ~0ull) {
        for (u64 win = pg_lo; win <= pg_hi; win += kPageBitmapBits) {
          const u64 win_hi =
              (win + kPageBitmapBits - 1 < pg_hi) ? (win + kPageBitmapBits - 1)
                                                  : pg_hi;
          for (u32 w = threadIdx.x; w < kPageBitmapWords; w += blockDim.x) {
            touched[w] = 0u;
          }
          __syncthreads();
          for (u64 entry = first + threadIdx.x; entry < last;
               entry += blockDim.x) {
            const u64 base = entry * kCjPackedInts;
            for (int jslot = 0; jslot < kJGroupSize; ++jslot) {
              const int cj = cjp.at(base + jslot);
              if (cj < 0) continue;
              const u64 ja0 = static_cast<u64>(cj) * kClusterSize;
              const u64 q0 = (ja0 * 4) / xpe;
              const u64 q1 = ((ja0 + kClusterSize - 1) * 4 + 3) / xpe;
              for (u64 q = q0; q <= q1; ++q) {
                if (q < win || q > win_hi) continue;
                const u32 b = static_cast<u32>(q - win);
                atomicOr(&touched[b >> 5], 1u << (b & 31u));
              }
            }
          }
          __syncthreads();

          for (u64 pg = win; pg <= win_hi; ++pg) {
            const u32 b = static_cast<u32>(pg - win);
            if ((touched[b >> 5] & (1u << (b & 31u))) == 0u) continue;
            gv::DeviceVector<float> xj = xq;
            co_await xj.HoldPageCoro(pg * xpe, xpe, &run);
            const u64 xlo = pg * xpe, xhi = xlo + xpe;

            for (u64 entry = first; entry < last; ++entry) {
              const u64 base = entry * kCjPackedInts;
              for (u32 t = threadIdx.x; t < kAtomsPerSc * kJGroupSize;
                   t += blockDim.x) {
                const u32 ii = t / kJGroupSize;
                const u32 jslot = t % kJGroupSize;
                const int cj = cjp.at(base + jslot);
                if (cj < 0) continue;
                const u32 icl = ii / kClusterSize;

                const u64 ia = i0 + ii;
                const float ix = xi_s[ii * 3 + 0];
                const float iy = xi_s[ii * 3 + 1];
                const float iz = xi_s[ii * 3 + 2];

                for (int jj = 0; jj < kClusterSize; ++jj) {
                  // TWO mask words, not one. nbnxm splits a cluster pair
                  // across two warps (imei[c_clusterPairSplit]) and each warp
                  // owns half of the j-cluster's atoms, so the two words can
                  // differ once pruning has run. Using imei[0] for all eight
                  // j-atoms -- which the standalone bench did, and said so --
                  // both keeps pairs the second warp pruned and drops pairs
                  // it kept.
                  const unsigned imask = static_cast<unsigned>(cjp.at(
                      base + kJGroupSize + (jj >= kClusterSize / 2 ? 1 : 0)));
                  if (!((imask >> (jslot * kClustersPerSc + icl)) & 1u)) continue;
                  // COUNT EACH UNORDERED PAIR ONCE, as nbnxm does. A pair of
                  // distinct clusters is listed in one direction only, so
                  // every (i, j) here is already unique. The self-cluster is
                  // the exception: it would yield both (a, b) and (b, a), so
                  // it is masked to the triangle -- the same rule as nbnxm's
                  // `nonSelfInteraction | (ci != cj)`.
                  const bool sameCluster =
                      (static_cast<u64>(cj) ==
                       static_cast<u64>(sc.sci) * kClustersPerSc + icl);
                  if (isCentral && sameCluster &&
                      jj <= static_cast<int>(ii % kClusterSize)) {
                    continue;
                  }
                  const u64 ja = static_cast<u64>(cj) * kClusterSize + jj;
                  if (ja == ia) continue;
                  const u64 jo = ja * 4;
                  if (jo < xlo || jo + 3 >= xhi) continue;
                  const float dx = ix - xj.at(jo + 0);
                  const float dy = iy - xj.at(jo + 1);
                  const float dz = iz - xj.at(jo + 2);
                  const float rsq = dx * dx + dy * dy + dz * dz;
                  ++n_pairs;
                  if (rsq >= prm.cutoffSq || rsq == 0.0f) continue;
                  float fscal, ener;
                  LjPair(rsq, prm, &fscal, &ener);
                  // No halving: nbnxm lists each unordered pair once, so
                  // every pair reached here is counted exactly once already.
                  e_local += static_cast<double>(ener);
                  atomicAdd(&f[ia * 4 + 0], dx * fscal);
                  atomicAdd(&f[ia * 4 + 1], dy * fscal);
                  atomicAdd(&f[ia * 4 + 2], dz * fscal);
                }
              }
            }
            __syncthreads();
          }
        }
      }
      __syncthreads();
      off = seg_end;
    }
  }

  atomicAdd(energy_out, e_local);
  atomicAdd(pairs_out, n_pairs);
  co_return;
}

__global__ void NbKernel(clio::run::IpcManagerGpuInfo info,
                         gv::DeviceVector<int> cjp, gv::DeviceVector<float> xq,
                         float *f, const float *shiftVec,
                         const Sci *scis, int numSci, NbParams prm,
                         double *energy_out, unsigned long long *pairs_out,
                         u64 *scratch, u32 nblocks, gy::YieldableView<> yv,
                         gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  cjp.block_override_ = yv.Block();
  xq.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(NbCoro(cjp, xq, f, shiftVec, scis, numSci, prm, energy_out,
                        pairs_out, scratch, yv.Block(), nblocks));
}

#endif  // ETERNIA_NB_CORO


#if !CTP_IS_DEVICE_PASS

namespace eternia_gmx {
namespace {
std::string g_err;
void SetErr(const char* s) { g_err = s; }
class YieldRunner {
 public:
  YieldRunner(unsigned nblocks, unsigned nthreads)
      : drv_(nblocks, nthreads), stack_(nblocks, nthreads, kYieldLaneBytes) {}
  template <typename LaunchT>
  u32 Run(LaunchT &&launch) {
    drv_.Reset();
    stack_.Reset();
    return drv_.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> v) { launch(g, b, v, stack_.View()); },
        [] {}, /*max_rounds=*/2000000);
  }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};

}  // namespace

struct Context {
  Config cfg;
  ClusterLayout lay;
  int natoms = 0, numSci = 0, numCjPacked = 0;
  gv::Vector<int>* vcj = nullptr;
  gv::Vector<float>* vxq = nullptr;
  Sci* d_sci = nullptr;
  double* d_energy = nullptr;
  unsigned long long* d_pairs = nullptr;
  u64* d_scratch = nullptr;
  Stats stats;
};

bool Available() { return true; }
const char* LastError() { return g_err.c_str(); }

Context* Create(const Config& cfg, const ClusterLayout& lay, int natoms,
                int numSci, int numCjPacked)
{
  if (natoms <= 0 || numSci <= 0 || numCjPacked <= 0) {
    SetErr("Create: non-positive sizes");
    return nullptr;
  }
  if (lay.clusterSize != kClusterSize || lay.clustersPerSc != kClustersPerSc ||
      lay.jGroupSize != kJGroupSize) {
    SetErr("Create: cluster layout does not match the compiled-in constants");
    return nullptr;
  }
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    SetErr("Clio runtime init failed"); return nullptr;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    SetErr("Clio CTE client init failed"); return nullptr;
  }
  auto* ctx = new Context();
  ctx->cfg = cfg; ctx->lay = lay;
  ctx->natoms = natoms; ctx->numSci = numSci; ctx->numCjPacked = numCjPacked;
  const u64 cj_elems = static_cast<u64>(numCjPacked) * kCjPackedInts;
  const u64 xq_elems = static_cast<u64>(natoms) * 4;
  try {
    ctx->vcj = new gv::Vector<int>(std::string(cfg.tag) + "_cj", {cfg.gpu_id},
                                   cfg.page_bytes, cfg.nblocks, cfg.slots, cj_elems);
    ctx->vxq = new gv::Vector<float>(std::string(cfg.tag) + "_xq", {cfg.gpu_id},
                                     cfg.page_bytes, cfg.nblocks, cfg.slots, xq_elems);
  } catch (const std::exception& e) {
    SetErr(e.what()); Destroy(ctx); return nullptr;
  }
  if (cfg.stats) { ctx->vcj->EnableStats(); ctx->vxq->EnableStats(); }
  cudaMalloc(&ctx->d_sci, static_cast<size_t>(numSci) * sizeof(Sci));
  cudaMalloc(&ctx->d_energy, sizeof(double));
  cudaMalloc(&ctx->d_pairs, sizeof(unsigned long long));
  cudaMalloc(&ctx->d_scratch, static_cast<u64>(cfg.nblocks) * kScratchU64PerBlock * sizeof(u64));
  return ctx;
}

void Destroy(Context* ctx)
{
  if (!ctx) return;
  delete ctx->vcj; delete ctx->vxq;
  if (ctx->d_sci) cudaFree(ctx->d_sci);
  if (ctx->d_energy) cudaFree(ctx->d_energy);
  if (ctx->d_pairs) cudaFree(ctx->d_pairs);
  if (ctx->d_scratch) cudaFree(ctx->d_scratch);
  delete ctx;
}

namespace {
template <typename T>
bool SeedVector(gv::Vector<T>& vec, const std::vector<T>& host, u64 page_bytes)
{
  clio::cte::core::Client core(clio::cte::core::kCtePoolId);
  const u64 pe = page_bytes / sizeof(T);
  const u64 np = (host.size() + pe - 1) / pe;
  std::vector<T> buf(pe);
  for (u64 p = 0; p < np; ++p) {
    std::memset(buf.data(), 0, buf.size() * sizeof(T));
    const u64 lo = p * pe;
    const u64 hi = std::min(lo + pe, static_cast<u64>(host.size()));
    std::memcpy(buf.data(), host.data() + lo, (hi - lo) * sizeof(T));
    char nm[32];
    gv::PageBlobName(p, nm);
    auto f = core.AsyncPutBlob(vec.TagId(), std::string(nm), 0,
                               buf.size() * sizeof(T),
                               reinterpret_cast<const char*>(buf.data()), 1.0f);
    f.Wait();
    if (f.get() == nullptr || f->GetReturnCode() != 0) return false;
  }
  return true;
}
}  // namespace

bool Upload(Context* ctx, const int* cj, const unsigned* imask,
            const float* xq, const Sci* scis)
{
  if (!ctx || !cj || !imask || !xq || !scis) {
    SetErr("Upload: null argument"); return false;
  }
  // Repack GROMACS's separate cj and imask arrays into this kernel's padded
  // entry layout: kCjPackedInts ints per entry, power-of-two so an entry never
  // straddles a page (see the bench for why that matters).
  std::vector<int> flat(static_cast<size_t>(ctx->numCjPacked) * kCjPackedInts, 0);
  for (int e = 0; e < ctx->numCjPacked; ++e) {
    for (int q = 0; q < kJGroupSize; ++q) {
      flat[static_cast<size_t>(e) * kCjPackedInts + q] =
        cj[static_cast<size_t>(e) * kJGroupSize + q];
    }
    flat[static_cast<size_t>(e) * kCjPackedInts + kJGroupSize] =
      static_cast<int>(imask[static_cast<size_t>(e) * 2 + 0]);
    flat[static_cast<size_t>(e) * kCjPackedInts + kJGroupSize + 1] =
      static_cast<int>(imask[static_cast<size_t>(e) * 2 + 1]);
  }
  if (!SeedVector(*ctx->vcj, flat, ctx->cfg.page_bytes)) {
    SetErr("Upload: pair-list page write failed"); return false;
  }
  std::vector<float> xqv(xq, xq + static_cast<size_t>(ctx->natoms) * 4);
  if (!SeedVector(*ctx->vxq, xqv, ctx->cfg.page_bytes)) {
    SetErr("Upload: coordinate page write failed"); return false;
  }
  cudaMemcpy(ctx->d_sci, scis, static_cast<size_t>(ctx->numSci) * sizeof(Sci),
             cudaMemcpyHostToDevice);
  return true;
}

bool Compute(Context* ctx, float c6, float c12, float cutoffSq,
             const float* shiftVec_device, float* forces_device,
             double* energy_out)
{
  if (!ctx || !forces_device) { SetErr("Compute: null argument"); return false; }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(ctx->cfg.gpu_id);
  cudaMemset(ctx->d_energy, 0, sizeof(double));
  cudaMemset(ctx->d_pairs, 0, sizeof(unsigned long long));

  NbParams prm{cutoffSq, c6, c12, ctx->cfg.centralShift};
  auto dcj = ctx->vcj->GetDevice(ctx->cfg.gpu_id);
  auto dxq = ctx->vxq->GetDevice(ctx->cfg.gpu_id);
  YieldRunner runner(ctx->cfg.nblocks, ctx->cfg.nthreads);
  const u32 rounds = runner.Run(
    [&](dim3 g, dim3 b, gy::YieldableView<> v, gy::YieldStackView sv) {
      NbKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
        gpu, dcj, dxq, forces_device, shiftVec_device, ctx->d_sci, ctx->numSci, prm,
        ctx->d_energy, ctx->d_pairs, ctx->d_scratch, ctx->cfg.nblocks, v, sv);
    });
  const cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) { SetErr(cudaGetErrorString(le)); return false; }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    SetErr(cudaGetErrorString(cudaGetLastError())); return false;
  }
  if (rounds == 0) { SetErr("yield driver made no progress"); return false; }

  if (energy_out) {
    cudaMemcpy(energy_out, ctx->d_energy, sizeof(double), cudaMemcpyDeviceToHost);
  }
  unsigned long long np = 0;
  cudaMemcpy(&np, ctx->d_pairs, sizeof(np), cudaMemcpyDeviceToHost);
  ctx->stats.pairs = np;
  if (ctx->cfg.stats) {
    const auto s1 = ctx->vcj->ReadStats(ctx->cfg.gpu_id);
    const auto s2 = ctx->vxq->ReadStats(ctx->cfg.gpu_id);
    ctx->stats.list_faults = s1.faults; ctx->stats.list_evicts = s1.evicts;
    ctx->stats.xq_faults = s2.faults;   ctx->stats.xq_evicts = s2.evicts;
    ctx->stats.get_errors = s1.get_errors + s2.get_errors;
  }
  return true;
}

Stats GetStats(Context* ctx) { return ctx ? ctx->stats : Stats(); }

}  // namespace eternia_gmx

#endif  // !CTP_IS_DEVICE_PASS

#else   // !ETERNIA_GMX_ENABLED

namespace eternia_gmx {
struct Context {};
bool Available() { return false; }
const char* LastError() { return "built without the Eternia backend"; }
Context* Create(const Config&, const ClusterLayout&, int, int, int) { return nullptr; }
void Destroy(Context*) {}
bool Upload(Context*, const int*, const unsigned*, const float*, const Sci*) { return false; }
bool Compute(Context*, float, float, float, const float*, float*, double*) { return false; }
Stats GetStats(Context*) { return Stats(); }
}  // namespace eternia_gmx

#endif
