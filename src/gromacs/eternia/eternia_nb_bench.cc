/* ----------------------------------------------------------------------
   GROMACS nonbonded (nbnxm) cluster pair interaction over an Eternia
   (Clio CTE gpu_vector) pair list and coordinate array.

   WHY THIS TARGET AND NOT PME
   ---------------------------
   Paging an array past VRAM only helps if its CONSUMER can stream it too.
   The PME grid fails that test: the stage right after spread is a cuFFT 3D
   transform, which takes raw device pointers to the whole real and complex
   grids and has no host-backed mode, so the ceiling stays at ~2x the grid
   however clever spread is. See ../eternia/README.md.

   The nonbonded path passes it. Its consumer is force accumulation, which is
   local: each i-cluster's forces depend on a bounded set of j-clusters and
   nothing reduces over the whole system. That is the same property that makes
   the LAMMPS integration work end to end.

   WHAT IS OUT OF CORE
   -------------------
   Two arrays, in GROMACS's own layout:

     cjPacked   the packed j-cluster list, {int cj[4]; uint imask[2];} per
                entry -- O(natoms * neighbours / clusterSize), and the thing
                that actually dominates memory at scale
     xq         coordinates + charge, float4 per atom

   Forces stay RESIDENT. They are only O(natoms), which is small beside the
   pair list, and keeping them resident means no paged array is ever written
   -- so the page-aligned-ownership rule that the PME spread needed does not
   arise here at all. (Two blocks may freely share a read-only page.)

   DECOMPOSITION
   -------------
   One block per i-supercluster, exactly as GROMACS does it. An sci entry
   owns a CONTIGUOUS range [cjPackedBegin, cjPackedEnd) of the pair list, so
   a block streams its own range page by page with block-collective holds --
   the same shape as the LAMMPS neighbour walk. Within a page, the j-cluster
   coordinates are gathered through the xq cache.
------------------------------------------------------------------------- */

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

/** Mirrors nbnxm_sci_t. */
struct Sci {
  int sci;
  int shift;
  int cjPackedBegin;
  int cjPackedEnd;
};

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
  u64 *pg_lo_s = scratch + static_cast<u64>(block) * kScratchU64PerBlock;
  u64 *pg_hi_s = pg_lo_s + 1;
  u32 *touched = reinterpret_cast<u32 *>(pg_lo_s + 2);
  float *xi_s = reinterpret_cast<float *>(pg_lo_s + 2 + kPageBitmapWords / 2);

  for (int s = block; s < numSci; s += nblocks) {
    const Sci sc = scis[s];
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
      for (u32 t = threadIdx.x; t < kAtomsPerSc * 3; t += blockDim.x) {
        xi_s[t] = xi.at(i0 * 4 + (t / 3) * 4 + (t % 3));
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
                const unsigned imask =
                    static_cast<unsigned>(cjp.at(base + kJGroupSize));
                const u32 icl = ii / kClusterSize;
                if (!((imask >> (jslot * kClustersPerSc + icl)) & 1u)) continue;

                const u64 ia = i0 + ii;
                const float ix = xi_s[ii * 3 + 0];
                const float iy = xi_s[ii * 3 + 1];
                const float iz = xi_s[ii * 3 + 2];

                for (int jj = 0; jj < kClusterSize; ++jj) {
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
                  e_local += 0.5 * static_cast<double>(ener);
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
                         float *f, const Sci *scis, int numSci, NbParams prm,
                         double *energy_out, unsigned long long *pairs_out,
                         u64 *scratch, u32 nblocks, gy::YieldableView<> yv,
                         gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  cjp.block_override_ = yv.Block();
  xq.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(NbCoro(cjp, xq, f, scis, numSci, prm, energy_out, pairs_out,
                        scratch, yv.Block(), nblocks));
}

#endif  // ETERNIA_NB_CORO

#if !CTP_IS_DEVICE_PASS

#if defined(ETERNIA_NB_CORO)
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
#endif

int main(int argc, char **argv) {
  u64 nsc = 256;          // superclusters -> natoms = nsc * 64
  double density = 0.8;
  float cutoff = 2.5f;
  u64 page_kb = 256;
  u32 blocks = 32, threads = 128, slots = 8;
  bool verify = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return static_cast<u64>(std::atoll(argv[++i])); };
    if (a == "--sc") nsc = next();
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--slots") slots = static_cast<u32>(next());
    else if (a == "--cutoff") cutoff = static_cast<float>(std::atof(argv[++i]));
    else if (a == "--no-verify") verify = false;
    else {
      std::fprintf(stderr, "usage: %s [--sc N] [--page-kb N] [--blocks N]\n"
                           "          [--threads N] [--slots N] [--cutoff R]\n"
                           "          [--no-verify]\n", argv[0]);
      return 2;
    }
  }
  const u64 natoms = nsc * kAtomsPerSc;
  const u64 nclusters = natoms / kClusterSize;

  // Atoms on a jittered lattice, laid out so cluster c holds atoms
  // [c*8, c*8+8) that are spatially close -- which is what the real nbnxm
  // grid produces and what makes paging pay.
  std::vector<float> xq(natoms * 4);
  // Cluster lattice: side^3 >= nclusters, so cluster c sits at
  // (c%side, (c/side)%side, c/side^2). Making the CLUSTER grid cubic (rather
  // than the atom grid) is what lets the neighbour list be enumerated
  // analytically below instead of searched, which is the difference between
  // O(nsc * nclusters) and O(nsc) host time -- and therefore between
  // stopping at a few thousand atoms and reaching past VRAM.
  const u64 side = static_cast<u64>(std::ceil(std::cbrt((double)nclusters)));
  const double spacing = std::cbrt(1.0 / density);
  {
    unsigned sd = 7u;
    auto rnd = [&]() { sd = sd * 1664525u + 1013904223u;
                       return (float)(sd >> 8) / (float)(1u << 24); };
    for (u64 a = 0; a < natoms; ++a) {
      // Morton-ish: consecutive atoms stay near each other in space.
      const u64 c = a / kClusterSize;
      const u64 cx = c % side, cy = (c / side) % side, cz = c / (side * side);
      (void)0;
      const u64 l = a % kClusterSize;
      xq[a * 4 + 0] = (float)((cx + 0.5 * (l & 1)) * spacing) + 0.01f * rnd();
      xq[a * 4 + 1] = (float)((cy + 0.5 * ((l >> 1) & 1)) * spacing) + 0.01f * rnd();
      xq[a * 4 + 2] = (float)((cz + 0.5 * ((l >> 2) & 1)) * spacing) + 0.01f * rnd();
      xq[a * 4 + 3] = 0.0f;
    }
  }

  // Cluster bounding centres, for a cheap pair search.
  std::vector<float> cc(nclusters * 3, 0.0f);
  for (u64 c = 0; c < nclusters; ++c) {
    for (int l = 0; l < kClusterSize; ++l)
      for (int d = 0; d < 3; ++d) cc[c * 3 + d] += xq[(c * kClusterSize + l) * 4 + d];
    for (int d = 0; d < 3; ++d) cc[c * 3 + d] /= kClusterSize;
  }

  // Build the packed pair list. SYMMETRIC (each cluster pair appears from
  // both sides), which is what makes sum(f) == 0 a real invariant below.
  const float searchSq = (cutoff + 2.0f * (float)spacing) * (cutoff + 2.0f * (float)spacing);
  std::vector<Sci> scis;
  std::vector<int> cjp;   // flattened: kCjPackedInts ints per entry
  for (u64 s = 0; s < nsc; ++s) {
    Sci sc{};
    sc.sci = (int)s; sc.shift = 0;
    sc.cjPackedBegin = (int)(cjp.size() / kCjPackedInts);
    std::vector<int> js;
    // Analytic neighbours: the i-supercluster's 8 clusters occupy a known
    // lattice span, and any cluster within R lattice steps of it is a
    // candidate. Enumerating that box is linear in the result size; the
    // earlier all-pairs scan was quadratic and could not have reached the
    // sizes this exists to test.
    {
      const int R = (int)std::ceil((cutoff + 2.0 * spacing) / spacing);
      long lo[3] = {LONG_MAX, LONG_MAX, LONG_MAX};
      long hi[3] = {LONG_MIN, LONG_MIN, LONG_MIN};
      for (int ic = 0; ic < kClustersPerSc; ++ic) {
        const u64 icl = s * kClustersPerSc + ic;
        if (icl >= nclusters) continue;
        const long p[3] = {(long)(icl % side), (long)((icl / side) % side),
                           (long)(icl / (side * side))};
        for (int d = 0; d < 3; ++d) {
          lo[d] = std::min(lo[d], p[d]); hi[d] = std::max(hi[d], p[d]);
        }
      }
      for (long z = lo[2] - R; z <= hi[2] + R; ++z) {
        if (z < 0 || z >= (long)side) continue;
        for (long y = lo[1] - R; y <= hi[1] + R; ++y) {
          if (y < 0 || y >= (long)side) continue;
          for (long x = lo[0] - R; x <= hi[0] + R; ++x) {
            if (x < 0 || x >= (long)side) continue;
            const u64 jc = (u64)x + (u64)y * side + (u64)z * side * side;
            if (jc < nclusters) js.push_back((int)jc);
          }
        }
      }
    }
    for (size_t b = 0; b < js.size(); b += kJGroupSize) {
      int cjv[kJGroupSize];
      for (int q = 0; q < kJGroupSize; ++q)
        cjv[q] = (b + q < js.size()) ? js[b + q] : -1;
      unsigned im[2] = {0u, 0u};
      for (int q = 0; q < kJGroupSize; ++q) {
        if (cjv[q] < 0) continue;
        for (int ic = 0; ic < kClustersPerSc; ++ic) {
          const int bit = q * kClustersPerSc + ic;
          im[ic / 4 >= 4 ? 1 : 0] |= 0u;      // keep both words defined
          if (bit < 32) im[0] |= (1u << bit); else im[1] |= (1u << (bit - 32));
        }
      }
      for (int q = 0; q < kJGroupSize; ++q) cjp.push_back(cjv[q]);
      cjp.push_back((int)im[0]);
      cjp.push_back((int)im[1]);
      for (int q = kJGroupSize + 2; q < kCjPackedInts; ++q) cjp.push_back(0);
    }
    sc.cjPackedEnd = (int)(cjp.size() / kCjPackedInts);
    scis.push_back(sc);
  }

  const double list_gib = (double)cjp.size() * sizeof(int) / (1024.0*1024.0*1024.0);
  const double xq_gib = (double)xq.size() * sizeof(float) / (1024.0*1024.0*1024.0);
  std::printf("GROMACS nbnxm nonbonded over an Eternia pair list\n"
              "  atoms=%llu clusters=%llu superclusters=%llu\n"
              "  cjPacked=%llu entries = %.3f GiB | xq = %.3f GiB | total %.3f GiB\n"
              "  page=%lluKB blocks=%u threads=%u slots=%u cutoff=%.2f\n",
              (unsigned long long)natoms, (unsigned long long)nclusters,
              (unsigned long long)nsc,
              (unsigned long long)(cjp.size() / kCjPackedInts), list_gib, xq_gib,
              list_gib + xq_gib, (unsigned long long)page_kb, blocks, threads,
              slots, cutoff);

#if !defined(ETERNIA_NB_CORO)
  std::fprintf(stderr, "built without CUDA + clang device coroutines\n");
  return 1;
#else
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) return 1;
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) return 1;
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  gv::Vector<int> vcj("gmx_eternia_nb_cj", {0}, page_kb * 1024, blocks, slots,
                      cjp.size());
  gv::Vector<float> vxq("gmx_eternia_nb_xq", {0}, page_kb * 1024, blocks, slots,
                        xq.size());
  vcj.EnableStats();
  vxq.EnableStats();

  clio::cte::core::Client core(clio::cte::core::kCtePoolId);
  // PIPELINED SEED. One synchronous PutBlob per page means a round trip per
  // page, and at these sizes that is tens of thousands of them -- setup
  // dominates the run and says nothing about the vector. Keep kInFlight puts
  // outstanding instead.
  //
  // The ring of buffers is required, not an optimisation: the private-memory
  // put stages through SHM and the staging buffer is released when the future
  // is waited on, so reusing one host buffer while a put is still in flight
  // would hand the runtime bytes that have already been overwritten.
  constexpr int kInFlight = 16;
  auto seed = [&](auto &vec, const auto &host, u64 esz) {
    const u64 pe = (page_kb * 1024) / esz;
    const u64 np = (host.size() + pe - 1) / pe;
    std::vector<std::vector<char>> bufs(kInFlight,
                                        std::vector<char>(page_kb * 1024));
    std::vector<clio::run::Future<clio::cte::core::PutBlobTask>> futs(kInFlight);
    std::vector<bool> live(kInFlight, false);
    for (u64 p = 0; p < np; ++p) {
      const int slot = (int)(p % kInFlight);
      if (live[slot]) {
        futs[slot].Wait();
        if (futs[slot].get() == nullptr || futs[slot]->GetReturnCode() != 0) {
          std::fprintf(stderr, "seed failed\n");
          std::exit(1);
        }
        live[slot] = false;
      }
      std::memset(bufs[slot].data(), 0, bufs[slot].size());
      const u64 lo = p * pe, hi = std::min(lo + pe, (u64)host.size());
      std::memcpy(bufs[slot].data(), host.data() + lo, (hi - lo) * esz);
      char nm[32]; gv::PageBlobName(p, nm);
      futs[slot] = core.AsyncPutBlob(vec.TagId(), std::string(nm), 0,
                                     bufs[slot].size(), bufs[slot].data(), 1.0f);
      live[slot] = true;
    }
    for (int q = 0; q < kInFlight; ++q) {
      if (!live[q]) continue;
      futs[q].Wait();
      if (futs[q].get() == nullptr || futs[q]->GetReturnCode() != 0) {
        std::fprintf(stderr, "seed failed\n");
        std::exit(1);
      }
    }
  };
  seed(vcj, cjp, sizeof(int));
  seed(vxq, xq, sizeof(float));

  Sci *dsci = nullptr; float *df = nullptr; double *de = nullptr;
  unsigned long long *dp = nullptr; u64 *dscratch = nullptr;
  cudaMalloc(&dsci, scis.size() * sizeof(Sci));
  cudaMemcpy(dsci, scis.data(), scis.size() * sizeof(Sci), cudaMemcpyHostToDevice);
  cudaMalloc(&df, natoms * 4 * sizeof(float));
  cudaMemset(df, 0, natoms * 4 * sizeof(float));
  cudaMalloc(&de, sizeof(double)); cudaMemset(de, 0, sizeof(double));
  cudaMalloc(&dp, sizeof(unsigned long long)); cudaMemset(dp, 0, sizeof(unsigned long long));
  cudaMalloc(&dscratch, (u64)blocks * kScratchU64PerBlock * sizeof(u64));

  NbParams prm{cutoff * cutoff, 1.0f, 1.0f};
  auto dcj = vcj.GetDevice(0);
  auto dxq = vxq.GetDevice(0);
  using clock = std::chrono::high_resolution_clock;
  const auto t0 = clock::now();
  YieldRunner runner(blocks, threads);
  const u32 rounds = runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> v,
                                    gy::YieldStackView sv) {
    NbKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dcj, dxq, df, dsci,
                                              (int)scis.size(), prm, de, dp,
                                              dscratch, blocks, v, sv);
  });
  const cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) { std::fprintf(stderr, "LAUNCH FAILED: %s\n",
                                        cudaGetErrorString(le)); return 1; }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "kernel failed: %s\n",
                 cudaGetErrorString(cudaGetLastError())); return 1; }
  const double ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();

  std::vector<float> hf(natoms * 4);
  double he = 0.0; unsigned long long hp = 0;
  cudaMemcpy(hf.data(), df, hf.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&he, de, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&hp, dp, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

  const auto sc1 = vcj.ReadStats(0);
  const auto sc2 = vxq.ReadStats(0);
  std::printf("  nb: %.1f ms rounds=%u | list faults=%llu evicts=%llu | "
              "xq faults=%llu evicts=%llu | get_err=%llu | pairs=%llu\n",
              ms, rounds, (unsigned long long)sc1.faults,
              (unsigned long long)sc1.evicts, (unsigned long long)sc2.faults,
              (unsigned long long)sc2.evicts,
              (unsigned long long)(sc1.get_errors + sc2.get_errors),
              (unsigned long long)hp);
  if (sc1.get_errors || sc2.get_errors) {
    std::fprintf(stderr, "FAIL: failed page reads\n"); return 1; }

  // NEWTON'S THIRD LAW: an INDEPENDENT invariant. The list is symmetric, so
  // every pair is evaluated from both sides and the total force must cancel.
  // A reference sharing the kernel's arithmetic cannot catch an error in that
  // arithmetic; this can.
  double fsum[3] = {0, 0, 0}, fmag = 0.0, fabs_sum = 0.0;
  for (u64 a = 0; a < natoms; ++a) {
    for (int d = 0; d < 3; ++d) {
      fsum[d] += hf[a * 4 + d];
      fabs_sum += std::fabs((double)hf[a * 4 + d]);
    }
    fmag = std::max(fmag, (double)std::fabs(hf[a * 4 + 0]));
  }
  const double fres = std::sqrt(fsum[0]*fsum[0] + fsum[1]*fsum[1] + fsum[2]*fsum[2]);
  // RELATIVE, not absolute. |sum f| is a cancelling sum of N float forces, so
  // its absolute size grows with the system while the underlying error per
  // term does not. Measured across three sizes, |sum f| / sum|f| sits at
  // 4.5e-08, 5.6e-08 and 6.3e-08 for 4k, 262k and 4.2M atoms -- flat, and
  // plainly float epsilon. An absolute threshold (the first version used
  // 1e-3 * peak) has no N in it and therefore fails every large run for
  // reasons that have nothing to do with the kernel.
  const double fratio = fres / std::max(fabs_sum, 1e-30);
  std::printf("  newton3: |sum f| = %.4e  sum|f| = %.4e  ratio = %.3e\n",
              fres, fabs_sum, fratio);

  bool ok = true;
  if (verify) {
    std::vector<double> rf(natoms * 3, 0.0);
    double re = 0.0;
    for (const Sci &sc : scis) {
      for (int e = sc.cjPackedBegin; e < sc.cjPackedEnd; ++e) {
        const int *ent = &cjp[(size_t)e * kCjPackedInts];
        for (int q = 0; q < kJGroupSize; ++q) {
          const int cj = ent[q];
          if (cj < 0) continue;
          for (int ii = 0; ii < kAtomsPerSc; ++ii) {
            const int bit = q * kClustersPerSc + (ii / kClusterSize);
            const unsigned w = (unsigned)ent[kJGroupSize];
            if (!((w >> bit) & 1u)) continue;
            const u64 ia = (u64)sc.sci * kAtomsPerSc + ii;
            for (int jj = 0; jj < kClusterSize; ++jj) {
              const u64 ja = (u64)cj * kClusterSize + jj;
              if (ja == ia) continue;
              float d[3];
              for (int t = 0; t < 3; ++t) d[t] = xq[ia*4+t] - xq[ja*4+t];
              const float rsq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
              if (rsq >= prm.cutoffSq || rsq == 0.0f) continue;
              float fs, en; LjPair(rsq, prm, &fs, &en);
              re += 0.5 * en;
              for (int t = 0; t < 3; ++t) rf[ia*3+t] += d[t] * fs;
            }
          }
        }
      }
    }
    double maxd = 0.0, peak = 0.0;
    for (u64 a = 0; a < natoms; ++a)
      for (int t = 0; t < 3; ++t) {
        peak = std::max(peak, std::fabs(rf[a*3+t]));
        maxd = std::max(maxd, std::fabs(rf[a*3+t] - (double)hf[a*4+t]));
      }
    const double tol = 1e-4 * std::max(peak, 1.0);
    std::printf("  verify: max|df|=%.4e peak=%.4e tol=%.4e | E %.9g vs %.9g\n",
                maxd, peak, tol, he, re);
    ok = ok && (maxd <= tol) && (std::fabs(he - re) <= 1e-4 * std::max(std::fabs(re), 1.0));
  }
  const double ftol = 1e-6;   // relative; see above
  ok = ok && (fratio <= ftol);
  (void)fmag;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
#endif
}
#endif  // !CTP_IS_DEVICE_PASS
