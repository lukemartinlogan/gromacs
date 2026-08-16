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
/** Ints per packed entry, since the vector is paged as int. */
constexpr int kCjPackedInts = kJGroupSize + 2;

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
  u64 *pg_lo_s = scratch + static_cast<u64>(block) * 2;
  u64 *pg_hi_s = pg_lo_s + 1;

  for (int s = block; s < numSci; s += nblocks) {
    const Sci sc = scis[s];
    const u64 i0 = static_cast<u64>(sc.sci) * kAtomsPerSc;

    // The i-supercluster's coordinates, held once for the whole sci. A
    // separate view so the j-side holds below cannot dislodge it.
    gv::DeviceVector<float> xi = xq;
    co_await xi.HoldPageCoro(i0 * 4, static_cast<u64>(kAtomsPerSc) * 4, &run);

    // Stream this sci's slice of the packed list, page by page.
    const u64 k0 = static_cast<u64>(sc.cjPackedBegin) * kCjPackedInts;
    const u64 k1 = static_cast<u64>(sc.cjPackedEnd) * kCjPackedInts;
    const u64 pe = cjp.h_->elems_per_page_;

    for (u64 off = k0; off < k1; ) {
      const u64 seg_end = ((off / pe) + 1) * pe < k1 ? ((off / pe) + 1) * pe : k1;
      co_await cjp.HoldPageCoro(off, seg_end - off, &run);

      // Whole packed entries fully inside this page. An entry straddling the
      // boundary is left to the segment that contains its start, which is
      // why the loop below re-derives the entry index rather than assuming
      // alignment.
      const u64 first = (off + kCjPackedInts - 1) / kCjPackedInts;
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

      // PASS B: hold each coordinate page once, block-collectively, and let
      // every thread evaluate the pairs of its own i-atom landing in it.
      if (pg_lo != ~0ull) {
        for (u64 pg = pg_lo; pg <= pg_hi; ++pg) {
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
              const unsigned imask = static_cast<unsigned>(
                  cjp.at(base + kJGroupSize + (ii / 32)));
              const u32 icl = ii / kClusterSize;
              if (!((imask >> (jslot * kClustersPerSc + icl)) & 1u)) continue;

              const u64 ia = i0 + ii;
              const float ix = xi.at(ia * 4 + 0);
              const float iy = xi.at(ia * 4 + 1);
              const float iz = xi.at(ia * 4 + 2);

              for (int jj = 0; jj < kClusterSize; ++jj) {
                const u64 ja = static_cast<u64>(cj) * kClusterSize + jj;
                if (ja == ia) continue;
                const u64 jo = ja * 4;
                if (jo < xlo || jo + 3 >= xhi) continue;  // another page
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
int main(int argc, char **argv) {
  std::printf("eternia nbnxm harness: scaffolding in place, kernel written.\n"
              "Driver, CPU reference and sweep are the next step.\n");
  (void)argc; (void)argv;
  return 0;
}
#endif
