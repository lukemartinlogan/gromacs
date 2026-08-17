/* -*- c++ -*- ----------------------------------------------------------
   Eternia paged nonbonded kernel for GROMACS -- HOST-SIDE BOUNDARY HEADER.

   Mentions neither CUDA nor Clio: the paging kernel suspends on a page fault
   using C++20 device coroutines, which only clang++-22 -x cuda compiles,
   while GROMACS builds its CUDA with nvcc, which rejects co_await in device
   code. The device half is its own ExternalProject with its own toolchain.

   WHAT THIS IS, AND WHAT IT IS NOT
   -------------------------------
   This is a VALIDATION path, not a replacement for nbnxm's production
   kernel. It computes plain Lennard-Jones over GROMACS's own cluster pair
   list, with the pair list and coordinates held out of core and paged into
   the GPU from inside the kernel.

   It deliberately does NOT implement electrostatics, per-atom exclusions
   beyond the cluster imask, LJ switch/shift modifiers, or the free-energy
   paths. Substituting it for the production kernel on a system that uses any
   of those would silently produce wrong forces, which is why it is wired as
   a comparison rather than a substitution: GROMACS runs its own kernel, this
   runs alongside on the same pair list, and the caller compares.

   A monatomic Lennard-Jones fluid -- argon, no charges, no bonds -- is the
   case where the two should agree, and that is what the comparison is for.

   WHY THE NONBONDED PATH AND NOT PME
   ----------------------------------
   Paging only lifts the VRAM ceiling if the CONSUMER can stream too. PME
   fails that: the stage after spread is a cuFFT 3D transform that needs the
   whole real and complex grids resident. Nonbonded passes: its consumer is
   local force accumulation.
------------------------------------------------------------------------- */

#ifndef GMX_ETERNIA_NB_H
#define GMX_ETERNIA_NB_H

#include <cstdint>

namespace eternia_gmx {

/** Cluster geometry, mirroring nbnxm's GPU layout constants. */
struct ClusterLayout {
  int clusterSize = 8;        //!< atoms per cluster
  int clustersPerSc = 8;      //!< clusters per supercluster
  int jGroupSize = 4;         //!< j-clusters per packed entry
};

struct Config {
  const char* tag = "gmx_eternia_nb";
  int gpu_id = 0;
  std::uint64_t page_bytes = 262144;
  std::uint32_t nblocks = 64;
  std::uint32_t nthreads = 128;
  std::uint32_t slots = 8;    //!< resident pages per block
  /** nbnxm's central (self-image) shift index. A cluster pair with this
   *  shift whose i- and j-clusters coincide is the diagonal, and must be
   *  counted triangularly or every intra-cluster pair is counted twice. */
  int centralShift = -1;
  bool stats = false;
};

struct Stats {
  std::uint64_t list_faults = 0;
  std::uint64_t list_evicts = 0;
  std::uint64_t xq_faults = 0;
  std::uint64_t xq_evicts = 0;
  std::uint64_t get_errors = 0;
  std::uint64_t pairs = 0;
};

/** One i-supercluster entry, mirroring nbnxm_sci_t. */
struct Sci {
  int sci;
  int shift;
  int cjPackedBegin;
  int cjPackedEnd;
};

struct Context;

/**
 * @param natoms   total atoms the coordinate array covers
 * @param numSci   number of i-supercluster entries
 * @param numCjPacked  number of packed j-cluster entries
 */
Context* Create(const Config& cfg, const ClusterLayout& lay, int natoms,
                int numSci, int numCjPacked);
void Destroy(Context* ctx);

/**
 * Copy the pair list and coordinates into the CTE, page by page.
 *
 * @param cj        numCjPacked * jGroupSize ints, the j-cluster indices
 * @param imask     numCjPacked * 2 words: nbnxm splits a cluster pair across
 *                  two warps (imei[c_clusterPairSplit]) and each warp owns
 *                  half of the j-cluster's atoms, so BOTH words are needed
 * @param xq        natoms * 4 floats, x/y/z/charge, as nbnxm stores it
 *
 * Page by page so the host never needs a second resident copy of an array
 * that by assumption does not fit in GPU memory.
 */
bool Upload(Context* ctx, const int* cj, const unsigned* imask,
            const float* xq, const Sci* scis);

/**
 * Compute plain LJ forces over the uploaded list into `forces`
 * (natoms * 4 floats, device memory), and return the potential energy.
 *
 * @param c6, c12   Lennard-Jones parameters, single type pair
 * @param cutoffSq  squared cutoff
 */
/**
 * @param shiftVec_device  numShifts * 3 floats on the device. nbnxm gives each
 *   i-supercluster entry a periodic shift index and the kernel adds
 *   shift_vec[shift] to the i-atom coordinates; without it every non-central
 *   entry computes distances for the wrong periodic image, and those pairs
 *   silently fall outside the cutoff. A 1728-atom box produced 111 sci
 *   entries for 27 superclusters -- about four shifts each -- so this is most
 *   of the list, not an edge case.
 */
bool Compute(Context* ctx, float c6, float c12, float cutoffSq,
             const float* shiftVec_device, float* forces_device,
             double* energy_out);

Stats GetStats(Context* ctx);
const char* LastError();
bool Available();

}  // namespace eternia_gmx

#endif
