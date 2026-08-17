# src/gromacs/eternia - PME grid paging over the Clio CTE gpu_vector

The PME real-space grid, held in the Context Transfer Engine and paged into
GPU memory from inside the spread kernel, so the grid can be larger than the
GPU. Grid layout matches GROMACS exactly:

    index(ix, iy, iz) = (ix * pny + iy) * pnz + iz

with interpolation order 4, as in `c_pmeGpuOrder`.

## Status: standalone, not yet wired into GROMACS

`eternia_pme_bench.cc` is a self-contained program - paged kernel, CPU
reference, config sweep - built by its own CMake with clang++-22.

That ordering is deliberate. The LAMMPS integration wired an unverified
paged kernel into the main build first, and then spent three rounds
debugging a device coroutine through someone else's 400-file CMake while a
race hid behind it. Proving the kernel against a reference first is cheaper.

## Building and running

```
cmake -S src/gromacs/eternia -B build-eternia \
  -DCMAKE_CUDA_COMPILER=clang++-22 -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_PREFIX_PATH=<clio-inst> \
  -Diowarp-core_DIR=<clio-inst>/lib/cmake/iowarp-core
cmake --build build-eternia -j
CLIO_SERVER_CONF=clio.yaml ./build-eternia/gmx_eternia_pme_bench \
  --grid 1408 --atoms 1000000 --page-kb 256 --blocks 64 --slots 8
```

## Why the kernel is grid-major

Spreading is a scatter: each atom adds to the 4x4x4 grid points around it.
Atom-major, a block would need a different page per atom, and every Eternia
hold is block-collective - the wall the LAMMPS pair style hit.

So the loop is inverted. Each block owns a **page-aligned** slice of the
grid and walks the atoms whose x-span can reach it, found from a host-side
x-sorted atom order. Every write lands in a page the block already holds,
and no two blocks ever touch the same grid point.

**Page-aligned, not plane-aligned, is load-bearing.** Caches are per block
and writeback granularity is a page, so two blocks owning parts of one page
each cache it, each fill their own part, and each flush the whole thing.
An earlier version split the grid into 4-plane slabs while a 256 KB page held
16 planes; four blocks shared every page and 59% of the charge disappeared
with every counter reporting clean.

## Verification

Two checks, and the second matters more than the first.

**Charge conservation** is an INDEPENDENT physical invariant: the spline
weights partition unity and the atoms are placed clear of the x edge, so
every charge lands on the grid in full and `sum(grid)` must equal `sum(q)`.
A reference that shares the kernel's arithmetic cannot catch an error in that
arithmetic; this can. It is what would have caught the spline bug below.

**A CPU reference** computing the identical spread. Checked across
configurations rather than in one, because how often a fault suspends the
kernel is what changes the code path:

| grid | atoms | page | blocks | slots | max abs diff | faults / evicts |
|------|-------|------|--------|-------|--------------|-----------------|
| 64   | 20k   | 256KB| 16     | 8     | 8.9e-08      | 4 / 0           |
| 64   | 20k   | 4KB  | 16     | 4     | 8.9e-08      | 256 / 192       |
| 64   | 20k   | 4KB  | 64     | 3     | 8.9e-08      | 256 / 64        |
| 64   | 20k   | 1MB  | 4      | 2     | 8.9e-08      | 1 / 0           |
| 128  | 200k  | 64KB | 32     | 3     | 1.2e-07      | 128 / 32        |
| 128  | 200k  | 16KB | 8      | 3     | 1.2e-07      | 512 / 488       |

Charge error is ~3e-06 on 20k atoms and ~1.4e-05 on 200k, i.e. the
sqrt(N)*eps expected of that many single-precision adds.

### A caveat worth stating plainly

An earlier version of `Bspline4` did not partition unity: `w[2]` was written
in terms of `(1 - frac)` and expanded to a second copy of `w[1]`, so the four
weights summed to 1.5 at frac = 0. Every test still passed, because the CPU
reference called the same function. The paging was genuinely being tested and
those results stand; the arithmetic being paged was not a PME spread. The
charge-conservation check exists so that cannot recur.

## IMPORTANT: this does not compose into an out-of-core PME

Paging the PME real-space grid does **not** let GROMACS run a PME larger than
VRAM, and the reason is not in this code.

The PME pipeline is spread -> 3D FFT -> solve -> inverse FFT -> gather, and
`gpu_3dfft_cufft.cu` calls
`cufftExecR2C(plan, realGrid_, complexGrid_)` with raw device pointers to the
whole real and complex grids. cuFFT has no streaming or host-backed mode --
cuFFTMp and heFFTe distribute across ranks and GPUs, not to host storage. So
the stage immediately after spread needs everything spread just wrote, plus a
complex grid of comparable size. The effective ceiling is roughly 2x the real
grid whatever spread does.

What follows below is therefore a demonstration that **the spread kernel
scales past VRAM**, and a measurement of paged scatter performance. It is not
a larger-than-VRAM PME, and it should not be presented as one.

The GROMACS target that DOES compose is the **nonbonded** path: coordinates
(`xq`) and the cluster pair list, consumed by local force accumulation with
no global transform in the way. That is the analogue of the LAMMPS
integration, which works end to end for exactly this reason.

## Larger than VRAM (spread only)

On an 8 GiB (7.99 GiB usable) RTX 4070 Laptop:

| grid   | points        | size        | spread   | faults / evicts | result |
|--------|---------------|-------------|----------|-----------------|--------|
| 512^3  | 134,217,728   | 0.50 GiB    | -        | 2,048 / 1,536   | PASS   |
| 1024^3 | 1,073,741,824 | 4.00 GiB    | -        | 16,384 / 15,872 | PASS   |
| 1408^3 | 2,791,309,312 | **10.40 GiB** | 2,988 ms | 42,592 / 42,080 | PASS   |

The last row holds a grid larger than the GPU's memory and pages it
throughout, against a GPU-side cache of `blocks * slots * page` = 64 * 8 *
256 KB = 128 MB, about 1.2% of the grid. Agreement with the CPU reference is
1.49e-07, i.e. single-precision rounding.

## Spline equivalence with GROMACS

The weights this kernel uses are the closed form of the cubic cardinal
B-spline. GROMACS computes the same thing with the de Boor recursion in
`calculate_splines` (`pme_gpu_calculate_splines.cuh`). The two agree to
**2.2e-16** over dr in [0, 1], sampled at 1000 points -- machine epsilon, so
this is the same polynomial and not merely a similar one.

That closes the arithmetic half of the drop-in. What remains is the gridline
index: GROMACS derives it through `d_fractShiftsTable` /
`d_gridlineIndicesTable`, which also handle triclinic boxes and the unit-cell
shift, where this harness takes `floor(t)` directly -- equivalent for an
orthorhombic box with no shift, and not yet equivalent in general.

## Why this is not a drop-in for pme_gpu_spread as it stands

`spread_charges` consumes `theta` in a **warp-interleaved layout**
(`getSplineParamIndexBase`) whose indexing is derived from the atom's warp
position *within GROMACS's own spread block*. A paged kernel uses a
different, page-aligned decomposition, so consuming that layout would couple
it to GROMACS's block size -- precisely the sort of coupling that produces
silent near-miss results.

The cleaner seam is the `computeSplines=true, spreadCharges=true` path: take
coordinates, charges and grid geometry, compute splines internally, produce
the grid. That is what this kernel does, and it is why the spline had to
match GROMACS's exactly rather than merely be spline-shaped.

## The nonbonded path, larger than VRAM

Unlike the PME grid above, this one composes: the consumer is local force
accumulation, so nothing downstream demands the whole array back.

On an 8 GiB (7.99 GiB usable) RTX 4070 Laptop:

| atoms      | pair list     | xq       | total         | kernel  | result |
|------------|---------------|----------|---------------|---------|--------|
| 4,194,304  | 1.03 GiB      | 0.06 GiB | 1.41 GiB      | 243 s   | PASS   |
| 35,200,000 | 10.38 GiB     | 0.53 GiB | **10.90 GiB** | 2,990 s | PASS   |

The large run evaluated 712,463,819,264 pair interactions across 151,910
scheduler rounds, with 592,431 pair-list faults and 8,540,745 coordinate
faults, and zero failed page reads. Resident GPU memory during the run was
4,363 MiB against a 10.90 GiB dataset -- the GPU held about 40% of the data
at any moment.

Correctness at that size is the Newton's-third-law ratio, |sum f| / sum|f|,
which is 4.43e-08 -- in line with 1.67e-08, 3.24e-08 and 4.08e-08 at 4k,
262k and 4.2M atoms. Flat across four orders of magnitude in system size is
what an epsilon-level residual looks like; a dropped or double-counted
interaction would not sit there.

The CPU reference is not run at this size (it is O(pairs), and 712 billion of
them is not a check, it is a second experiment). The reference covers the
sweep at small sizes; the invariant covers the large ones.

## Validated inside GROMACS, against nbnxm itself

`GMX_ETERNIA=ON` links the paged kernel into libgromacs; `GMX_ETERNIA_NB=1`
runs it from `gpu_launch_kernel` alongside the production kernel, on the same
cluster pair list. On a 1728-atom argon LJ fluid (no charges, no bonds, no
LJ modifier, rvdw = 1.0):

    [eternia] atoms=1728 sci=111 cjPacked=266 | list faults=64 evicts=0
              | xq faults=64 evicts=0 | get_err=0 | pairs=185760
              | E=-8491.76907

    GROMACS LJ (SR) = -8491.76

Reaching that took three corrections, and every one was a semantic difference
between this kernel's assumptions and nbnxm's, invisible to the standalone
bench because its reference shared those assumptions:

1. **Two mask words, not one.** nbnxm splits a cluster pair across two warps
   (`imei[c_clusterPairSplit]`) and each warp owns half the j-cluster's
   atoms, so the words can differ once pruning has run. (On a fresh list they
   are identical, so this one did not move the energy -- it is still wrong to
   ignore.)

2. **Each unordered pair is listed once, not twice.** A pair of distinct
   clusters appears in one direction only, so halving the energy -- as this
   kernel did, assuming a full list -- undercounts by two. The self-cluster
   is the exception and must be masked to the triangle j > i, which is
   nbnxm's `nonSelfInteraction | (ci != cj)`.

3. **Periodic shifts.** Each i-supercluster entry carries a shift index and
   nbnxm adds `shift_vec[shift]` to the i-atom coordinates. Ignoring it makes
   every non-central entry compute distances for the wrong periodic image,
   and those pairs fall outside the cutoff and vanish. This was most of the
   list, not an edge case: 111 sci entries for 27 superclusters is about four
   shifts each.

The progression was -4610.08, then -7097.99, then -8491.77 against a
reference of -8491.76.

## Next

Wire this into GROMACS proper: replace `pme_gpu_spread` and the matching
gather with the paged kernels behind a runtime switch, and validate against
`gmx nonbonded-benchmark` / a PME regression test rather than a synthetic
reference.
