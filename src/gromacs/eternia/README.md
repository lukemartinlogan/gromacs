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

Against a CPU reference computing the identical spread. Checked across
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

## Larger than VRAM

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

## Next

Wire this into GROMACS proper: replace `pme_gpu_spread` and the matching
gather with the paged kernels behind a runtime switch, and validate against
`gmx nonbonded-benchmark` / a PME regression test rather than a synthetic
reference.
