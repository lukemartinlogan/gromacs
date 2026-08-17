#!/bin/sh
# Compare the eternia paged nonbonded kernel against nbnxm's own, on nbnxm's
# own pair list, inside a running mdrun.
#
#   GMX=<gmx binary> CLIO_SERVER_CONF=<clio.yaml> ./compare.sh [cells] [nsteps]
#
# The paged kernel runs ALONGSIDE the production kernel and prints its energy;
# GROMACS prints its own. They should agree to GROMACS's printed precision.
#
# Note: the hook writes to stderr, and the Clio runtime currently segfaults at
# teardown (IpcGpu2Cpu::RecvIn) AFTER results are produced, which loses
# unflushed stderr. Running under gdb is the reliable way to see the output.
set -e
# A correctly sized Clio config ships next to this script. Defaulting to it
# matters more than it looks: the hbm tier's capacity_limit is preallocated on
# the GPU, so borrowing a config with a large tier makes the paged path appear
# to cost far more memory than it saves.
: "${CLIO_SERVER_CONF:=$(cd "$(dirname "$0")" && pwd)/clio.yaml}"
export CLIO_SERVER_CONF
: "${GMX:?set GMX to the gmx binary}"
CELLS=${1:-12}
NSTEPS=${2:-10}

# Both inputs are copied into the working directory, not read in place:
# make_argon.py reads topol.top from the CURRENT directory and rewrites it with
# the atom count it just generated, so running the script from anywhere other
# than its own directory failed on a missing topol.top -- and running it from
# its own directory would edit a tracked file.
cp "$(dirname "$0")/topol.top" topol.top
python3 "$(dirname "$0")/make_argon.py" "$CELLS"
cp "$(dirname "$0")/md10.mdp" md.mdp
sed -i "s/^nsteps .*/nsteps          = $NSTEPS/" md.mdp
"$GMX" grompp -f md.mdp -c conf.gro -p topol.top -o t.tpr -maxwarn 5 >/dev/null 2>&1

# The exact answer, before either kernel gets a say. GROMACS is not a
# sufficient reference on its own at large pair counts -- at 216,000 atoms its
# mixed-precision accumulation is 9.8e-05 per atom off the truth, 100x worse
# than the paged kernel -- so a two-way comparison there would convict the
# wrong side.
# STEP 0 ONLY: the lattice sum is exact for the generated configuration, and
# gen-vel puts the atoms in motion immediately, so it says nothing about the
# rows below the first.
echo "== exact (lattice sum, double) -- step 0 only =="
python3 "$(dirname "$0")/exact_lattice.py" --atoms "$(python3 -c "print($CELLS**3)")" |
  sed -n 's/^exact total.*: /  LJ(SR) = /p'

echo "== GROMACS =="
"$GMX" mdrun -s t.tpr -nb gpu -ntmpi 1 -ntomp 4 -deffnm ref -nsteps "$NSTEPS" >/dev/null 2>&1
grep -A3 "Energies (kJ/mol)" ref.log | grep -E "^ +-?[0-9]" | awk '{print "  LJ(SR) = "$1}'

echo "== eternia (paged) =="
# argon: sigma = 0.3345 nm, epsilon = 1.045128 kJ/mol
C6=$(python3 -c "s=0.3345;e=1.045128;print(4*e*s**6)")
C12=$(python3 -c "s=0.3345;e=1.045128;print(4*e*s**12)")
GMX_ETERNIA_NB=1 GMX_ETERNIA_C6="$C6" GMX_ETERNIA_C12="$C12" GMX_ETERNIA_RC=1.0 \
GMX_ETERNIA_PAGE_KB=${PAGE_KB:-4} GMX_ETERNIA_BLOCKS=${BLOCKS:-32} \
GMX_ETERNIA_SLOTS=${SLOTS:-2} \
  gdb -batch -ex run --args "$GMX" mdrun -s t.tpr -nb gpu -ntmpi 1 -ntomp 2 \
      -deffnm etn -nsteps "$NSTEPS" 2>&1 | grep -oE "E=[-0-9.]+" | sed 's/^/  /'
