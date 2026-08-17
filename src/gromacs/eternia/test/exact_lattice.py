#!/usr/bin/env python3
"""Exact per-atom LJ energy for the argon lattice, in double precision.

WHY THIS EXISTS
---------------
compare.sh checks the paged kernel against GROMACS. That is a two-way
comparison, and a two-way comparison can agree while both sides are wrong --
which is not hypothetical here. At 216,000 atoms the two disagreed by 2e-5,
and the question of which one was right could not be answered from the
comparison itself.

It could be answered exactly. make_argon.py builds a PERFECT cubic lattice, and
md10.mdp asks for plain truncated LJ (vdw-modifier = none, DispCorr = no), so
every atom has an identical environment and the per-atom energy is a finite sum
over the neighbours inside the cutoff. No simulation required, and no floating
point beyond what a few hundred terms need.

The answer was that the paged kernel was right to the precision it prints at
(9.3e-07) and GROMACS was 100x further off (9.8e-05), because a
mixed-precision build accumulates 23.2 million pair terms in float. So this
also documents the limit of using GROMACS as the reference at large pair
counts: below roughly 10^6 pairs its error hides under the printed precision,
above that it does not.

USAGE
-----
    ./exact_lattice.py [spacing] [sigma] [epsilon] [rcut]

Defaults match make_argon.py and md10.mdp. Prints the per-atom energy, and the
total for a given atom count if one is supplied via --atoms.
"""
import argparse
import math

p = argparse.ArgumentParser()
p.add_argument("--spacing", type=float, default=0.34, help="lattice spacing, nm")
p.add_argument("--sigma", type=float, default=0.3345, help="LJ sigma, nm")
p.add_argument("--epsilon", type=float, default=1.045128, help="LJ epsilon, kJ/mol")
p.add_argument("--rcut", type=float, default=1.0, help="cutoff, nm")
p.add_argument("--atoms", type=int, default=0, help="also print the total for N atoms")
a = p.parse_args()

# Every lattice vector shorter than the cutoff. The cutoff must fit inside half
# the box for this to be the whole story; make_argon.py satisfies that for any
# n_side >= 6 at these defaults.
lim = int(a.rcut / a.spacing) + 2
terms = []
for i in range(-lim, lim + 1):
    for j in range(-lim, lim + 1):
        for k in range(-lim, lim + 1):
            if i == j == k == 0:
                continue
            r2 = (a.spacing * a.spacing) * (i * i + j * j + k * k)
            if r2 < a.rcut * a.rcut:
                s6 = (a.sigma * a.sigma / r2) ** 3
                terms.append(4.0 * a.epsilon * (s6 * s6 - s6))

# fsum, and smallest first, so the reference is not itself limited by the
# summation order it exists to check.
terms.sort(key=abs)
per_atom = 0.5 * math.fsum(terms)

print("neighbours within %.3f nm : %d" % (a.rcut, len(terms)))
print("exact per-atom energy     : %.6f kJ/mol" % per_atom)
if a.atoms:
    print("exact total for %d atoms : %.4f kJ/mol" % (a.atoms, per_atom * a.atoms))
    print("pairs within cutoff       : %d" % (a.atoms * len(terms) // 2))
