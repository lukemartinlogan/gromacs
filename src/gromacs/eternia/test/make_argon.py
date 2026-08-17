#!/usr/bin/env python3
"""Generate a monatomic argon LJ box for the eternia comparison.

Argon because the paged kernel computes plain Lennard-Jones: no charges, no
bonds, and therefore no electrostatics and no per-atom exclusions, which is
the case where it and nbnxm should agree exactly. Any other system would be
comparing different physics.
"""
import sys

n_side = int(sys.argv[1]) if len(sys.argv) > 1 else 12
spacing = 0.34                      # nm, near argon's sigma
box = n_side * spacing
n = n_side ** 3

with open("conf.gro", "w") as f:
    f.write("argon %d\n%d\n" % (n, n))
    i = 0
    for x in range(n_side):
        for y in range(n_side):
            for z in range(n_side):
                i += 1
                # gro fields are fixed 5 wide; indices wrap past 99999
                w = i % 100000
                f.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"
                        % (w, "AR", "Ar", w, x * spacing, y * spacing, z * spacing))
    f.write("%10.5f%10.5f%10.5f\n" % (box, box, box))

with open("topol.top") as f:
    top = f.read()
import re
top = re.sub(r"^Ar\s+\d+", "Ar   %d" % n, top, flags=re.M)
with open("topol.top", "w") as f:
    f.write(top)
print("atoms %d  box %.2f nm" % (n, box))
