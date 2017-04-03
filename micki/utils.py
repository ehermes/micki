#!/usr/bin/env python

from __future__ import division

import numpy as np
from ase.data import vdw_radii

def calculate_avg_vdw_radius(atoms, npoints=8001):
    if npoints % 2 == 0:
        npoints += 1

    vecs = np.zeros((npoints, 3))

    # Create a Fibonacci spiral
    offset = 2 / npoints
    increment = np.pi * (3 - np.sqrt(5))
    for i in range(npoints):
        y = i * offset - 1 + offset/2
        r = np.sqrt(1 - y**2)
        phi = (i + 1 % npoints) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        vecs[i] = np.array([x, y, z])

    natoms = len(atoms)
    pos = atoms.get_positions()
    pos -= atoms.get_center_of_mass()
    rad = np.array([vdw_radii[atom.number] for atom in atoms])

    Rmol = np.zeros(npoints)
    for i, vec in enumerate(vecs):
        for j in range(natoms):
            Xparr = vec * np.dot(pos[j], vec)
            Xperp = pos[j] - Xparr
            Ratom = rad[j]
	    Rperp = np.linalg.norm(Xperp)
	    # Skip atom if line doesn't intersect its vdW sphere
	    if Rperp > Ratom:
	        continue
	    Rparr = np.linalg.norm(Xparr) * np.sign(np.dot(pos[j], vec))
	    b = -2 * Rperp
	    c = Rperp**2 - Ratom**2
	    d1 = np.abs((-b + np.sqrt(b**2 - 4 * c))/2)
	    d2 = np.abs((-b - np.sqrt(b**2 - 4 * c))/2)
	    if d1 > Ratom and d2 < Ratom:
	        d = d2
	    elif d1 < Ratom and d2 > Ratom:
	        d = d1
	    elif d1 > Ratom and d2 > Ratom:
	        raise ValueError, 'Error! Both distances greater than Ratom!'
	    elif d1 < Ratom and d2 < Ratom:
	        raise ValueError, 'Error! Both distances less than Ratom!'
            else:
                raise RuntimeError, "This should be impossible! d1 = {}, d2 = {}, Ratom = {}".format(d1, d2, Ratom)
	    if Rparr + d > Rmol[i]:
	        Rmol[i] = Rparr + d
    return np.average(Rmol)
