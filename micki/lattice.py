"""Lattice stuff"""

from __future__ import division

from .reactants import _Thermo
import numpy as np
from ase.units import kB


class Lattice(object):
    def __init__(self, neighborlist):
        self.neighborlist = neighborlist
        self.sites = [site for site in neighborlist]

        # Sanity check the input
        self.totneighbors = {}
        for site, neighbors in neighborlist.items():
            if not isinstance(site, _Thermo):
                raise ValueError("Must be a _Thermo object")  # XXX Make this error better
            self.totneighbors[site] = 0
            for neighbor, val in neighbors.items():
                self.totneighbors[site] += val
                if neighbor not in self.neighborlist:
                    raise ValueError("Neighbor {} is unknown!".format(neighbor))
            for site in self.sites:
                if site not in neighbors:
                    neighbors[site] = 0

        # Create a neighbor list matrix
        nsites = len(self.sites)
        if nsites == 1:
            self.ratio = {self.sites[0]: 1}
            return
        nmat = np.zeros((nsites, nsites), dtype=float)
        for i, a in enumerate(self.sites):
            for j, b in enumerate(self.sites):
                nmat[i, j] = self.neighborlist[b].get(a, 0) / self.totneighbors[a]
        
        # Diagonalize to find the element ratio. Only one eigenvector should
        # have all positive values. This is the eigenvector that describes
        # the element ratio
        eigenvals, eigenvecs = np.linalg.eig(nmat)
        for i in range(nsites):
            if np.all(eigenvecs[:, i] < 0) or np.all(eigenvecs[:, i] > 0):
                ratio = np.abs(eigenvecs[:, i])
                ratio /= ratio.sum()
                self.ratio = {site: ratio[i] for i, site in enumerate(self.sites)}
                break
        else:
            print("Eigenvectors: {}".format(eigenvecs))
            raise ValueError("Failed to find the element ratio! Please "
                             "double-check your neighbor count.")

    def get_S_conf(self, sites):
        if sites is None or isinstance(sites, _Thermo) or len(sites) == 1:
            return 0
        nconfs = 1
        for i in range(1, len(sites)):
            ncount = self.neighborlist[sites[i - 1]][sites[i]]
            if ncount == 0:
                raise ValueError("This binding geometry is impossible!")
            nconfs *= ncount / self.ratio[sites[i]]
        return kB * np.log(nconfs)
