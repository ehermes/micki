"""Reference atoms object"""

import numpy as np

from ase.io import read


class EnergyReference(object):
    def __init__(self, species):
        symbols = []
        energies = []
        elements = set()

        for sp in species:
            conf = read(sp, index=0)
            symbols.append(conf.get_chemical_symbols())
            elements = elements.union(symbols[-1])
            energies.append(conf.get_potential_energy())

        elements = list(elements)
        energies = np.array(energies, dtype=float)
        size = len(elements)
        if len(energies) < size:
            raise ValueError, "System is underdetermined!"
        elif len(energies) > size:
            raise ValueError, "System is overdetermined!"

        coeff = np.zeros((size, size), dtype=float)

        for i in xrange(size):
            for j in xrange(size):
                coeff[i, j] = symbols[i].count(elements[j])
        eref = np.linalg.solve(coeff, energies)

        self.references = {}
        for i, symbol in enumerate(elements):
            self.references[symbol] = eref[i]

    def __repr__(self):
        return self.references.__repr__()

    def __len__(self):
        return len(self.references)

    def __getitem__(self, key):
        return self.references[key]

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def __contains__(self, key):
        return key in self.references

    def __iter__(self):
        for ref in self.references:
            yield ref
