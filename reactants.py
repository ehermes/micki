"""This module contains object definitions of species
and collections of species"""

import numpy as np

from ase.io import read

from ase.units import J, kJ, mol, _hplanck, m, kg, _k, kB, _c

from masses import masses


class _Thermo(object):
    """Generic thermodynamics object"""
    def __init__(self, outcar, linear=False, symm=1, spin=0., ts=False, \
            label=None, eref=None, metal=None):
        self.outcar = outcar
        self.atoms = read(self.outcar, index=0)
        self.mass = [masses[atom.symbol] for atom in self.atoms]
        self.atoms.set_masses(self.mass)
        self.eref = eref
        self.metal = metal
        self._read_hess()
        self.T = None
        self.potential_energy = self.atoms.get_potential_energy()
        self.linear = linear
        self.symm = symm
        self.spin = spin
        self.ts = ts
        self.label = label
        self.qtot = None
        self.qelec = None
        self.qtrans2D = None
        self.qtrans3D = None
        self.qrot = None
        self.qvib = None
        self.Stot = None
        self.Selec = None
        self.Strans2D = None
        self.Strans3D = None
        self.Srot = None
        self.Svib = None
        self.Etot = None
        self.Eelec = None
        self.Etrans2D = None
        self.Etrans3D = None
        self.Erot = None
        self.Evib = None
        self.Htot = None

        self.scale = {}
        self.scale_params = ['Stot', 'Selec', 'Strans2D', 'Strans3D', 'Srot', 'Svib', \
                'Etot', 'Eelec', 'Etrans2D', 'Etrans3D', 'Erot', 'Evib', 'Htot']
        for param in self.scale_params:
            self.scale[param] = 1.0
        self.scale_old = self.scale.copy()

        if self.eref is not None:
            for symbol in self.atoms.get_chemical_symbols():
                self.potential_energy -= self.eref[symbol]

    def update(self, T=None):
        if not self.is_update_needed(T):
            return

        if T is None:
            T = self.T

        self.T = T
        self._calc_q(T)
        self.scale_old = self.scale.copy()

    def is_update_needed(self, T):
        needed = True
        while needed:
            if self.qtot is None:
                break
            if T is not None and T != self.T:
                break
            if np.any([self.scale[param] != self.scale_old[param] \
                    for param in self.scale_params]):
                break
            needed = False
        return needed

    def get_H(self, T=None):
        self.update(T)
        return self.Htot * self.scale['Htot']

    def get_S(self, T=None):
        self.update(T)
        return self.Stot * self.scale['Stot']

    def get_E(self, T=None):
        self.update(T)
        return self.Etot * self.scale['Etot']

    def get_q(self, T=None):
        self.update(T)
        return self.qtot

    def get_reference_state(self):
        raise NotImplementedError

    def get_scale(self, param):
        try:
            return self.scale[param]
        except KeyError:
            print "{} is not a valid scaling parameter name!".format(param)
            return None

    def set_scale(self, param, value):
        try:
            self.scale[param] = value
        except KeyError:
            print "{} is not a valid scaling parameter name!".format(param)

    def _calc_q(self, T):
        raise NotImplementedError

    def _calc_qtrans2D(self, T):
        mtot = sum(self.mass) / kg
        self.qtrans2D = 2 * np.pi * mtot * _k * T / _hplanck**2 / self.rho0
        self.Etrans2D = kB * T * self.scale['Etrans2D']
        self.Strans2D = kB * (2 + np.log(self.qtrans2D)) * self.scale['Strans2D']

    def _calc_qtrans3D(self, T):
        mtot = sum(self.mass) / kg
        self.qtrans3D = (2 * np.pi * mtot * _k * T / _hplanck**2)**(3./2.) / self.rho0
        self.Etrans3D = 3. * kB * T / 2. * self.scale['Etrans3D']
        self.Strans3D = kB * (5./2. + np.log(self.qtrans3D)) * self.scale['Strans3D']

    def _calc_qrot(self, T):
        com = self.atoms.get_center_of_mass()
        if self.linear:
            I = 0
            for atom in self.atoms:
                I += atom.mass * np.linalg.norm(atom.position - com)**2
            I /= (kg * m**2)
            self.qrot = 8 * np.pi**2 * I * _k * T / (_hplanck**2 * self.symm)
            self.Erot = kB * T * self.scale['Erot']
            self.Srot = kB * (1. + np.log(self.qrot)) * self.scale['Srot']
        else:
            I = self.atoms.get_moments_of_inertia() / (kg * m**2)
            thetarot = _hplanck**2 / (8 * np.pi**2 * I * _k)
            self.qrot = np.sqrt(np.pi * T**3 / np.prod(thetarot)) / self.symm
            self.Erot = 3. * kB * T / 2. * self.scale['Erot']
            self.Srot = kB * (3./2. + np.log(self.qrot)) * self.scale['Srot']

    def _calc_qvib(self, T, ncut=0):
        thetavib =100 * _c * _hplanck * self.freqs[ncut:] / _k
        self.qvib = np.prod(np.exp(-thetavib/(2. * T)) / (1. - np.exp(-thetavib/T)))
        self.Evib = kB * sum(thetavib * (1./2. + 1./(np.exp(thetavib/T) - 1.))) \
                * self.scale['Evib']
        self.Svib = kB * sum((thetavib/T)/(np.exp(thetavib/T) - 1.) \
                - np.log(1. - np.exp(-thetavib/T))) * self.scale['Svib']

    def _calc_qelec(self, T):
        self.qelec = 2. * self.spin + 1. * np.exp(-self.potential_energy / (_k * T))
        self.Eelec = self.potential_energy * self.scale['Eelec']
        self.Selec = kB * np.log(2. * self.spin + 1.) * self.scale['Selec']

    def _read_hess(self):
        # This reads the hessian from the OUTCAR and diagonalizes it
        # to find the frequencies, rather than reading the frequencies
        # directly from the OUTCAR. This is to ensure we use the same
        # unit conversion factors, and also to make sure we use the same
        # atom masses for all calculations. Also, allows for the possibility
        # of doing partial hessian diagonalization should we want to do that.
        hessblock = 0
        with open(self.outcar, 'r') as f:
            for line in f:
                line = line.strip()
                if line != '':
                    if hessblock == 1:
                        if line.startswith('---'):
                            hessblock = 2

                    elif hessblock == 2:
                        line = line.split()
                        dof = len(line)
                        self.hess = np.zeros((dof, dof), dtype=float)
                        index = np.zeros(dof, dtype=int)
                        cart = np.zeros(dof, dtype=int)
                        for i, direction in enumerate(line):
                            index[i] = int(direction[:-1]) - 1
                            if direction[-1] == 'X':
                                cart[i] = 0
                            elif direction[-1] == 'Y':
                                cart[i] = 1
                            elif direction[-1] == 'Z':
                                cart[i] = 2
                            else:
                                raise ValueError, "Error reading Hessian!"
                        hessblock = 3
                        j = 0

                    elif hessblock == 3:
                        line = line.split()
                        self.hess[j] = np.array([float(val) for val in line[1:]], \
                                dtype=float)
                        j += 1

                    elif line.startswith('SECOND DERIVATIVES'):
                        hessblock = 1

                elif hessblock == 3:
                    break

        # Temporary work around: My test system OUTCARs include some
        # metal atoms in the hessian, this seems to cause some problems
        # with the MKM. So, here I'm taking only the non-metal part
        # of the hessian and diagonalizing that.
        nonmetal = []
        for i, j in enumerate(index):
            if self.atoms[j].symbol != self.metal:
                nonmetal.append(i)
        if not nonmetal:
            self.freqs = np.array([])
            return
        newhess = np.zeros((len(nonmetal), len(nonmetal)))
        newindex = np.zeros(len(nonmetal), dtype=int)
        for i, a in enumerate(nonmetal):
            newindex[i] = index[a]
            for j, b in enumerate(nonmetal):
                newhess[i, j] = self.hess[a, b]
        self.hess = newhess
        index = newindex

        self.hess = -(self.hess + self.hess.T) / 2.
        mass = np.array([self.atoms[i].mass for i in index], dtype=float)
        self.hess /= np.sqrt(np.outer(mass, mass))
        self.hess *= _hplanck**2 * J * m**2 * kg / (4 * np.pi**2)
        v, w = np.linalg.eig(self.hess)

        # We're taking the square root of an array that could include
        # negative numbers, so the result has to be complex.
        freq = np.sqrt(np.array(v, dtype=complex))
        self.freqs = np.zeros_like(freq, dtype=float)

        # We don't want to deal with complex numbers, so we just convert
        # imaginary numbers to negative reals.
        for i, val in enumerate(freq):
            if val.imag == 0:
                self.freqs[i] = val.real
            else:
                self.freqs[i] = -val.imag
        self.freqs.sort()

    def copy(self):
        raise NotImplementedError

    def __repr__(self):
        if self.label is not None:
            return self.label
        else:
            return self.atoms.get_chemical_formula()

    def __add__(self, other):
        return _Reactants([self, other])

    def __mul__(self, factor):
        assert isinstance(factor, int)
        return _Reactants([self for i in xrange(factor)])

    def __rmul__(self, factor):
        return self.__mul__(factor)


class _Fluid(_Thermo):
    def __init__(self, outcar, linear=False, symm=1, spin=0., \
            label=None, eref=None, rhoref=1.):
        super(_Fluid, self).__init__(outcar, linear, symm, \
                spin, False, label, eref, None)
        self.rho0 = rhoref
        assert np.all(self.freqs[6:] > 0), "Imaginary frequencies found!"

    def get_reference_state(self):
        return self.rho0

    def _calc_q(self, T):
        raise NotImplementedError

    def copy(self):
        return self.__class__(self.outcar, self.linear, self.symm, self.spin, \
                self.label, self.eref)


class Gas(_Fluid):
    def _calc_q(self, T):
        self._calc_qelec(T)
        self._calc_qtrans3D(T)
        self._calc_qrot(T)
        self._calc_qvib(T, ncut=7 if self.ts else 6)
        self.qtot = self.qelec * self.qtrans3D * self.qrot * self.qvib
        self.Etot = self.Eelec + self.Etrans3D + self.Erot + self.Evib
        self.Htot = self.Etot + _k * T
        self.Stot = self.Selec + self.Strans3D + self.Srot + self.Svib


class Liquid(_Fluid):
    def _calc_q(self, T):
        self._calc_qelec(T)
        self._calc_qvib(T, ncut=7 if self.ts else 6)
        self.qtot = self.qelec * self.qvib
        self.Etot = self.Eelec + self.Evib
        self.Htot = self.Etot + _k * T
        self.Stot = self.Selec + self.Svib


class Adsorbate(_Thermo):
    def __init__(self, outcar, spin=0., ts=False, coord=1, label=None, \
            eref=None, metal=None):
        super(Adsorbate, self).__init__(outcar, False, None, \
                spin, ts, label, eref, metal)
        assert np.all(self.freqs[1 if ts else 0:] > 0), "Imaginary frequencies found!"
        self.coord = coord

    def get_reference_state(self):
        return 1.

    def _calc_q(self, T):
        self._calc_qvib(T, ncut=1 if self.ts else 0)
        self._calc_qelec(T)
        self.qtot = self.qelec * self.qvib
        self.Etot = self.Eelec + self.Evib
        self.Htot = self.Etot + _k * T
        self.Stot = self.Selec + self.Svib

    def copy(self):
        return self.__class__(self.outcar, self.T, self.ts)


class Shomate(_Thermo):
    def __init__(self):
        raise NotImplementedError

#    def __init__(self, label, elements, shomatepars, fluid, T, coord=1):
#        self.label = label
#        self.elements = elements
#        self.shomatepars = shomatepars
#        self.A = shomatepars[0]
#        self.B = shomatepars[1]
#        self.C = shomatepars[2]
#        self.D = shomatepars[3]
#        self.E = shomatepars[4]
#        self.F = shomatepars[5]
#        self.G = shomatepars[6]
#        self.T = T
#        self.fluid = fluid
#        self.coord = coord
#
#    def get_enthalpy(self, T=None):
#        if T is None:
#            T = self.T
#        t = T / 1000
#        H = self.A*t + self.B*t**2/2. + self.C*t**3/3. + self.D*t**4/4. \
#                - self.E/t + self.F
#        H *= kJ / mol
#        return H
#
#    def get_entropy(self, T=None, P=None):
#        if T is None:
#            T = self.T
#        t = T / 1000
#        S = self.A*np.log(t) + self.B*t + self.C*t**2/2. + self.D*t**3/3. \
#                - self.E/(2*t**2) + self.G
#        S *= J / mol
#        return S
#
#    def copy(self):
#        return self.__class__(self.label, self.shomatepars, self.fluid, self.T)
#
#    def __repr__(self):
#        return self.label


class _Reactants(object):
    def __init__(self, species):
        self.species = []
        self.elements = {}
        for other in species:
            if isinstance(other, _Reactants):
                # If we're adding a _Reactants object to another
                # _Reactants object, just merge species and elements.
                self.species += other.species
                for key in other.elements:
                    if key in self.elements:
                        self.elements[key] += other.elements[key]
                    else:
                        self.elements[key] = other.elements[key]

            elif isinstance(other, _Thermo):
                # If we're adding a _Thermo object to a reactants
                # object, append the _Thermo to species and update
                # elements
                self.species.append(other)
                if isinstance(other, Shomate):
                    for symbol in other.elements:
                        if symbol in self.elements:
                            self.elements[symbol] += other.elements[symbol]
                        else:
                            self.elements[symbol] = other.elements[symbol]
                else:
                    for symbol in other.atoms.get_chemical_symbols():
                        if symbol in self.elements:
                            self.elements[symbol] += 1
                        else:
                            self.elements[symbol] = 1

            else:
                raise NotImplementedError
        self.reference_state = 1.
        for species in self.species:
            self.reference_state *= species.get_reference_state()

    def get_H(self, T=None):
        H = 0.
        for species in self.species:
            H += species.get_H(T)
        return H

    def get_S(self, T=None):
        S = 0.
        for species in self.species:
            S += species.get_S(T)
        return S

    def get_E(self, T=None):
        E = 0.
        for species in self.species:
            E += species.get_E(T)
        return E

    def get_q(self, T=None):
        q = 1.
        for species in self.species:
            q *= species.get_q(T)
        return q

    def get_reference_state(self):
        return self.reference_state

    def copy(self):
        return self.__class__(self.species)

    def get_mass(self):
        return sum([species.atoms.get_masses().sum() for species in self.species])

    def __iadd__(self, other):
        if isinstance(other, _Reactants):
            self.species += other.species
            for key in other.elements:
                if key in self.elements:
                    self.elements[key] += other.elements[key]
                else:
                    self.elements[key] = other.elements[key]

        elif isinstance(other, _Thermo):
            self.species.append(other)
            for symbol in other.atoms.get_chemical_symbols():
                if symbol in self.elements:
                    self.elements[symbol] += 1
                else:
                    self.elements[symbol] = 1
        else:
            raise NotImplementedError

    def __add__(self, other):
        new = self.copy()
        new += other
        return new

    def __imul__(self, factor):
        raise ValueError, "Reactants can only be multiplied by an integer"
        self.species *= factor
        for key in self.elements:
            self.elements[key] *= factor

    def __mul__(self, factor):
        new = self.copy()
        new *= factor
        return new

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __repr__(self):
        return ' + '.join([species.__repr__() for species in self.species])

    def __getitem__(self, i):
        return self.species[i]

    def __getslice__(self, i, j):
        return _Reactants([self.species[i:j]])

    def __len__(self):
        return len(self.species)
