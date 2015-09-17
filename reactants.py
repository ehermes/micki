"""This module contains object definitions of species
and collections of species"""

import copy

import numpy as np

from ase.io import read

from ase.units import J, mol, _hplanck, m, kg, _k, kB, _c, Pascal, _Nav

from micki.masses import masses


class _Thermo(object):
    """Generic thermodynamics object
    
    This is the base object that all reactant objects inherit from.
    It initializes many parameters and provides methods for calculating
    modeutions to the partition function from translation, rotation,
    and vibration."""

    def __init__(self, outcar, linear=False, symm=1, spin=0., ts=False, \
            label=None, eref=None, metal=None):
        self.T = None

        self.mode = ['tot', 'trans', 'trans2D', 'rot', 'vib', 'elec']

        self.q = dict.fromkeys(self.mode)
        self.S = dict.fromkeys(self.mode)
        self.E = dict.fromkeys(self.mode)
        self.H = None

        self.scale = {'E': dict.fromkeys(self.mode, 1.0),
                      'S': dict.fromkeys(self.mode, 1.0),
                      'H': 1.0}

        self.outcar = outcar
        self.metal = metal
        self.atoms = read(self.outcar, index=0)
        self.mass = [masses[atom.symbol] for atom in self.atoms]
        self.atoms.set_masses(self.mass)
        self.eref = eref
        if 'OUTCAR' in self.outcar:
            self._read_hess_outcar()
        elif self.outcar.endswith('.xml'):
            self._read_hess_xml()
        self.potential_energy = self.atoms.get_potential_energy()
        self.linear = linear
        self.symm = symm
        self.spin = spin
        self.ts = ts
        self.label = label
        self.coverage_dependence = 0.
        self.coord = 0

        self.scale_old = copy.deepcopy(self.scale)

        if self.eref is not None:
            for symbol in self.atoms.get_chemical_symbols():
                self.potential_energy -= self.eref[symbol]

    def update(self, T=None):
        """Updates the object's thermodynamic properties"""
        if not self.is_update_needed(T):
            return

        if T is None:
            T = self.T

        self.T = T
        self._calc_q(T)
        self.scale_old = copy.deepcopy(self.scale)
        self.E['tot'] += self.coverage_dependence
        self.H += self.coverage_dependence

    def is_update_needed(self, T):
        if self.q['tot'] is None:
            return True
        if T is not None and T != self.T:
            return True
        if self.scale != self.scale_old:
            return True
        return False

    def get_H(self, T=None):
        self.update(T)
        return self.H * self.scale['H']

    def get_S(self, T=None):
        self.update(T)
        return self.S['tot'] * self.scale['S']['tot']

    def get_E(self, T=None):
        self.update(T)
        return self.E['tot'] * self.scale['E']['tot']

    def get_q(self, T=None):
        self.update(T)
        return self.q['tot']

    def get_reference_state(self):
        raise NotImplementedError

    def _calc_q(self, T):
        raise NotImplementedError

    def _calc_qtrans2D(self, T):
        mtot = sum(self.mass) / kg
        self.q['trans2D'] = 2 * np.pi * mtot * _k * T / _hplanck**2 / self.rho0
        self.E['trans2D'] = kB * T * self.scale['E']['trans2D']
        self.S['trans2D'] = kB * (2 + np.log(self.q['trans2D'])) * self.scale['S']['trans2D']

    def _calc_qtrans(self, T):
        mtot = sum(self.mass) / kg
        self.q['trans'] = 0.001 * (2 * np.pi * mtot * _k * T / _hplanck**2)**(3./2.) \
                / (mol * self.rho0)
        self.E['trans'] = 3. * kB * T / 2. * self.scale['E']['trans']
        self.S['trans'] = kB * (5./2. + np.log(self.q['trans'])) * self.scale['S']['trans']

    def _calc_qrot(self, T):
        com = self.atoms.get_center_of_mass()
        if self.linear:
            I = 0
            for atom in self.atoms:
                I += atom.mass * np.linalg.norm(atom.position - com)**2
            I /= (kg * m**2)
            self.q['rot'] = 8 * np.pi**2 * I * _k * T / (_hplanck**2 * self.symm)
            self.E['rot'] = kB * T * self.scale['E']['rot']
            self.S['rot'] = kB * (1. + np.log(self.q['rot'])) * self.scale['S']['rot']
        else:
            I = self.atoms.get_moments_of_inertia() / (kg * m**2)
            thetarot = _hplanck**2 / (8 * np.pi**2 * I * _k)
            self.q['rot'] = np.sqrt(np.pi * T**3 / np.prod(thetarot)) / self.symm
            self.E['rot'] = 3. * kB * T / 2. * self.scale['E']['rot']
            self.S['rot'] = kB * (3./2. + np.log(self.q['rot'])) * self.scale['S']['rot']

    def _calc_qvib(self, T, ncut=0):
        thetavib = self.freqs[ncut:] / kB
        self.q['vib'] = np.prod(np.exp(-thetavib/(2. * T)) / (1. - np.exp(-thetavib/T)))
        self.E['vib'] = kB * sum(thetavib * (1./2. + 1./(np.exp(thetavib/T) - 1.))) \
                * self.scale['E']['vib']
        self.S['vib'] = kB * sum((thetavib/T)/(np.exp(thetavib/T) - 1.) \
                - np.log(1. - np.exp(-thetavib/T))) * self.scale['S']['vib']

    def _calc_qelec(self, T):
        self.E['elec'] = self.potential_energy * self.scale['E']['elec']
        self.S['elec'] = kB * np.log(2. * self.spin + 1.) * self.scale['S']['elec']

    def _read_hess_outcar(self):
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
                        hess = np.zeros((dof, dof), dtype=float)
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
                                raise ValueError("Error reading Hessian!")
                        hessblock = 3
                        j = 0

                    elif hessblock == 3:
                        line = line.split()
                        hess[j] = np.array([float(val) for val in line[1:]], \
                                dtype=float)
                        j += 1

                    elif line.startswith('SECOND DERIVATIVES'):
                        hessblock = 1

                elif hessblock == 3:
                    break

        hess = -(hess + hess.T) / 2.
        self._diagonalize(index, hess)

    def _read_hess_xml(self):
        import xml.etree.ElementTree as ET

        tree = ET.parse(self.outcar)
        root = tree.getroot()

        vasp_mass = {}

        for element in root.find("atominfo/array[@name='atomtypes']/set"):
            vasp_mass[element[1].text.strip()] = float(element[2].text)

        selective = np.ones((len(self.atoms), 3), dtype=bool)
        constblock = root.find('structure[@name="initialpos"]/varray[@name="selective"]')
        if constblock is not None:
            for i, v in enumerate(constblock):
                selective[i] = np.array(v.text.split() == np.array(['T', 'T', 'T']))
        index = []
        for i, atom in enumerate(self.atoms):
            for direction in selective[i]:
                if direction:
                    index.append(i)
        
        hess = np.zeros((len(index), len(index)), dtype=float)

        for i, v in enumerate(root.find('calculation/dynmat/varray[@name="hessian"]')):
            hess[i] = -np.array([float(val) for val in v.text.split()])

        vasp_massvec = np.zeros(len(index), dtype=float)
        for i, j in enumerate(index):
            vasp_massvec[i] = vasp_mass[self.atoms[j].symbol]

        hess *= np.sqrt(np.outer(vasp_massvec, vasp_massvec))

        self._diagonalize(index, hess)

    def _diagonalize(self, index, hess):
        mass = np.array([self.atoms[i].mass for i in index], dtype=float)
        hess /= np.sqrt(np.outer(mass, mass))

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
        self.hess = np.zeros((len(nonmetal), len(nonmetal)))
        self.index = np.zeros(len(nonmetal), dtype=int)
        for i, a in enumerate(nonmetal):
            self.index[i] = index[a]
            for j, b in enumerate(nonmetal):
                self.hess[i, j] = hess[a, b]

#        mass = np.array([self.atoms[i].mass for i in self.index], dtype=float)
#        self.hess /= np.sqrt(np.outer(mass, mass))
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

    def __iadd__(self, other):
        raise NotImplementedError

    def __mul__(self, factor):
        assert isinstance(factor, int)
        return _Reactants([self for i in range(factor)])

    def __rmul__(self, factor):
        return self.__mul__(factor)


class _Fluid(_Thermo):
    """Master object for both liquids and gasses"""
    def __init__(self, outcar, linear=False, symm=1, spin=0., \
            label=None, eref=None, rhoref=1.):
        _Thermo.__init__(self, outcar, linear, symm, \
                spin, False, label, eref, None)
        self.rho0 = rhoref
        assert np.all(self.freqs[6:] > 0), "Imaginary frequencies found!"

    def get_reference_state(self):
        return self.rho0

    def copy(self):
        return self.__class__(self.outcar, self.linear, self.symm, self.spin, \
                self.label, self.eref)

    def _calc_q(self, T):
        self._calc_qelec(T)
        self._calc_qtrans(T)
        self._calc_qrot(T)
        self._calc_qvib(T, ncut=7 if self.ts else 6)
        self.q['tot'] = self.q['trans'] * self.q['rot'] * self.q['vib']
        self.E['tot'] = self.E['elec'] + self.E['trans'] + self.E['rot'] + self.E['vib']
        self.H = self.E['tot'] + kB * T
        self.S['tot'] = self.S['elec'] + self.S['trans'] + self.S['rot'] + self.S['vib']


class Gas(_Fluid):
    pass


class Liquid(_Fluid):
    def _calc_q(self, T):
        _Fluid._calc_q(self, T)
        self.S['tot'] += kB * np.log(kB * T * mol / (100 * Pascal * m**3)) - 8.7291e-4


class Adsorbate(_Thermo):
    def __init__(self, outcar, spin=0., ts=False, coord=1, label=None, \
            eref=None, metal=None, coverage_dependence=0., symbol=None):
        _Thermo.__init__(self, outcar, False, None, \
                spin, ts, label, eref, metal)
        assert np.all(self.freqs[1 if ts else 0:] > 0), "Imaginary frequencies found!"
        self.coord = coord
        self.coverage_dependence = coverage_dependence
        self.symbol = symbol

    def get_reference_state(self):
        return 1.

    def _calc_q(self, T):
        self._calc_qvib(T, ncut=1 if self.ts else 0)
        self._calc_qelec(T)
        self.q['tot'] = self.q['vib']
        self.E['tot'] = self.E['elec'] + self.E['vib']
        self.H = self.E['tot'] + kB * T
        self.S['tot'] = self.S['elec'] + self.S['vib']

    def copy(self):
        return self.__class__(self.outcar, self.spin, self.ts, self.coord, \
                self.label, self.eref, self.metal)

class _DummyThermo(_Thermo):
    def __init__(self, linear, symm, spin, ts, freqs, label, E, rhoref):
        self.T = None
        self.potential_energy = E
        self.linear = linear
        self.symm = symm
        self.spin = spin
        self.ts = ts
        self.label = label
        self.freqs = np.array(freqs) * _hplanck * 100 * J * _c
        self.rho0 = rhoref
        self.scale_old = self.scale.copy()
        self.coverage_dependence = 0

    def _calc_qvib(self, T, ncut=0):
        self.E['vib'] = sum(self.freqs[ncut:])/2. * self.scale['E']['vib']
        thetavib = self.freqs[ncut:] / kB
        x = self.freqs[ncut:] / (kB * T)
        self.S['vib'] = kB * np.sum(x / (np.exp(x) - 1) - np.log(1-np.exp(-x))) * self.scale['S']['vib']

#        self.q['vib'] = np.prod(np.exp(-thetavib/(2. * T)) / (1. - np.exp(-thetavib/T)))
#        self.S['vib'] = kB * sum((thetavib/T)/(np.exp(thetavib/T) - 1.) \
#                - np.log(1. - np.exp(-thetavib/T))) * self.scale['S']['vib']

    def get_reference_state(self):
        return self.rho0

class DummyFluid(_DummyThermo):
    def __init__(self, geometry, linear=False, symm=1, spin=0., ts=False, \
            freqs=[], label=None, E=0., rhoref=1., Stot=None, monatomic=False,
            H=None):
        self.geometry = geometry
        self.atoms = read(geometry)
        self.mass = [masses[atom.symbol] for atom in self.atoms]
        self.atoms.set_masses(self.mass)
        self.Stot_in = Stot
        if self.Stot_in is not None:
            self.Stot_in *= J / _Nav
        self.monatomic = monatomic
        self.coord = 0
        self.H_in = H
        _DummyThermo.__init__(self, linear, symm, spin, ts, freqs, label, E, rhoref)
        if self.linear and len(self.freqs) < 3 * len(self.atoms) - 5:
            for i in range(3 * len(self.atoms) - 5 - len(self.freqs)):
                self.freqs = np.append(self.freqs, 200 * _hplanck * 100 * J * _c)
        elif not self.linear and len(self.freqs) < 3 * len(self.atoms) - 6:
            for i in range(3 * len(self.atoms) - 6 - len(self.freqs)):
                self.freqs = np.append(self.freqs, 200 * _hplanck * 100 * J * _c)

    def copy(self):
        return self.__class__(self.geometry, self.linear, self.symm, self.spin, \
                self.ts, self.freqs, self.label, self.potential_energy, self.rho0)

    def _calc_q(self, T):
        self._calc_qelec(T)
        if len(self.atoms) > 0:
            self._calc_qtrans(T)
        else:
            self.S['trans'] = 0
            self.E['trans'] = 0
        if self.monatomic:
            self.E['rot'] = 0
            self.E['vib'] = 0
            self.S['rot'] = 0
            self.S['vib'] = 0
        else:
            self._calc_qrot(T)
            self._calc_qvib(T)
        self.E['tot'] = self.E['elec'] + self.E['trans'] + self.E['rot'] + self.E['vib']
        if self.H_in is not None:
            self.H = self.H_in
        else:
            self.H = self.E['tot'] + kB * T
        self.S['tot'] = self.S['elec'] + self.S['trans'] + self.S['rot'] + self.S['vib']


class DummyAdsorbate(_DummyThermo):
    def __init__(self, label, spin=0., ts=False, coord=1, freqs=[], E=0, \
            coverage_dependence=0., gas=None, Floc=1., natoms=None, ZPE=None):
        self.coord = coord
        _DummyThermo.__init__(self, False, 1, spin, ts, freqs, label, E, 1.)
        self.coverage_dependence = coverage_dependence
        self.gas = gas
        if self.gas is not None:
            self.atoms = self.gas.atoms
        self.Floc = Floc
        self.natoms = natoms
        self.ZPE = ZPE
#        if self.ts and len(self.freqs) < 3 * self.natoms - 1:
#            for i in range(3 * self.natoms - 1 - len(self.freqs)):
#                self.freqs = np.append(self.freqs, 200 * _hplanck * 100 * J * _c)
#        elif not self.ts and len(self.freqs) < 3 * self.natoms:
#            for i in range(3 * self.natoms - len(self.freqs)):
#                self.freqs = np.append(self.freqs, 200 * _hplanck * 100 * J * _c)

    def _calc_q(self, T):
        self._calc_qvib(T)
        self._calc_qelec(T)
        if self.gas is not None:
            self.gas._calc_q(T)
            if self.ZPE is not None:
                self.gas.E['vib'] = self.E['vib'] - self.ZPE
            if self.gas.Stot_in is not None:
                self.Stot_gas = self.gas.Stot_in
            else:
                self.Stot_gas = (self.gas.S['trans'] + self.gas.S['rot']
                        + self.gas.S['vib'] + self.gas.S['elec'])
        self.E['tot'] = self.E['elec'] + self.E['vib']
        self.H = self.E['tot'] + kB * T
        if self.gas is not None:
            self.H -= self.gas.Evib
        self.S['tot'] = self.S['elec'] + self.S['vib']

    def get_S_gas(self, T):
        assert self.gas is not None
        self.update(T)
        return self.Floc * (self.Stot_gas - self.gas.S['trans'])
    
    def get_dZPE(self, T):
        self.gas.update(T)
        self.update(T)
        return self.E['vib'] - self.gas.E['vib']

    def copy(self):
        return self.__class__(self.label, self.spin, self.ts, self.coord, \
                self.freqs, self.potential_energy)


class Shomate(_Thermo):
    def __init__(self):
        raise NotImplementedError

class _Reactants(object):
    def __init__(self, species):
        self.species = []
        self.elements = {}
        for i, other in enumerate(species):
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
                elif isinstance(other, DummyAdsorbate):
                    pass
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
        return self

    def __add__(self, other):
        return _Reactants([self, other])

    def __imul__(self, factor):
        assert isinstance(factor, int) and factor > 0
        self.species *= factor
        for key in self.elements:
            self.elements[key] *= factor
        return self

    def __mul__(self, factor):
        assert isinstance(factor, int) and factor > 0
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
