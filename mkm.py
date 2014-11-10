#!/usr/bin/env python

import numpy as np
from ase.io import read
from ase.units import J, kJ, mol, kB, _hplanck
from ase.thermochemistry import IdealGasThermo, HarmonicThermo
from eref import eref
from masses import masses
from odespy import Radau5Implicit
import sympy as sym

hplanck = _hplanck * J

class Thermo(object):
    def __init__(self, outcar, T=298.15, P=101325, linear=False, symm=1, \
            spin=0., ts=False):
        self.outcar = outcar
        self.atoms = read(self.outcar, index=0)
        self.mass = [masses[atom.symbol] for atom in self.atoms]
        self.atoms.set_masses(selfmass)
        self._read_hess()
        self.T = T
        self.P = P
        self.geometry = 'linear' if linear else 'nonlinear'
        self.e_elec = atoms.get_potential_energy()
        self.symm = symm
        self.spin = spin
        self.ts = ts
        for atom in self.atoms:
            if atom.symbol in eref:
                self.e_elec -= eref[atom.symbol]
    def get_enthalpy(self, T=None):
        if T is None:
            T = self.T
        return self.thermo.get_enthalpy(T, verbose=False) #* mol / kJ
    def get_entropy(self, T=None, P=None):
        if T is None:
            T = self.T
        return self.thermo.get_entropy(T, verbose=False) #* 1000 * mol / kJ
    def copy(self):
        raise NotImplementedError
    def _read_hess(self):
        hessblock = 0
        with open(self.outcar, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
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
                        self.hess[j] = np.array([float(val) for val in line[1:]])
                        j += 1
                    elif line.startswith('SECOND DERIVATIVES'):
                        hessblock = 1
                elif hessblock == 3:
                    break
        self.hess = -(self.hess + self.hess.T) / 2.
        mass = np.array([slab[i].mass for i in index])
        self.hess /= np.sqrt(np.outer(mass, mass))
        self.hess *= _hplanck**2 * J * m**2 * kg / (4 * np.pi**2)
        v, w = np.linalg.eig(self.hess)
        freq = np.sqrt(np.array(v, dtype=complex))
        self.freqs = np.zeros_like(freq, dtype=float)
        for i, val in enumerate(freq):
            if val.imag == 0:
                self.freqs[i] = val.real
            else:
                self.freqs[i] = -val.imag
        newfreq.sort()
    def __repr__(self):
        return self.atoms.get_chemical_formula()
    def __add__(self, other):
        return Reactants([self, other])
    def __mul__(self, factor):
        assert isinstance(factor, int)
        return Reactants([self for i in int])

class IdealGas(Thermo):
    def __init__(self, atoms, freqs, T=298.15, P=101325, linear=False, symm=1, \
            spin=0.):
        super(IdealGas, self).__init__(atoms, freqs, T, P, linear, symm, \
                spin, False)
        self.thermo = IdealGasThermo(
                self.freqs[6:],
                self.geometry,
                electronicenergy=self.e_elec,
                symmetrynumber=self.symm,
                spin=self.spin,
                atoms=self.atoms,
                )
    def get_entropy(self, T=None, P=None):
        if T is None:
            T = self.T
        if P is None:
            P = self.P
        return self.thermo.get_entropy(T, P, verbose=False) #* 1000 * mol / kJ
    def copy(self):
        return self.__class__(self.atoms, self.freqs, self.T, self.P, \
                self.linear, selfsymm, self.spin)


class Harmonic(Thermo):
    def __init__(self, atoms, freqs, eslab, T=298.15, ts=False):
        super(Harmonic, self).__init__(atoms, freqs, T, None, None, None, \
                None, ts)
        self.e_elec -= eslab
        self.thermo = HarmonicThermo(
                self.freqs[1 if ts else 0:],
                electronicenergy=self.e_elec,
                )

class Reactants(object):
    def __init__(self, species):
        self.species = []
        self.elements = {}
        for other in species:
            if isinstance(other, Reactants):
                self.species += other.species
                for key in other.elements:
                    if key in self.elements:
                        self.elements[key] += other.elements[key]
                    else:
                        self.elements[key] = other.elements[key]
            elif isinstance(other, Thermo):
                self.species.append(other)
                for symbol in other.atoms.get_chemical_symbols():
                    if symbol in self.elements:
                        self.elements[symbol] += 1
                    else:
                        self.elements[symbol] = 1
            else:
                raise NotImplementedError
    def get_enthalpy(self, T=None):
        H = 0.
        for species in self.species:
            H += species.get_enthalpy(T)
        return H
    def get_entropy(self, T=None, P=None):
        S = 0.
        for species in self.species:
            S += species.get_entropy(T, P)
        return S
    def copy(self):
        return self.__class__(self.species)
    def get_mass(self):
        mass = 0
        for species in self.species:
            mass += species.atoms.get_masses().sum()
        return mass
    def __iadd__(self, other):
        if isinstance(other, Reactants):
            self.species += other.species
            for key in other.elements:
                if key in self.elements:
                    self.elements[key] += other.elements[key]
                else:
                    self.elements[key] = other.elements[key]
        elif isinstance(other, Thermo):
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
        if not isinstance(factor, int):
            raise NotImplementedError
        self.species *= factor
        for key in self.elements:
            self.elements[key] *= factor
    def __mul__(self, factor):
        new = self.copy()
        new *= factor
        return new
    def __repr__(self):
        string = ''
        for species in self.species:
            if string:
                string += ' + '
            string += species.__repr__()
        return string
    def __getitem__(self, i):
        return self.species[i]
    def __getslice__(self, i, j):
        return Reactants([self.species[i:j]])
    def __len__(self):
        return len(self.species)

class Reaction(object):
    def __init__(self, reactants, products, ts=None, T=None, method=None, S0=None):
        self.T = T
        if isinstance(reactants, Thermo):
            self.reactants = Reactants([reactants])
        elif isinstance(reactants, Reactants):
            self.reactants = reactants
        else:
            raise NotImplementedError
        if isinstance(products, Thermo):
            self.products = Reactants([products])
        elif isinstance(products, Reactants):
            self.products = products
        else:
            raise NotImplementedError
        self.ts = None
        if ts is not None:
            if isinstance(ts, Thermo):
                self.ts = Reactants([ts])
            elif isinstance(ts, Reactants):
                self.ts = ts
            else:
                raise NotImplementedError
        for element in self.reactants.elements:
            assert self.reactants.elements[element] == self.products.elements[element]
            if self.ts is not None:
                assert self.reactants.elements[element] == self.ts.elements[element]
        self.method = method
        self.S0 = S0
        if self.method is not None:
            assert len(self.reactants) == len(self.products) == 1, \
                    "Only adsorption reactions support the method argument!"
            assert self.ts is None, \
                    "ts and method arguments are not supported together!"
            assert isinstance(self.reactants[0], IdealGas)
            assert isinstance(self.products[0], Harmonic)
        if method != 'CT' and S0 is not None:
            print "Warning! Parameter S0 is only valid for CT calculations"
    def get_keq(self):
        self.ds = self.products.get_entropy(self.T) \
                - self.reactants.get_entropy(self.T) 
        self.dh = self.products.get_enthalpy(self.T) \
                - self.reactants.get_enthalpy(self.T)
        self.keq = np.exp(self.ds/kB - self.dh/(kB * self.T))
        return self.keq
    def get_kfor(self, N0):
        if self.ts is None:
            if self.method == 'des':
                krev = self.get_krev(N0)
                keq = self.get_keq()
                self.kfor = krev * keq
                return self.kfor
            elif self.method == 'CT':
                if self.S0 is None:
                    self.S0 = 1.
            self.kfor = self.S0 / (self.N0 * \
                    np.sqrt(2 * np.pi * self.reactants.get_masses() * kB * self.T))
        else:
            self.ds_act = self.ts.get_entropy(self.T) \
                    - self.reactants.get_entropy(self.T)
            self.dh_act = self.ts.get_enthalpy(self.T) \
                    - self.reactants.get_enthalpy(self.T)
            self.kfor = (kB * self.T / hplanck) * np.exp(self.ds_act / kB) \
                    * np.exp(-self.dh_act / (kB * self.T))
        return self.kfor
    def get_krev(self, N0):
        if self.ts is None:
            if self.method == 'des':
                # FIXME
                raise NotImplementedError
        keq = self.get_keq()
        kfor = self.get_kfor(N0)
        self.krev = kfor / keq
        return self.krev
    def __repr__(self):
        string = self.reactants.__repr__() + ' <-> '
        if self.ts is not None:
            string += self.ts.__repr__() + ' <-> '
        string += self.products.__repr__()
        return string

class Model(object):
    def __init__(self, reactions, reactor, N0):
        self.reactions = []
        self.species = []
        self.reactor = reactor
        for reaction in reactions:
            assert isinstance(reaction, Reaction)
            self.reactions.append(reaction)
            for species in reaction.reactants:
                self.add_species(species)
            for species in reaction.products:
                self.add_species(species)
        self.N0 = N0
    def add_reaction(self, reaction):
        assert isinstance(reaction, Reaction)
        self.reactions.append(reaction)
        for species in reaction.reactants:
            self.add_species(species)
    def add_species(self, species):
        assert isinstance(species, Thermo)
        if species not in self.species:
            self.species.append(species)
    def set_vacancy(self, species):
        assert isinstance(species, Harmonic)
        if species not in self.species:
            self.species.append(species)
        for a in self.species:
            if isinstance(a, Harmonic):
                for b in a:
                    b.e_elec -= species.e_elec
        self.vacancy = species
    def set_initial_conditions(self, U0):
        self.U0 = []
        for species in U0:
            if species not in self.species:
                raise ValueError, "Unknown species!"
        for species in self.species:
            self.U0.append(U0[species])
        self._initialize()
    def _initialize(self):
        self.M = np.identity(len(self.species))
        for i, species in self.species:
            if species is self.vacancy:
                self.M[i, i] = 0
        self.symbols = sym.symbols('x0:{}'.format(len(self.species)))
        self.symbols_dict = {}
        for i, species in enumerate(self.species):
            self.symbols_dict[species] = self.symbols[i]
        self.rates = []
        self.rate_count[]
        for reaction in self.reactions:
            rate_count = {species:0 for species in self.species}
            rate_for = reaction.get_kfor(self.N0)
            for species in reaction.reactants:
                rate_for *= self.symbols_dict[species]
                rate_count[species] += 1
            rate_rev = reaction.get_krev(self.N0)
            for species in reaction.products:
                rate_rev *= self.symbols_dict[species]
                rate_count[species] -= 1
            self.rates.append(rate_for - rate_rev)
            self.rate_count.append(rate_count)
        self.f_sym = []
        self.f_exec = []
        for species in self.species:
            f = 0
            if species is self.vacancy:
                f = 1
                for a in self.species:
                    if isinstance(a, Harmonic):
                        f -= self.symbols_dict[a]
            elif isinstance(species, Harmonic):
                for i, rate in enumerate(self.rates):
                    f += self.rate_count[i][species] * rate
            self.f_sym.append(f)
            self.f_exec.append(sym.lambdify(self.symbols, f))

        self.jac_exec = []
        for f in self.f_sym:
            jac_exec = []
            for species in self.species:
                jac_exec.append(sym.lambdify(self.symbols, sym.diff(f, \
                        self.symbols_dict[species])))
            self.jac_exec.append(jac_exec)
        self.model = Radau5Implicit(f=self.f, jac=self.jac, mas=self.mas, \
                rtol=1e-8)
        self.model.initial_condition(self.U0)
    def f(self, x, t):
        y = np.zeros_like(self.f_exec)
        for i in xrange(len(self.species)):
            y[i] = self.f_exec[i](x)
        return y
    def jac(self, x, t):
        y = np.zeros_like(self.jac_exec)
        for i in xrange(len(self.species)):
            for j in xrange(len(self.species)):
                y[i, j] = self.jac_exec[i][j](x)
        return y
    def mas(self):
        return self.M
    def solve(self, t):
        U1, t1 = self.model.solve(t)
        return U1, t1
