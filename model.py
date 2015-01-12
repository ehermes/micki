#!/usr/bin/env python

import numpy as np
from ase.units import kB, _hplanck, kg, _k, _Nav
from odespy import Radau5Implicit
import sympy as sym
from mkm.reactants import _Thermo, _Reactants

class Reaction(object):
    def __init__(self, reactants, products, ts=None, method=None, S0=1., \
            dh_scale=1.0, ds_scale=1.0, dh_act_scale=1.0, ds_act_scale=1.0):
        if isinstance(reactants, _Thermo):
            self.reactants = _Reactants([reactants])
        elif isinstance(reactants, _Reactants):
            self.reactants = reactants
        else:
            raise NotImplementedError
        if isinstance(products, _Thermo):
            self.products = _Reactants([products])
        elif isinstance(products, _Reactants):
            self.products = products
        else:
            raise NotImplementedError
        self.ts = None
        if ts is not None:
            if isinstance(ts, _Thermo):
                self.ts = _Reactants([ts])
            elif isinstance(ts, _Reactants):
                self.ts = ts
            else:
                raise NotImplementedError
        # Mass balance requires that each element in the reactant is preserved
        # in the product and in any transition states.
        for element in self.reactants.elements:
            assert self.reactants.elements[element] == self.products.elements[element]
            if self.ts is not None:
                assert self.reactants.elements[element] == self.ts.elements[element]
        self.method = method
        if isinstance(self.method, str):
            self.method =self.method.upper()
        if self.method not in [None, 'CT']:
            raise ValueError, "Method {} unrecognized!".format(self.method)
        self.S0 = S0
        self.dh_scale = dh_scale
        self.ds_scale = ds_scale
        self.dh_act_scale = dh_act_scale
        self.ds_act_scale = ds_act_scale
    def get_keq(self, T):
        # Keq = e^(DS/kB - DH/(kB * T))
        self.ds = self.products.get_entropy(T) \
                - self.reactants.get_entropy(T)
        self.dh = self.products.get_enthalpy(T) \
                - self.reactants.get_enthalpy(T)
        self.dh *= self.dh_scale
        self.ds *= self.ds_scale
        self.keq = np.exp(self.ds/kB - self.dh/(kB * T))
        return self.keq
    def get_kfor(self, T, N0):
        if self.ts is None:
            if self.method is None:
#                self.kfor = 1e13
                self.kfor = _k * T / _hplanck
                keq = self.get_keq(T)
                if keq < 1:
                    self.kfor *= keq
            elif self.method == 'CT':
                # Collision theory:
                # kfor = S0 / (N0 * sqrt(2 * pi * m * kB * T))
                self.kfor = (1000 * self.S0 * _Nav/ N0) * np.sqrt(_k * T * kg \
                        / (2 * np.pi * self.reactants.get_mass()))
        else:
            # Transition State Theory:
            # kfor = (kB * T / h) * e^(DS_act/kB - DH_act/(kB * T))
            self.ds_act = self.ts.get_entropy(T) \
                    - self.reactants.get_entropy(T)
            self.dh_act = self.ts.get_enthalpy(T) \
                    - self.reactants.get_enthalpy(T)
            self.ds_act *= self.ds_act_scale
            self.dh_act *= self.dh_act_scale
            self.kfor = (_k * T / _hplanck) * np.exp(self.ds_act / kB) \
                    * np.exp(-self.dh_act / (kB * T))
        return self.kfor
    def get_krev(self, T, N0):
        keq = self.get_keq(T)
        if (self.ts is None) and (self.method is None):
#            self.krev = 1e13
            self.krev = _k * T / _hplanck
            if keq > 1:
                self.krev /= keq
        elif self.method == 'CT':
            self.de = self.products.get_energy() - self.reactants.get_energy()
#            self.krev = 1e13 * np.exp(-self.de / (kB * T))
            self.krev = (_k * T / _hplanck) * np.exp(self.de / (kB * T))
        else:
            kfor = self.get_kfor(T, N0)
            self.krev = kfor / keq
        return self.krev
    def set_scale(self, dh=None, ds=None, dh_act=None, ds_act=None):
        if dh is not None:
            self.dh_scale = dh
        if ds is not None:
            self.ds_scale = ds
        if dh_act is not None:
            self.dh_act_scale = dh_act
        if ds_act is not None:
            self.ds_act_scale = ds_act
    def __repr__(self):
        string = self.reactants.__repr__() + ' <-> '
        if self.ts is not None:
            string += self.ts.__repr__() + ' <-> '
        string += self.products.__repr__()
        return string

class Model(object):
    def __init__(self, reactions, T, V, nsites):
        self.reactions = []
        self.species = []
        for reaction in reactions:
            assert isinstance(reaction, Reaction)
            self.reactions.append(reaction)
            for species in reaction.reactants:
                self.add_species(species)
            for species in reaction.products:
                self.add_species(species)
        self.T = T
        self.V = V
        self.nsites = nsites
    def add_reaction(self, reaction):
        assert isinstance(reaction, Reaction)
        self.reactions.append(reaction)
        for species in reaction.reactants:
            self.add_species(species)
    def add_species(self, species):
        assert isinstance(species, _Thermo)
        if species not in self.species:
            self.species.append(species)
    def set_vacancy(self, species, N0):
        assert species.gas is False
        if species not in self.species:
            self.species.append(species)
        self.vacancy = species
        newspecies = []
        self.ngas = 0
        for species in self.species:
            if species.gas:
                self.ngas += 1
                newspecies.append(species)
        for species in self.species:
            if not species.gas and species is not self.vacancy:
                newspecies.append(species)
        newspecies.append(self.vacancy)
        self.species = newspecies
        self.N0 = N0
    def set_initial_conditions(self, U0):
        self.U0 = []
        for species in U0:
            if species not in self.species:
                raise ValueError, "Unknown species!"
        for species in self.species:
            if species in U0:
                self.U0.append(U0[species])
            else:
                self.U0.append(0.)
        self._initialize()
    def _initialize(self):
        raise NotImplementedError
    def _rate_calc(self):
        self.rates = []
        self.rate_count = []
        for reaction in self.reactions:
            rate_count = {species:0 for species in self.species}
            rate_for = reaction.get_kfor(self.T, self.N0)
            for species in reaction.reactants:
                rate_for *= self.symbols_dict[species]
                rate_count[species] -= 1
            rate_rev = reaction.get_krev(self.T, self.N0)
            for species in reaction.products:
                rate_rev *= self.symbols_dict[species]
                rate_count[species] += 1
            self.rates.append(rate_for - rate_rev)
            self.rate_count.append(rate_count)
    def f(self, x, t):
        y = np.zeros_like(self.f_exec, dtype=float)
        for i in xrange(len(self.symbols)):
            y[i] = self.f_exec[i](*x)
        return y
    def jac(self, x, t):
        y = np.zeros_like(self.jac_exec, dtype=float)
        for i in xrange(len(self.symbols)):
            for j in xrange(len(self.symbols)):
                y[i, j] = self.jac_exec[i][j](*x)
        return y
    def mas(self):
        return self.M
    def solve(self, t):
        self.U1, self.t = self.model.solve(t)
        return self._results()
    def _results(self):
        self.U =[]
        for i, t in enumerate(self.t):
            Ui = {}
            for j, species in enumerate(self.species):
                Ui[species] = self.U1[i][j]
            self.U.append(Ui)
        return self.U, self.t



class GasPhaseModel(Model):
    def __init__(self, reactions, T, V, nsites):
        super(GasPhaseModel, self).__init__(reactions, T, V, nsites)
    def _initialize(self):
        # Initialize the mass matrix
        self.M = np.identity(len(self.species), dtype=int)
        self.M[-1, -1] = 0
        for i, species in enumerate(self.species):
            if species.fixed:
                self.M[i, i] = 0
        # Create sympy symbols for creating f and jac
        self.symbols = sym.symbols('x0:{}'.format(len(self.species)))
        self.symbols_dict = {}
        for i, species in enumerate(self.species):
            self.symbols_dict[species] = self.symbols[i]
        # Set up rates
        self._rate_calc()
        self.f_sym = []
        self.f_exec = []
        for species in self.species:
            f = 0
            if species is self.vacancy:
                f = 1
                for a in self.species:
                    if not a.gas:
                        f -= self.symbols_dict[a] * a.coord
            elif species.gas:
                if not species.fixed:
                    for i, rate in enumerate(self.rates):
                        f += self.rate_count[i][species] * rate * self.nsites / self.V
            else:
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
        self.model.set_initial_condition(self.U0)

class LiquidPhaseModel(Model):
    def __init__(self, reactions, T, V, nsites, z, dz):
        self.z = z
        self.dz = dz
        self.nz = int(np.ceil(self.z/self.dz))
        assert self.nz >= 3, "Grid too coarse! Increase z or decrease dz."
        super(LiquidPhaseModel, self).__init__(reactions, T, V, nsites)
    def set_initial_conditions(self, U0):
        self.U0 = []
        for species in U0:
            if species not in self.species:
                raise ValueError, "Unknown species!"
        for species in self.species:
            if species in U0:
                U0i = U0[species]
            else:
                U0i = 0.
            if species.gas:
                for i in xrange(self.nz):
                    self.U0.append(U0i)
            else:
                self.U0.append(U0i)
        self._initialize()
    def _initialize(self):
        size = (self.nz - 1) * self.ngas + len(self.species)
        self.M = np.identity(size, dtype=int)
        self.M[-1, -1] = 0.
        self.symbols = sym.symbols('x0:{}'.format(size))
        self.symbols_dict = {}
        j = 0
        for species in self.species:
            if species.gas:
                self.symbols_dict[species] = self.symbols[j]
                for i in xrange(self.nz):
                    self.symbols_dict[(species, i)] = self.symbols[j]
                    j += 1
            else:
                self.symbols_dict[species] = self.symbols[j]
                j += 1
        self.dV = self.N0 * self.dz # Volume of grid thickness per ads. site
        self._rate_calc()
        self.f_sym = []
        self.f_exec = []
        for species in self.species:
            f = 0
            if species.gas:
                for i in xrange(self.nz):
                    f = 0
                    if i == 0:
                        f += (species.D / self.dz**2) \
                                * (self.symbols_dict[(species, i + 1)] \
                                - self.symbols_dict[(species, i)])
                        for j, rate in enumerate(self.rates):
                            f += self.rate_count[j][species] * rate * self.nsites / self.V
                    elif i == self.nz - 2:
                        f += (species.D / self.dz**2) \
                                * (self.symbols_dict[(species, i + 1)] \
                                - 2 * self.symbols_dict[(species, i)] \
                                + self.symbols_dict[(species, i - 1)])
                    elif i == self.nz - 1:
                        if not species.fixed:
                            f += (species.D  * self.dV / (self.dz**2 * self.V)) \
                                    * (-self.symbols_dict[(species, i)] \
                                    + self.symbols_dict[(species, i - 1)])

                    else:
                        f += (species.D / self.dz**2) \
                                * (self.symbols_dict[(species, i + 1)] \
                                - 2 * self.symbols_dict[(species, i)] \
                                + self.symbols_dict[(species, i - 1)])
                    self.f_sym.append(f)
                    self.f_exec.append(sym.lambdify(self.symbols, f))
                else:
                    continue
            elif species is not self.vacancy:
                for i, rate in enumerate(self.rates):
                    f += self.rate_count[i][species] * rate
            else:
                f = 1
                for a in self.species:
                    if not a.gas:
                        f -= self.symbols_dict[a] * a.coord
            self.f_sym.append(f)
            self.f_exec.append(sym.lambdify(self.symbols, f))

        self.jac_exec = []
        for f in self.f_sym:
            jac_exec = []
            for symbol in self.symbols:
                jac_exec.append(sym.lambdify(self.symbols, sym.diff(f, \
                        symbol)))
            self.jac_exec.append(jac_exec)
        self.model = Radau5Implicit(f=self.f, jac=self.jac, mas=self.mas, \
                rtol=1e-8)
        self.model.set_initial_condition(self.U0)
    def _results(self):
        self.U =[]
        for i, t in enumerate(self.t):
            Ui = {}
            j = 0
            for species in self.species:
                if species.gas:
                    for k in xrange(self.nz):
                        Ui[(species, k)] = self.U1[i][j]
                        j += 1
                else:
                    Ui[species] = self.U1[i][j]
                    j += 1
            self.U.append(Ui)
        return self.U, self.t

