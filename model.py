#!/usr/bin/env python

import numpy as np
from ase.units import kB, _hplanck, kg, _k, _Nav
from odespy import Radau5Implicit
import sympy as sym
from mkm.reactants import _Thermo, _Reactants

class Reaction(object):
    def __init__(self, reactants, products, ts=None, method=None, S0=None, \
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
        self.S0 = S0
        if self.method is not None:
            assert self.ts is None, \
                    "ts and method arguments are not supported together!"
#            assert isinstance(self.reactants[0], IdealGas)
#            assert isinstance(self.products[0], Harmonic)
        if method != 'CT' and S0 is not None:
            print "Warning! Parameter S0 is only valid for CT calculations"
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
#        self.keq = np.exp(self.ds/R - self.dh/(R * T))
        if self.method is not None:
            if self.method.upper() == 'SPECIAL':
                self.keq = 1e-60
        return self.keq
    def get_kfor(self, T, N0):
        if self.ts is None:
            if self.method.upper() == 'DES':
                krev = self.get_krev(T, N0)
                keq = self.get_keq(T)
                self.kfor = krev * keq
            elif self.method.upper() == 'CT':
                # Collision theory:
                # kfor = S0 / (N0 * sqrt(2 * pi * m * kB * T))
                if self.S0 is None:
                    self.S0 = 1.
                self.kfor = (1000 * self.S0 * _Nav/ N0) * np.sqrt(_k * T * kg \
                        / (2 * np.pi * self.reactants.get_mass()))
#                self.kfor = self.S0 / (N0 * \
#                    np.sqrt(2 * np.pi * self.reactants.get_mass() * kB * T \
#                    / (J * kg)))
            elif self.method.upper() == 'EQUIL':
                self.kfor = 1e30
            elif self.method.upper() == 'SPECIAL':
                self.kfor = 1e-30
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
#            self.kfor = (kB * T / hplanck) * np.exp(self.ds_act / kB) \
#                    * np.exp(-self.dh_act / (kB * T))
#            self.kfor = (kB * T / hplanck) * np.exp(self.ds_act / R) \
#                    * np.exp(-self.dh_act / (R * T))
        return self.kfor
    def get_krev(self, T, N0):
        if self.ts is None:
            if self.method.upper() == 'DES':
                # FIXME
                raise NotImplementedError
        keq = self.get_keq(T)
        kfor = self.get_kfor(T, N0)
        self.krev = kfor / keq
        return self.krev
    def set_scale(self, dh=1.0, ds=1.0, dh_act=1.0, ds_act=1.0):
        self.dh_scale = dh
        self.ds_scale = ds
        self.dh_act_scale = dh_act
        self.ds_act_scale = ds_act
    def __repr__(self):
        string = self.reactants.__repr__() + ' <-> '
        if self.ts is not None:
            string += self.ts.__repr__() + ' <-> '
        string += self.products.__repr__()
        return string

class Model(object):
#    def __init__(self, reactions, reactor, T, N0, Ptot, Pbuffer):
    def __init__(self, reactions, reactor, T, N0, V, nsites):
        self.reactions = []
        self.species = []
        self.reactor = reactor
        self.T = T
#        self.Ptot = Ptot
#        self.Pbuffer = Pbuffer
        self.V = V
        self.nsites = nsites
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
        assert isinstance(species, _Thermo)
        if species not in self.species:
            self.species.append(species)
    def set_vacancy(self, species):
        assert species.gas is False
#        assert isinstance(species, Harmonic)
        if species not in self.species:
            self.species.append(species)
        self.vacancy = species
        newspecies = []
        for species in self.species:
            if species.gas:
                newspecies.append(species)
        for species in self.species:
            if not species.gas and species is not self.vacancy:
                newspecies.append(species)
        newspecies.append(self.vacancy)
        self.species = newspecies
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
        # Initialize the mass matrix
        self.M = np.identity(len(self.species))
        for i, species in enumerate(self.species):
            if species is self.vacancy:
                self.M[i, i] = 0
#            if not species.gas:
#                if self.reactor.upper() in ['BATCH', 'PFR']:
#                    self.M[i, i] = 0
        # Create sympy symbols for creating f and jac
        self.symbols = sym.symbols('x0:{}'.format(len(self.species)))
        self.symbols_dict = {}
        for i, species in enumerate(self.species):
            self.symbols_dict[species] = self.symbols[i]
        # Normalize gas pressure
#        self.P = self.Pbuffer
#        for species in self.species:
#            if species.gas:
#                self.P += self.symbols_dict[species]
#        self.Pnorm = self.Ptot / self.P
        # Set up rates
        self.rates = []
        self.rate_count = []
        for reaction in self.reactions:
            rate_count = {species:0 for species in self.species}
            rate_for = reaction.get_kfor(self.T, self.N0)
            for species in reaction.reactants:
                rate_for *= self.symbols_dict[species]
#                if species.gas:
#                    rate_for *= self.Pnorm
                rate_count[species] -= 1
            rate_rev = reaction.get_krev(self.T, self.N0)
            for species in reaction.products:
                rate_rev *= self.symbols_dict[species]
#                if species.gas:
#                    rate_for *= self.Pnorm
                rate_count[species] += 1
            self.rates.append(rate_for - rate_rev)
            self.rate_count.append(rate_count)
        self.f_sym = []
        self.f_exec = []
        for species in self.species:
            f = 0
            if species is self.vacancy:
                f = 1
                for a in self.species:
                    if not a.gas:
                        f -= self.symbols_dict[a] * a.nsites
            elif species.gas:
                if not species.fixed:
                    for i, rate in enumerate(self.rates):
                        f += self.rate_count[i][species] * rate * self.nsites \
                                / self.V
            else:
                for i, rate in enumerate(self.rates):
                    f += self.rate_count[i][species] * rate
        #    elif not species.gas:
        #        for i, rate in enumerate(self.rates):
        #            f += self.rate_count[i][species] * rate
        #    else:
        #        if self.reactor.upper() != 'CSTR':
        #            for i, rate in enumerate(self.rates):
        #                f += self.rate_count[i][species] * rate * self.nsites \
        #                        / self.V
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
#        self.model = Radau5Implicit(f=self.f, mas=self.mas, rtol=1e-8)
        self.model.set_initial_condition(self.U0)
    def f(self, x, t):
        y = np.zeros_like(self.f_exec)
        for i in xrange(len(self.species)):
            y[i] = self.f_exec[i](*x)
        return y
    def jac(self, x, t):
        y = np.zeros_like(self.jac_exec)
        for i in xrange(len(self.species)):
            for j in xrange(len(self.species)):
                y[i, j] = self.jac_exec[i][j](*x)
        return y
    def mas(self):
        return self.M
    def solve(self, t):
        U1, self.t = self.model.solve(t)
        self.U =[]
        for i, t in enumerate(self.t):
            Ui = {}
            for j, species in enumerate(self.species):
                Ui[species] = U1[i][j]
            self.U.append(Ui)
        return self.U, self.t

