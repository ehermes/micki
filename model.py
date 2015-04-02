"""Microkinetic modeling objects"""

import numpy as np

from scipy.integrate import odeint

from ase.units import kB, _hplanck, kg, _k, _Nav, mol

from odespy import Radau5Implicit

import sympy as sym

from mkm.reactants import _Thermo, _Fluid, _Reactants, Gas, Liquid, Adsorbate


class Reaction(object):
    def __init__(self, reactants, products, ts=None, method='EQUIL', S0=1., \
            adsorption=False):
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
        self.species = []
        for species in self.reactants:
            self.species.append(species)
        for species in self.products:
            self.species.append(species)
        self.ts = None
        if ts is not None:
            if isinstance(ts, _Thermo):
                self.ts = _Reactants([ts])
            elif isinstance(ts, _Reactants):
                self.ts = ts
            else:
                raise NotImplementedError
#        # Mass balance requires that each element in the reactant is preserved
#        # in the product and in any transition states.
#        for element in self.reactants.elements:
#            assert self.reactants.elements[element] == self.products.elements[element]
#            if self.ts is not None:
#                assert self.reactants.elements[element] == self.ts.elements[element]
        self.adsorption = adsorption
        self.method = method
        if isinstance(self.method, str):
            self.method =self.method.upper()
        if self.adsorption and self.method not in ['EQUIL', 'CT']:
            raise ValueError, "Method {} unrecognized!".format(self.method)
        self.S0 = S0
        self.keq = None
        self.kfor = None
        self.krev = None
        self.T = None
        self.N0 = None
        self.scale_params=['dH', 'dS', 'dH_act', 'dS_act']
        self.scale = {}
        for param in self.scale_params:
            self.scale[param] = 1.0
        self.scale_old = self.scale.copy()

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

    def update(self, T=None, N0=None):
        if not self.is_update_needed(T, N0):
            return

        self.T = T
        self.N0 = N0
        self.dH = self.products.get_H(T) - self.reactants.get_H(T)
        self.dH *= self.scale['dH']
        self.dS = self.products.get_S(T) - self.reactants.get_S(T)
        self.dS *= self.scale['dS']
        if self.ts is not None:
            self.dH_act = self.ts.get_H(T) - self.reactants.get_H(T)
            self.dH_act *= self.scale['dH_act']
            self.dS_act = self.ts.get_S(T) - self.reactants.get_S(T)
            self.dS_act *= self.scale['dS_act']
        self._calc_keq(T)
        self._calc_kfor(T, N0)
        self._calc_krev(T, N0)
        self.scale_old = self.scale.copy()

    def is_update_needed(self, T, N0):
        needed = True
        while needed:
            if self.keq is None:
                break
            if T is not None and T != self.T:
                break
            if np.any([self.scale[param] != self.scale_old[param] \
                    for param in self.scale_params]):
                break
            needed = False
        return needed

    def get_keq(self, T):
        self.update(T)
        return self.keq

    def get_kfor(self, T, N0):
        self.update(T, N0)
        return self.kfor

    def get_krev(self, T, N0):
        self.update(T, N0)
        return self.krev

    def _calc_keq(self, T):
        self.keq = np.exp(-self.dH / (kB * T) + self.dS / kB) \
                * self.products.get_reference_state() \
                / self.reactants.get_reference_state()

    def _calc_kfor(self, T, N0):
        if self.ts is None:
            if self.method == 'EQUIL':
                self.kfor = _k * T / _hplanck
                if self.keq < 1:
                    self.kfor *= self.keq
            elif self.method == 'CT':
                # Collision Theory
                # kfor = S0 / (N0 * sqrt(2 * pi * m * kB * T))
                self.kfor = (1000 * self.S0 * _Nav / N0) * np.sqrt(_k * T * kg \
                        / (2 * np.pi * self.reactants.get_mass()))
        else:
            #Transition State Theory
            self.kfor = (_k * T / _hplanck) * np.exp(-self.dH_act / (kB * T) \
                    + self.dS_act / kB) * self.ts.get_reference_state() \
                    / self.reactants.get_reference_state()

    def _calc_krev(self, T, N0):
        if self.ts is None:
            self.krev = _k * T / _hplanck
            if self.keq > 1:
                self.krev /= self.keq
            else:
                self.krev = self.kfor / self.keq
        else:
            self.krev = self.kfor / self.keq

    def __repr__(self):
        string = self.reactants.__repr__() + ' <-> '
        if self.ts is not None:
            string += self.ts.__repr__() + ' <-> '
        string += self.products.__repr__()
        return string

class Model(object):
    def __init__(self, reactions, vacancy, T, V, nsites, N0, coverage=1.0, z=0., nz=0, \
            shape='FLAT', steady_state=None, D=None):
        # Set up list of reactions and species
        self.reactions = []
        self.species = []
        for reaction in reactions:
            assert isinstance(reaction, Reaction)
            self.reactions.append(reaction)
            for species in reaction.species:
                self.add_species(species)
        self.add_species(vacancy)
        self.vacancy = vacancy

        self.steady_state = steady_state
        self.N0 = N0
        self.T = T
        self.V = V
        self.nsites = nsites
        self.coverage = coverage
        self.z = z
        self.nz = nz
        self.shape = shape
        self.D = D

        # Do we need to consider diffusion?
        self.diffusion = False
        for species in self.species:
            if isinstance(species, Liquid):
                assert nz > 3, "Must have at least three grid points for diffusion!"
                assert z > 0., "Must specify stagnant layer thickness for diffusion!"
                self.diffusion = True
                break
        self.method = 'Equil' if self.diffusion else 'CT'

        # Reorder species such that Liquid -> Gas -> Adsorbate -> Vacancy
        newspecies = []
        self.nliquid = 0
        for species in self.species:
            if isinstance(species, Liquid):
                assert species in self.D, \
                        "Specify diffusion constant for {}!".format(species)
                newspecies.append(species)
                self.nliquid += 1
        for species in self.species:
            if isinstance(species, Gas):
                newspecies.append(species)
        for species in self.species:
            if isinstance(species, Adsorbate):
                if species not in self.steady_state and species is not self.vacancy:
                    newspecies.append(species)
        for species in self.steady_state:
            newspecies.append(species)
        newspecies.append(self.vacancy)
        self.species = newspecies

        # Set up diffusion grid, if necessary
        if self.diffusion:
            self.set_up_grid()

    def add_species(self, species):
        assert isinstance(species, _Thermo)
        if species not in self.species:
            self.species.append(species)

    def set_up_grid(self):
        if self.shape.upper() == 'FLAT':
            self.dz = np.ones(self.nz - 1, dtype=float)
        elif self.shape.upper() == 'GAUSSIAN':
            self.dz = 1. / (np.exp(10 * (0.5 - np.arange(self.nz - 1, dtype=float) \
                    / (self.nz - 2))) + 1.)
        elif self.shape.upper() == 'EXP':
            self.dz = np.exp(7. * np.arange(self.nz - 1, dtype=float) / (self.nz - 2))
        self.dz *= self.z / self.dz.sum()
        self.zi = np.zeros(self.nz, dtype=float)
        for i in xrange(self.nz):
            # We don't calculate the width of the last grid point here,
            # that is done later to ensure that the sum over all grid points
            # gives the total system volume.
            if i < self.nz - 1:
                self.zi[i] += self.dz[i]/2.
                if i > 0:
                    self.zi[i] += self.dz[i-1]/2.

    def _rate_calc(self):
        self.rates = []
        self.rate_count = []
        self.is_rate_ads = []
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
            if reaction.adsorption:
                for species in reaction.species:
                    if isinstance(species, Liquid):
                        rate_count[species] *= self.nsites / self.dV[0]
                    elif isinstance(species, Gas):
                        rate_count[species] *= self.nsites / self.V
            self.rates.append(rate_for - rate_rev)
            self.rate_count.append(rate_count)
            self.is_rate_ads.append(reaction.adsorption)

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
            if isinstance(species, Liquid):
                for i in xrange(self.nz):
                    self.U0.append(U0i)
            else:
                self.U0.append(U0i)
        self._initialize()

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

    def _initialize(self):
        size = (self.nz - 1) * self.nliquid + len(self.species)
        self.M = np.identity(size, dtype=int)
        i = 0
        for species in self.species:
            if isinstance(species, Liquid):
                i += self.nz
            else:
                if species in self.steady_state or species is self.vacancy:
                    self.M[i, i] = 0
                i += 1
        self.symbols = sym.symbols('x0:{}'.format(size))
        self.symbols_dict = {}
        j = 0
        for species in self.species:
            if isinstance(species, Liquid):
                self.symbols_dict[species] = self.symbols[j]
                for i in xrange(self.nz):
                    self.symbols_dict[(species, i)] = self.symbols[j]
                    j += 1
            else:
                self.symbols_dict[species] = self.symbols[j]
                j += 1

        # Volume of grid thickness per ads. site
        if self.diffusion:
            self.dV = mol * self.zi * self.nsites * 1000 / (self.N0 * self.coverage)
            self.dV[-1] = self.V - self.dV.sum()
            self.zi[-1] = self.N0 * self.coverage * self.dV[-1] \
                    / (1000 * mol * self.nsites)
            self.dz = list(self.dz)
            self.dz.append(2 * self.zi[-1] - self.dz[-2])
            self.dz = np.array(self.dz)
        self._rate_calc()
        self.f_sym = []
        self.f_exec = []
        for species in self.species:
            if isinstance(species, Liquid):
                for i in xrange(self.nz):
                    f = 0
                    diff = self.D[species] / self.zi[i]
                    if i > 0:
                        if not (i == self.nz - 1 and species not in self.steady_state):
                            f += diff * (self.symbols_dict[(species, i-1)] \
                                    - self.symbols_dict[(species, i)]) / self.dz[i-1]
                    if i < self.nz - 1:
                        f += diff * (self.symbols_dict[(species, i+1)] \
                                - self.symbols_dict[(species, i)]) / self.dz[i]
                    if i == 0:
                        for j, rate in enumerate(self.rates):
                            if self.is_rate_ads[j]:
                                f += self.rate_count[j][species] * rate
                    self.f_sym.append(f)
                    self.f_exec.append(sym.lambdify(self.symbols, f))
                else:
                    continue
            elif isinstance(species, Gas):
                f = 0
                for j, rate in enumerate(self.rates):
                    f += self.rate_count[j][species] * rate
            elif species is not self.vacancy:
                f = 0
                for i, rate in enumerate(self.rates):
                    f += self.rate_count[i][species] * rate
            else:
                f = 1
                for a in self.species:
                    if not isinstance(a, _Fluid):
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
        self.U = []
        for i, t in enumerate(self.t):
            Ui = {}
            j = 0
            for species in self.species:
                if isinstance(species, Liquid):
                    for k in xrange(self.nz):
                        Ui[(species, k)] = self.U1[i][j]
                        j += 1
                    Ui[species] = self.U1[i][j-1]
                else:
                    Ui[species] = self.U1[i][j]
                    j += 1
            self.U.append(Ui)
        return self.U, self.t
