"""Microkinetic modeling objects"""

import numpy as np

from scipy.integrate import odeint

from ase.units import kB, _hplanck, kg, _k, _Nav, mol

import sympy as sym

from mkm.reactants import _Thermo, _Fluid, _Reactants, Gas, Liquid, Adsorbate

from assimulo.problem import Implicit_Problem

from assimulo.solvers import IDA

from copy import copy


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
        self.scale_params=['dH', 'dS', 'dH_act', 'dS_act', 'kfor', 'krev']
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
        if self.keq is None:
            return True
        if T is not None and T != self.T:
            return True
        if N0 is not None and N0 != self.N0:
            return True
        for param in self.scale_params:
            if self.scale[param] != self.scale_old[param]:
                return True
        return False

    def get_keq(self, T, N0):
        self.update(T, N0)
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
                / self.reactants.get_reference_state() \
                * self.scale['kfor'] / self.scale['krev']

    def _calc_kfor(self, T, N0):
        if self.ts is None:
            if self.method == 'EQUIL':
                self.kfor = _k * T / _hplanck
                if self.keq < 1:
                    self.kfor *= self.keq * self.scale['krev'] / self.scale['kfor']
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
        self.kfor *= self.scale['kfor']

    def _calc_krev(self, T, N0):
        self.krev = self.kfor / self.keq

    def __repr__(self):
        string = self.reactants.__repr__() + ' <-> '
        if self.ts is not None:
            string += self.ts.__repr__() + ' <-> '
        string += self.products.__repr__()
        return string

class Model(object):
    def __init__(self, reactions, vacancy, T, V, nsites, N0, coverage=1.0, z=0., nz=0, \
            shape='FLAT', steady_state=[], fixed=[], D=None, solvent=None):
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
        self.solvent = solvent
        self.fixed = fixed
        if self.solvent is not None and self.solvent not in self.fixed:
            self.fixed.append(self.solvent)
        for species in self.fixed:
            assert species in self.species, "Unknown fixed species {}".format(species)
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

        # Reorder species such that Liquid -> Gas -> Adsorbate -> Vacancy
        newspecies = []
        self.nliquid = 0
        for species in self.species:
            if isinstance(species, Liquid):
                assert species in self.D, \
                        "Specify diffusion constant for {}!".format(species)
                newspecies.append(species)
                self.nliquid += 1
                assert self.solvent is not None, "Must specify solvent!"
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

    def _rate_init(self):
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
#            self.rates_exec.append(sym.lambdify(self.symbols, rate_for - rate_rev))
            self.rate_count.append(rate_count)
            self.is_rate_ads.append(reaction.adsorption)

    def rate_calc(self, U):
        r = np.zeros_like(self.rates_exec, dtype=float)
        for i in xrange(len(self.reactions)):
            r[i] = self.rates_exec[i](*U)
        return r

    def set_initial_conditions(self, U0):
        self.U0 = U0
        occsites = 0
        for species in self.U0:
            if type(species) is tuple:
                species = species[0]
            if species not in self.species:
                raise ValueError, "Unknown species!"
            if isinstance(species, Adsorbate) and species is not self.vacancy:
                occsites += self.U0[species] * species.coord
            elif isinstance(species, Liquid):
                if (species, self.nz - 1) in self.U0:
                    assert abs(self.U0[species] - self.U0[(species, self.nz - 1)]) < 1e-6, \
                            "Liquid concentrations not consistent!"
        assert occsites <= 1., "Too many adsorbates!"
        if self.vacancy not in U0:
            U0[self.vacancy] = 1. - occsites
        else:
            assert abs(1. - U0[self.vacancy] - occsites) < 1e-6, \
                    "Vacancy concentration not consistent!"
        for species in self.species:
            if species not in self.U0:
                self.U0[species] = 0.
            if isinstance(species, Liquid):
                U0i = self.U0[species]
                if species is not self.solvent:
                    for i in xrange(self.nz):
                        if (species, i) not in self.U0:
                            self.U0[(species, i)] = U0i
                        else:
                            U0i = self.U0[(species, i)]
        self._initialize()

    def f(self, x, t=None):
        y = np.zeros_like(self.f_exec, dtype=float)
        for i in xrange(len(self.symbols)):
            y[i] = self.f_exec[i](*x)
        return y

    def jac(self, c, t, y, yd):
        jac = np.zeros_like(self.jac_exec, dtype=float)
        for i in xrange(len(self.symbols)):
            for j in xrange(len(self.symbols)):
                jac[i, j] = self.jac_exec[i][j](*y)
        jac -= c * self.M
        return jac

    def mas(self):
        return self.M

    def res(self, t, x, s):
        return self.f(x, t) - np.dot(self.M, s)

    def adda(self, x, t, p):
        return p + self.M

    def solve(self, t, ncp):
        self.sim = IDA(self.model)
        self.sim.verbosity = 50
        self.t, self.U1, self.dU1 = self.sim.simulate(t, ncp)
        return self._results()

    def _initialize(self):
        size = (self.nz - 1) * (self.nliquid - 1) + len(self.species)
        M = np.ones(size, dtype=int)
#        self.M = np.identity(size, dtype=int)
        i = 0
        for species in self.species:
            if isinstance(species, Liquid) and species is not self.solvent:
                i += self.nz
            else:
                if species in self.steady_state or species is self.vacancy:
                    M[i] = 0
#                    self.M[i, i] = 0
                i += 1
        self.symbols_all = sym.symbols('x0:{}'.format(size))
        self.symbols_dict = {}
        self.symbols = []
        i = 0
        for species in self.species:
            if isinstance(species, Liquid):
                self.symbols_dict[species] = self.symbols_all[i]
                if species is not self.solvent:
                    for j in xrange(self.nz):
                        self.symbols_dict[(species, j)] = self.symbols_all[i]
                        if not (j == self.nz - 1 and species in self.fixed):
                            self.symbols.append(self.symbols_all[i])
                        i += 1
                else:
                    self.symbols_dict[(species, 0)] = self.symbols_all[i]
                    self.symbols_dict[(species, 9)] = self.symbols_all[i]
                    i += 1
#                self.symbols_dict[species] = self.symbols_all[i - 1]
            else:
                self.symbols_dict[species] = self.symbols_all[i]
                if species not in self.fixed:
                    self.symbols.append(self.symbols_all[i])
                i += 1
        self.M = np.zeros((len(self.symbols), len(self.symbols)), dtype=int)
        for i, symboli in enumerate(self.symbols_all):
            for j, symbolj in enumerate(self.symbols):
                if symboli == symbolj:
                    self.M[j, j] = M[i]

        # Volume of grid thickness per ads. site
        if self.diffusion:
            self.dV = mol * self.zi * self.nsites * 1000 / (self.N0 * self.coverage)
            self.dV[-1] = self.V - self.dV.sum()
            self.zi[-1] = self.N0 * self.coverage * self.dV[-1] \
                    / (1000 * mol * self.nsites)
            self.dz = list(self.dz)
            self.dz.append(2 * self.zi[-1] - self.dz[-2])
            self.dz = np.array(self.dz)
        self._rate_init()
        self.f_sym = []
        for species in self.species:
            f = 0
            if isinstance(species, Liquid) and species is not self.solvent:
                for i in xrange(self.nz):
                    f = 0
                    diff = self.D[species] / self.zi[i]
                    if i > 0:
#                        if not (i == self.nz - 1 and species in self.fixed):
                        f += diff * (self.symbols_dict[(species, i-1)] \
                                - self.symbols_dict[(species, i)]) / self.dz[i-1]
                    if i < self.nz - 1:
                        f += diff * (self.symbols_dict[(species, i+1)] \
                                - self.symbols_dict[(species, i)]) / self.dz[i]
                    if i == 0:
                        for j, rate in enumerate(self.rates):
                            if self.is_rate_ads[j]:
                                f += self.rate_count[j][species] * rate
                    if not (i == self.nz - 1 and species in self.fixed):
                        self.f_sym.append(f)
#                    self.f_exec.append(sym.lambdify(self.symbols, f))
                else:
                    continue
            elif isinstance(species, Gas):
                if species not in self.fixed:
                    for j, rate in enumerate(self.rates):
                        f += self.rate_count[j][species] * rate
            elif isinstance(species, Adsorbate) and species is not self.vacancy:
                for i, rate in enumerate(self.rates):
                    f += self.rate_count[i][species] * rate
            elif species is self.vacancy:
                f = 1
                for a in self.species:
                    if not isinstance(a, _Fluid):
                        f -= self.symbols_dict[a] * a.coord
            if species not in self.fixed:
                self.f_sym.append(f)
            else:
                assert f == 0, "Fixed species rate of change not zero!"
#            self.f_exec.append(sym.lambdify(self.symbols, f))

        subs = {}
        for species in self.species:
            if species in self.fixed:
                if isinstance(species, Liquid) and species is not self.solvent:
                    species = (species, self.nz - 1)
                subs[self.symbols_dict[species]] = self.U0[species]

        self.f_exec = []
        for i, f in enumerate(self.f_sym):
            self.f_sym[i] = f.subs(subs)
            self.f_exec.append(sym.lambdify(self.symbols, self.f_sym[i]))
        self.jac_exec = []
        for f in self.f_sym:
            jac_exec = []
            for symbol in self.symbols:
                jac_exec.append(sym.lambdify(self.symbols, sym.diff(f, \
                        symbol)))
            self.jac_exec.append(jac_exec)
        self.rates_exec = []
        for i, r in enumerate(self.rates):
            self.rates[i] = r.subs(subs)
            self.rates_exec.append(sym.lambdify(self.symbols, self.rates[i]))
        U0 = []
        for symbol in self.symbols:
            for species, isymbol in self.symbols_dict.iteritems():
                if symbol == isymbol:
                    U0.append(self.U0[species])
                    break
#        for species in self.species:
#            U0.append(self.U0[species])
        self.model = Implicit_Problem(self.res, U0, self.f(U0), 0.)
        self.model.jac = self.jac

    def _results(self):
        self.U = []
        self.dU = []
        self.r = []
        for i, t in enumerate(self.t):
            dU1 = self.f(self.U1[i], t)
            Ui = copy(self.U0)
            dUi = {}
            for species in self.species:
                if species in self.fixed:
                    if isinstance(species, Liquid):
                        species = (species, self.nz - 1)
                    dUi[species] = 0.
            j = 0
            for j, symbol in enumerate(self.symbols):
                for species, isymbol in self.symbols_dict.iteritems():
                    if symbol == isymbol:
                        Ui[species] = self.U1[i][j]
                        dUi[species] = dU1[j]
#            for species in self.species:
#                if isinstance(species, Liquid):
#                    for k in xrange(self.nz):
#                        Ui[(species, k)] = self.U1[i][j]
#                        dUi[(species, k)] = dU1[j]
#                        j += 1
#                    Ui[species] = self.U1[i][j-1]
#                    dUi[species] = dU1[j-1]
#                else:
#                    Ui[species] = self.U1[i][j]
#                    dUi[species] = dU1[j]
#                    j += 1
            self.U.append(Ui)
            self.dU.append(dUi)

            r = self.rate_calc(self.U1[i])
            ri = {}
            for j, reaction in enumerate(self.reactions):
                ri[reaction] = r[j]
            self.r.append(ri)
        return self.U, self.dU, self.r
