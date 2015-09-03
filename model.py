"""Microkinetic modeling objects"""

import os

import numpy as np

from ase.units import kB, _hplanck, kg, _k, _Nav, mol

import sympy as sym

from micki.reactants import _Thermo, _Fluid, _Reactants, Gas, Liquid, Adsorbate
from micki.reactants import DummyFluid, DummyAdsorbate

from assimulo.problem import Implicit_Problem

from assimulo.solvers import IDA

from copy import copy

import warnings


class Reaction(object):
    def __init__(self, reactants, products, ts=None, method=None, S0=1., \
            adsorption=False, vacancy=None):
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
        netcoord = 0
        for species in self.reactants:
            if species not in self.species:
                self.species.append(species)
            netcoord -= species.coord
        for species in self.products:
            if species not in self.species:
                self.species.append(species)
            netcoord += species.coord

        if netcoord > 0:
            self.reactants += netcoord * vacancy
        elif netcoord < 0:
            self.products += -netcoord * vacancy

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
        if method is None:
            if self.ts is not None:
                self.method = 'TST'
            else:
                self.method = 'EQUIL'
        else:
            self.method = method
        if isinstance(self.method, str):
            self.method = self.method.upper()
        if self.adsorption and self.method not in ['EQUIL', 'CT']:
            raise ValueError, "Method {} unsupported for adsorption reactions!".format(self.method)
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
        self.dG = self.dH - self.T * self.dS
        if self.ts is not None:
            self.dH_act = self.ts.get_H(T) - self.reactants.get_H(T)
            self.dH_act *= self.scale['dH_act']
            self.dS_act = self.ts.get_S(T) - self.reactants.get_S(T)
            self.dS_act *= self.scale['dS_act']
            self.dG_act = self.dH_act - self.T * self.dS_act
            if self.dG_act < 0.:
                warnings.warn('Negative activation energy found for {}. \
                        Rounding to 0.'.format(self), RuntimeWarning, \
                        stacklevel=2)
                self.dG_act = 0.
            if self.dG_act - self.dG < 0.:
                warnings.warn('Negative activation energy found for {}. \
                        Rounding to {}'.format(self, self.dG), RuntimeWarning, \
                        stacklevel=2)
                self.dG_act = self.dG
        self._calc_keq(T)
        self._calc_kfor(T, N0)
        self._calc_krev(T, N0)
        self.scale_old = self.scale.copy()

    def is_update_needed(self, T, N0):
        for species in self.species:
            if species.is_update_needed(T):
                return True
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
        self.keq = sym.exp(-self.dG / (kB * T)) \
                * self.products.get_reference_state() \
                / self.reactants.get_reference_state() \
                * self.scale['kfor'] / self.scale['krev']

    def _calc_kfor(self, T, N0):
        barr = 1
        if self.ts is not None:
            barr *= sym.exp(-self.dG_act / (kB * T)) \
                    * self.ts.get_reference_state() \
                    / self.reactants.get_reference_state()
        if self.method == 'EQUIL':
            self.kfor = _k * T / _hplanck
            if self.keq < 1:
                self.kfor *= self.keq * self.scale['krev'] / self.scale['kfor']
        elif self.method == 'CT':
            # Collision Theory
            # kfor = S0 / (N0 * sqrt(2 * pi * m * kB * T))
            self.kfor = barr * (1000 * self.S0 * _Nav / N0) \
                    * np.sqrt(_k * T * kg \
                    / (2 * np.pi * self.reactants.get_mass()))
        elif self.method == 'TST':
            #Transition State Theory
            self.kfor = (_k * T / _hplanck) * barr
        else:
            raise ValueError, "Method {} is not recognized!".format(self.method)
        self.kfor *= self.scale['kfor']

    def _calc_krev(self, T, N0):
        self.krev = self.kfor / self.keq

    def __repr__(self):
        string = self.reactants.__repr__() + ' <-> '
        if self.ts is not None:
            string += self.ts.__repr__() + ' <-> '
        string += self.products.__repr__()
        return string

class DummyReaction(Reaction):
    def __init__(self, reactants, products, ts=None, method='EQUIL', S0=1., 
            vacancy=None, Ei=None, A=None, adsorption=False):
        Reaction.__init__(self, reactants, products, ts, method, S0, adsorption, 
                vacancy)
        self.Ei = Ei
        self.A = A

    def _calc_keq(self, T):
        self.dS = 0.
        self.dH = self.products.get_H(T) - self.reactants.get_H(T)
        for species in self.reactants:
            if isinstance(species, DummyFluid):
                self.dS -= species.Stot
            elif isinstance(species, DummyAdsorbate):
                self.dS -= species.get_S_gas(T)
            else:
                raise ValueError, "Must pass dummy object!"
        for species in self.products:
            if isinstance(species, DummyFluid):
                self.dS += species.Stot
            elif isinstance(species, DummyAdsorbate):
                self.dS += species.get_S_gas(T)
            else:
                raise ValueError, "Must pass dummy object!"
        self.keq = sym.exp(-self.dH / (kB * T) + self.dS / kB) \
                * self.products.get_reference_state() \
                / self.reactants.get_reference_state() \
                * self.scale['kfor'] / self.scale['krev']

    def _calc_kfor(self, T, N0):
        if self.method == 'CT':
            self.kfor = (1000 * self.S0 * _Nav / N0) * np.sqrt(_k * T * kg \
                    / (2 * np.pi * self.reactants.get_mass()))
        else:
            if self.A is not None:
                A = self.A
            else:
                dS_act = 0
                for ts in self.ts:
                    ts.update(T)
                    dS_act += ts.Svib
                for reactant in self.reactants:
                    reactant.update(T)
                    dS_act -= reactant.Svib
                A = (_k * T / _hplanck) * sym.exp(dS_act / kB)

            if self.Ei is not None:
                Ei = self.Ei
            else:
                Ei = 0
                for ts in self.ts:
                    ts.update(T)
                    Ei += ts.Eelec + ts.Evib
                for reactant in self.reactants:
                    reactant.update(T)
                    Ei -= reactant.Eelec + reactant.Evib
            self.kfor = A * sym.exp(-Ei / (kB * T))

    def _calc_krev(self, T, N0):
        self.krev = self.kfor / self.keq        


class Model(object):
    def __init__(self, reactions, vacancy, T, V, nsites, N0, coverage=1.0, z=0., nz=0, \
            shape='FLAT', steady_state=[], fixed=[], D=None, solvent=None, U0=None):
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
            if isinstance(species, Gas) or isinstance(species, DummyFluid):
                newspecies.append(species)
        for species in self.species:
            if isinstance(species, Adsorbate) or isinstance(species, DummyAdsorbate):
                if species not in self.steady_state and species is not self.vacancy:
                    newspecies.append(species)
        for species in self.steady_state:
            newspecies.append(species)
        newspecies.append(self.vacancy)
        self.species = newspecies

        # Set up diffusion grid, if necessary
        if self.diffusion:
            self.set_up_grid()

        self.U0 = U0

        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def add_species(self, species):
        assert isinstance(species, _Thermo)
        if species not in self.species:
            self.species.append(species)
    
    def set_temperature(self, T):
        self.T = T
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

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
            if self.trans_cov_symbols is not None:
                if isinstance(rate_for, sym.Basic):
                    rate_for = rate_for.subs(self.trans_cov_symbols)
                if isinstance(rate_rev, sym.Basic):
                    rate_rev = rate_rev.subs(self.trans_cov_symbols)
            if reaction.adsorption:
                for species in reaction.reactants:
                    if isinstance(species, Liquid):
                        rate_count[species] *= self.nsites / self.dV[0]
                    elif isinstance(species, (Gas, DummyFluid)):
                        rate_count[species] *= self.nsites / self.V
            self.rates.append(rate_for - rate_rev)
            self.rate_count.append(rate_count)
            self.is_rate_ads.append(reaction.adsorption)

    def rate_calc(self, U):
        self.update(U)
        return self.rates_last
#        r = np.zeros_like(self.rates_exec, dtype=float)
#        for i in xrange(len(self.reactions)):
#            r[i] = self.rates_exec[i](*U)
#        return r

    def set_initial_conditions(self, U0):
        self.U0 = U0
        occsites = 0
        for species in self.U0:
            if type(species) is tuple:
                species = species[0]
            if species not in self.species:
                raise ValueError, "Unknown species!"
            if (isinstance(species, (Adsorbate, DummyAdsorbate)) 
                    and species is not self.vacancy):
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
        self.symbols_all = sym.symbols('modelparam0:{}'.format(size))
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
        self.nsymbols = len(self.symbols)
        self.trans_cov_symbols = {}
        for species in self.species:
            if isinstance(species, Adsorbate) and species.symbol is not None:
                self.trans_cov_symbols[species.symbol] = self.symbols_dict[species]
#        self.trans_cov_symbols = None
#        if self.coverage_symbols is not None:
#            self.trans_cov_symbols = {}
#            for species, symbol in self.coverage_symbols.iteritems():
#                self.trans_cov_symbols[symbol] = self.symbols_dict[species]
            
        self.M = np.zeros((self.nsymbols, self.nsymbols), dtype=int)
        algvar = np.zeros(self.nsymbols, dtype=bool)
        for i, symboli in enumerate(self.symbols_all):
            for j, symbolj in enumerate(self.symbols):
                if symboli == symbolj:
                    self.M[j, j] = M[i]
                    algvar[j] = M[i]

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
                else:
                    continue
            elif isinstance(species, (Gas, DummyFluid)):
                if species not in self.fixed:
                    for j, rate in enumerate(self.rates):
                        f += self.rate_count[j][species] * rate
            elif (isinstance(species, (Adsorbate, DummyAdsorbate)) 
                    and species is not self.vacancy):
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

        subs = {}
        for species in self.species:
            if species in self.fixed:
                if isinstance(species, Liquid) and species is not self.solvent:
                    species = (species, self.nz - 1)
                subs[self.symbols_dict[species]] = self.U0[species]

        for i, f in enumerate(self.f_sym):
            self.f_sym[i] = sym.sympify(f).subs(subs)
        self.jac_sym = np.zeros((self.nsymbols, self.nsymbols), dtype=object)
        for i, f in enumerate(self.f_sym):
            for j, symbol in enumerate(self.symbols):
                self.jac_sym[i, j] = sym.diff(f, symbol)
        for i, r in enumerate(self.rates):
            self.rates[i] = sym.sympify(r).subs(subs)

        self.setup_execs()
        U0 = []
        for symbol in self.symbols:
            for species, isymbol in self.symbols_dict.iteritems():
                if symbol == isymbol:
                    U0.append(self.U0[species])
                    break
        self.last_x = np.zeros_like(U0, dtype=float)
        self.update(np.array(U0))
        self.problem = Implicit_Problem(res=self.res, y0=U0, yd0=self.f(U0), t0=0.)
        self.problem.jac = self.jac
        self.sim = IDA(self.problem)
        self.sim.verbosity = 50
        self.sim.algvar = algvar
        self.sim.atol = 1e-10
        self.sim.rtol = 1e-8
        try:
            nthreads = os.environ['OMP_NUM_THREADS']
        except KeyError:
            nthreads = 1
        self.sim.num_threads = nthreads
#        self.model.suppress_alg = True

    def setup_execs(self):
        expressions = []
        expressions.extend(self.f_sym)
        expressions.extend(self.jac_sym.reshape(self.nsymbols**2))
        expressions.extend(self.rates)

        sub, red = sym.cse(expressions)

        self.subexp = []
        subsym = []
        for xi, val in sub:
            self.subexp.append(sym.lambdify(self.symbols + subsym, val, modules="numpy"))
            subsym.append(xi)

        self.execs = [sym.lambdify(self.symbols + subsym, ired, modules="numpy") \
                for ired in red]

    def update(self, x):
        if (x != self.last_x).any():
            self.last_x = x
            out = np.zeros(len(self.execs), dtype=float)
            for expression in self.subexp:
                x = np.append(x, expression(*x))
            for i, expression in enumerate(self.execs):
                out[i] = expression(*x)
            self.f_last = out[:self.nsymbols]
            self.jac_last = out[self.nsymbols:self.nsymbols**2 + self.nsymbols].reshape(self.nsymbols, self.nsymbols)
            self.rates_last = out[-len(self.rates):]

    def f(self, x, t=None):
        self.update(x)
        return self.f_last

    def jac(self, c, t, y, yd):
        self.update(y)
        return self.jac_last - c * self.M

    def mas(self):
        return self.M

    def res(self, t, x, s):
        return self.f(x, t) - np.dot(self.M, s)

    def adda(self, x, t, p):
        return p + self.M

    def solve(self, t, ncp):
        self.t, self.U1, self.dU1 = self.sim.simulate(t, ncp)
        return self._results()

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
            self.U.append(Ui)
            self.dU.append(dUi)

            r = self.rate_calc(self.U1[i])
            ri = {}
            for j, reaction in enumerate(self.reactions):
                ri[reaction] = r[j]
            self.r.append(ri)
        return self.U, self.dU, self.r
    
    def copy(self):
        return Model(self.reactions, self.vacancy, self.T, self.V, self.nsites, \
                self.N0, self.coverage, self.z, self.nz, self.shape, \
                self.steady_state, self.fixed, self.D, self.solvent, self.U0)
