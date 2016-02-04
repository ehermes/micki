"""Microkinetic modeling objects"""

from __future__ import print_function

import os
import tempfile
import shutil

import numpy as np

from ase.units import kB, _hplanck, kg, _k, _Nav, mol

import sympy as sym

from micki.reactants import _Thermo, _Fluid, _Reactants, Gas, Liquid, Adsorbate
from micki.reactants import DummyFluid, DummyAdsorbate, Electron

from assimulo.problem import Implicit_Problem

from assimulo.solvers import IDA

from copy import copy

import warnings

import time


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
            raise ValueError("Method {} unsupported for adsorption reactions!".format(self.method))
        self.S0 = S0
        self.keq = None
        self.kfor = None
        self.krev = None
        self.T = None
        self.Asite = None
        self.scale_params=['dH', 'dS', 'dH_act', 'dS_act', 'kfor', 'krev']
        self.scale = {}
        for param in self.scale_params:
            self.scale[param] = 1.0
        self.scale_old = self.scale.copy()

        self.Nreact_fluid = 0
        self.Nreact_ads = 0
        for species in self.reactants:
            if isinstance(species, _Fluid):
                self.Nreact_fluid += 1
            elif isinstance(species, Adsorbate):
                self.Nreact_ads += 1

        self.Nprod_fluid = 0
        self.Nprod_ads = 0
        for species in self.products:
            if isinstance(species, _Fluid):
                self.Nprod_fluid += 1
            elif isinstance(species, Adsorbate):
                self.Nprod_ads += 1

        self.Nfluid = self.Nreact_fluid + self.Nprod_fluid
        self.Nads = self.Nreact_ads + self.Nprod_ads

    def get_scale(self, param):
        try:
            return self.scale[param]
        except KeyError:
            print("{} is not a valid scaling parameter name!".format(param))
            return None

    def set_scale(self, param, value):
        try:
            self.scale[param] = value
        except KeyError:
            print("{} is not a valid scaling parameter name!".format(param))

    def update(self, T=None, Asite=None, force=False):
        if not force and not self.is_update_needed(T, Asite):
            return

        self.T = T
        self.Asite = Asite
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

            # If there is a coverage dependence, assume everything has coverage 0
            dG_act = self.dG_act
            if isinstance(dG_act, sym.Basic):
                subs = {}
                for atom in dG_act.atoms():
                    if isinstance(atom, sym.Symbol):
                        subs[atom] = 0.
                dG_act = dG_act.subs(subs)

            if dG_act < 0.:
                warnings.warn('Negative activation energy found for {}. '
                        'Rounding to 0.'.format(self), RuntimeWarning, \
                        stacklevel=2)
                self.dG_act = 0.

            dG_rev = self.dG_act - self.dG
            if isinstance(dG_rev, sym.Basic):
                subs = {}
                for atom in dG_rev.atoms():
                    if isinstance(atom, sym.Symbol):
                        subs[atom] = 0.
                dG_rev = dG_rev.subs(subs)

            if dG_rev < 0.:
                warnings.warn('Negative activation energy found for {}. '
                        'Rounding to {}'.format(self, self.dG), RuntimeWarning, \
                        stacklevel=2)
                self.dG_act = self.dG
        self._calc_keq(T)
        self._calc_kfor(T, Asite)
        self._calc_krev(T, Asite)
        self.scale_old = self.scale.copy()

    def is_update_needed(self, T, Asite):
        for species in self.species:
            if species.is_update_needed(T):
                return True
        if self.keq is None:
            return True
        if T is not None and T != self.T:
            return True
        if Asite is not None and Asite != self.Asite:
            return True
        for param in self.scale_params:
            if self.scale[param] != self.scale_old[param]:
                return True
        return False

    def get_keq(self, T, Asite):
        self.update(T, Asite)
        return self.keq

    def get_kfor(self, T, Asite):
        self.update(T, Asite)
        return self.kfor

    def get_krev(self, T, Asite):
        self.update(T, Asite)
        return self.krev

    def _calc_keq(self, T):
        self.keq = sym.exp(-self.dG / (kB * T)) \
                * self.products.get_reference_state() \
                / self.reactants.get_reference_state() \
                * self.scale['kfor'] / self.scale['krev']

    def _calc_kfor(self, T, Asite):
        barr = 1
        if self.ts is not None:
            barr *= sym.exp(-self.dG_act / (kB * T)) \
                    * self.ts.get_reference_state() \
                    / self.reactants.get_reference_state()
        if self.method == 'EQUIL':
            self.kfor = _k * T / _hplanck
            if isinstance(self.keq, sym.Basic):
                subs = {}
                for atom in self.keq.atoms():
                    if isinstance(atom, sym.Symbol):
                        subs[atom] = 0.
                keq = self.keq.subs(subs)
            else:
                keq = self.keq
            if keq < 1:
                self.kfor *= self.keq * self.scale['krev'] / self.scale['kfor']
        elif self.method == 'CT':
            # Collision Theory
            # kfor = S0 * Asite / (sqrt(2 * pi * m * kB * T))
            self.kfor = barr * 1000 * self.S0 * _Nav * Asite \
                    * np.sqrt(_k * T * kg \
                    / (2 * np.pi * self.reactants.get_mass()))
        elif self.method == 'TST':
            #Transition State Theory
            self.kfor = (_k * T / _hplanck) * barr
        else:
            raise ValueError("Method {} is not recognized!".format(self.method))

    def _calc_krev(self, T, Asite):
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

    def _calc_keq(self, T, rhoref):
        self.dS = 0.
        self.dH = self.products.get_H(T) - self.reactants.get_H(T)
        for species in self.reactants:
            if isinstance(species, DummyFluid):
                self.dS -= species.Stot
            elif isinstance(species, DummyAdsorbate):
                self.dS -= species.get_S_gas(T)
            else:
                raise ValueError("Must pass dummy object!")
        for species in self.products:
            if isinstance(species, DummyFluid):
                self.dS += species.Stot
            elif isinstance(species, DummyAdsorbate):
                self.dS += species.get_S_gas(T)
            else:
                raise ValueError("Must pass dummy object!")
        self.keq = sym.exp(-self.dH / (kB * T) + self.dS / kB) \
                * self.products.get_reference_state() \
                / self.reactants.get_reference_state() \
                * self.scale['kfor'] / self.scale['krev'] \
                * rhoref**(self.Nreact_fluid - self.Nprod_fluid)

    def _calc_kfor(self, T, Asite, rhoref):
        if self.method == 'CT':
            self.kfor = 1000 * self.S0 * _Nav * Asite * rhoref * np.sqrt(_k * T * kg \
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
        self.kfor *= rhoref**(self.Nreact_fluid - 1)

    def _calc_krev(self, T, Asite, rhoref):
        self.krev = self.kfor / self.keq        


class Model(object):
#    def __init__(self, reactions, vacancy, T, V, nsites, N0, coverage=1.0, z=0., nz=0, \
#            shape='FLAT', steady_state=[], fixed=[], D=None, solvent=None, U0=None, fortran=False):
    def __init__(self, reactions, vacancy, T, Asite, rhoref=1, coverage=1.0, z=0., nz=0, \
            shape='FLAT', steady_state=[], fixed=[], D=None, solvent=None, U0=None, V=None, fortran=False):
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
        self.T = T
        self.Asite = Asite
        self.rhoref = rhoref
        self.coverage = coverage
        self.z = z
        self.nz = nz
        self.shape = shape
        self.D = D
        self.fortran = fortran
        self.V = V

        # Do we need to consider diffusion?
        self.diffusion = False
        for species in self.species:
            if isinstance(species, Liquid):
                assert nz > 3, "Must have at least three grid points for diffusion!"
                assert z > 0., "Must specify stagnant layer thickness for diffusion!"
                assert self.V is not None, "Must define volume for systems with diffusion!"
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
            if isinstance(species, Gas) or isinstance(species, DummyFluid) or isinstance(species, Electron):
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

        self.initialized = False
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
        for i in range(self.nz):
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

            rate_for = reaction.get_kfor(self.T, self.Asite)
            if reaction.method == 'CT':
                rate_for *= self.rhoref
            elif reaction.Nfluid != 0:
                assert reaction.Nads == 0, "Error: Reaction not using CT, but has both Fluid and Adsorbate species!"
                rate_for *= self.rhoref**(reaction.Nreact_fluid - 1)
            for species in reaction.reactants:
                rate_for *= self.symbols_dict[species]
                rate_count[species] -= 1

            rate_rev = reaction.get_krev(self.T, self.Asite)
            if reaction.method == 'CT':
                rate_rev *= self.rhoref
            if reaction.Nfluid != 0:
                rate_rev *= self.rhoref**(reaction.Nprod_fluid - 1)
            for species in reaction.products:
                rate_rev *= self.symbols_dict[species]
                rate_count[species] += 1

            if self.trans_cov_symbols is not None:
                if isinstance(rate_for, sym.Basic):
                    rate_for = rate_for.subs(self.trans_cov_symbols)
                if isinstance(rate_rev, sym.Basic):
                    rate_rev = rate_rev.subs(self.trans_cov_symbols)
#            if reaction.adsorption:
#                for species in reaction.reactants:
#                    if isinstance(species, (_Fluid, DummyFluid)):
#                        rate_count[species] *= self.rhoref
            self.rates.append(rate_for - rate_rev)
            self.rate_count.append(rate_count)
            self.is_rate_ads.append(reaction.adsorption)

    def rate_calc(self, U):
        self.update(U)
        return self.rates_last

    def set_initial_conditions(self, U0):
        if self.initialized and self.fortran:
            self.finalize()
        self.U0 = U0
        occsites = 0
        for species in self.U0:
            if type(species) is tuple:
                species = species[0]
            if species not in self.species:
                raise ValueError("Unknown species!")
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
                    for i in range(self.nz):
                        if (species, i) not in self.U0:
                            self.U0[(species, i)] = U0i
                        else:
                            U0i = self.U0[(species, i)]
        size = (self.nz - 1) * (self.nliquid - 1) + len(self.species)
        M = np.ones(size, dtype=int)
        i = 0
        for species in self.species:
            if isinstance(species, Liquid) and species is not self.solvent:
                i += self.nz
            else:
                if species in self.steady_state or species is self.vacancy:
                    M[i] = 0
                i += 1

#        self.symbols_all = sym.symbols('modelparam0:{}'.format(size))
        self.symbols_all = sym.symbols(' '.join(['modelparam{}'.format(str(i).zfill(3)) for i in range(size)]))
        self.symbols_dict = {}
        self.symbols = []
        i = 0
        for species in self.species:
            if isinstance(species, Liquid):
                self.symbols_dict[species] = self.symbols_all[i]
                if species is not self.solvent:
                    for j in range(self.nz):
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
            if species.symbol is not None:
                self.trans_cov_symbols[species.symbol] = self.symbols_dict[species]
        
        self.M = np.zeros((self.nsymbols, self.nsymbols), dtype=int)
        algvar = np.zeros(self.nsymbols, dtype=bool)
        for i, symboli in enumerate(self.symbols_all):
            for j, symbolj in enumerate(self.symbols):
                if symboli == symbolj:
                    self.M[j, j] = M[i]
                    algvar[j] = M[i]

        # Volume of grid thickness per ads. site
        if self.diffusion:
            self.dV = mol * self.zi * self.nsites * 1000 * self.Asite / self.coverage
            self.dV[-1] = self.V - self.dV.sum()
            self.zi[-1] = self.coverage * self.dV[-1] \
                    / (1000 * mol * self.nsites * self.Asite)
            self.dz = list(self.dz)
            self.dz.append(2 * self.zi[-1] - self.dz[-2])
            self.dz = np.array(self.dz)
        self._rate_init()
        self.f_sym = []

        for species in self.species:
            f = 0
            if isinstance(species, Liquid) and species is not self.solvent:
                for i in range(self.nz):
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
            elif species is not self.vacancy and species not in self.fixed:
                for i, rate in enumerate(self.rates):
                    f += self.rate_count[i][species] * rate
            elif species is self.vacancy:
                f = 1
                for a in self.species:
                    if isinstance(a, Adsorbate):
                        f -= self.symbols_dict[a] * a.coord
            if species not in self.fixed:
                self.f_sym.append(f)
            else:
                assert f == 0, "Fixed species rate of change not zero!"

        subs = {}
        for species in self.species:
            if species in self.fixed:
                liquid = False
                if isinstance(species, Liquid) and species is not self.solvent:
                    species = (species, self.nz - 1)
                    liquid = True
                U0i = self.U0[species]
                if liquid or isinstance(species, (_Fluid, DummyFluid)):
                    U0i /= self.rhoref
                subs[self.symbols_dict[species]] = U0i 

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
            for species, isymbol in self.symbols_dict.items():
                if symbol == isymbol:
                    U0i = self.U0[species]
                    if type(species) is tuple or isinstance(species, (_Fluid, DummyFluid)):
                        U0i /= self.rhoref
                    U0.append(U0i)
                    break

        if self.fortran:
            self.finitialize(U0, 1e-9, [1e-11]*self.nsymbols, [], [], algvar)
        else:
            self.last_x = np.zeros_like(U0, dtype=float)
            self.update(np.array(U0))
            self.problem = Implicit_Problem(res=self.res, y0=U0, yd0=self.f(U0), t0=0.)
            self.problem.jac = self.jac
            self.sim = IDA(self.problem)
            self.sim.verbosity = 50
            self.sim.algvar = algvar
            self.sim.atol = 1e-8
            self.sim.rtol = 1e-6
            try:
                nthreads = os.environ['OMP_NUM_THREADS']
            except KeyError:
                nthreads = 1
            self.sim.num_threads = nthreads
        self.initialized = True

    def setup_execs(self):
        expressions = []
        expressions.extend(self.f_sym)
        expressions.extend(self.jac_sym.reshape(self.nsymbols**2))
        expressions.extend(self.rates)

        if self.fortran:
            from micki.fortran import f90_template, pyf_template
            from numpy import f2py

            y_vec = sym.IndexedBase('yin', shape=(self.nsymbols,))
            trans = {self.symbols[i]: y_vec[i + 1] for i in range(self.nsymbols)}
            str_trans = {sym.fcode(self.symbols[i], source_format='free'): sym.fcode(y_vec[i + 1], source_format='free') for i in range(self.nsymbols)}
    
            rescode = []
            jaccode = []
            ratecode = []
    
            for i in range(self.nsymbols):
                fcode = sym.fcode(self.f_sym[i], source_format='free')
                for key, val in str_trans.items():
                    fcode = fcode.replace(key, val)
                rescode.append('   res({}) = '.format(i + 1) + fcode)
    
            for i in range(self.nsymbols):
                for j in range(self.nsymbols):
                    expr = self.jac_sym[j, i]
                    if expr != 0:
                        fcode = sym.fcode(expr, source_format='free')
                        for key, val in str_trans.items():
                            fcode = fcode.replace(key, val)
                        jaccode.append('   jac({}, {}) = '.format(j + 1, i + 1) + fcode)

            for i, rate in enumerate(self.rates):
                fcode = sym.fcode(rate, source_format='free')
                for key, val in str_trans.items():
                    fcode = fcode.replace(key, val)
                ratecode.append('   rates({}) = '.format(i + 1) + fcode)

            program = f90_template.format(neq=self.nsymbols, nx=1, nrates=len(self.rates), rescalc='\n'.join(rescode),
                    jaccalc='\n'.join(jaccode), ratecalc='\n'.join(ratecode))

            dname = tempfile.mkdtemp()
            modname = os.path.split(dname)[1]
            fname = modname + '.f90'
            pyfname = modname + '.pyf'

            with open('solve_ida.f90', 'w') as f:
                f.write(program)

            with open(os.path.join(dname, pyfname), 'w') as f:
                f.write(pyf_template.format(modname=modname, neq=self.nsymbols, nrates=len(self.rates)))

            f2py.compile(program, modulename=modname, extra_args='--compiler=intelem --fcompiler=intelem --quiet '
            '--f90flags="-O3" '
            '/usr/local/tmp/lib/libsundials_fida.a /usr/local/tmp/lib/libsundials_ida.a '
            '/usr/local/tmp/lib/libsundials_fnvecserial.a /usr/local/tmp/lib/libsundials_nvecserial.a '
            '/opt/intel/composerxe-2013.3.174/mkl/lib/intel64/libmkl_rt.so ' +
            os.path.join(dname, pyfname), source_fn=os.path.join(dname, fname), verbose=0)

            shutil.rmtree(dname)

            solve_ida = __import__(modname)

            self.finitialize = solve_ida.initialize
            self.fsolve = solve_ida.solve
            self.ffinalize = solve_ida.finalize

            os.remove(modname + '.so')


    def update(self, x):
        if (x != self.last_x).any():
            self.last_x = x
            out = np.zeros(len(self.execs), dtype=float)
#            for expression in self.subexp:
#                x = np.append(x, expression(*x))
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
        if self.fortran:
            self.t, U1, dU1, r1 = self.fsolve(self.nsymbols, len(self.rates), ncp, t)
            self.U1 = U1.T
            self.dU1 = dU1.T
            self.r1 = r1.T
            self.U = []
            self.dU = []
            self.r = []
            for i, t in enumerate(self.t):
                Ui = {}
                dUi = {}
                ri = {}
                for species in self.fixed:
                    if isinstance(species, Liquid):
                        species = (species, self.nz - 1)
                    dUi[species] = 0.
                    Ui[species] = self.U0[species]
                for j, symbol in enumerate(self.symbols):
                    for species, isymbol in self.symbols_dict.items():
                        if symbol == isymbol:
                            Uij = self.U1[i][j]
                            dUij = self.dU1[i][j]
                            if type(species) is tuple or isinstance(species, (_Fluid, DummyFluid)):
                                Uij *= self.rhoref
                                dUij *= self.rhoref
                            Ui[species] = Uij
                            dUi[species] = dUij
                for j, reaction in enumerate(self.reactions):
                    ri[reaction] = r1[j][i]
                self.U.append(Ui)
                self.dU.append(dUi)
                self.r.append(ri)
            return self.U, self.dU, self.r
        else:
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
                for species, isymbol in self.symbols_dict.items():
                    if symbol == isymbol:
                        Uij = self.U1[i][j]
                        dUij = dU1[j]
                        if type(species) is tuple or isinstance(species, (_Fluid, DummyFluid)):
                            Uij *= self.rhoref
                            dUij *= self.rhoref
                        Ui[species] = Uij
                        dUi[species] = dUij
            self.U.append(Ui)
            self.dU.append(dUi)
            
            r = self.rate_calc(self.U1[i])
            ri = {}
            for j, reaction in enumerate(self.reactions):
                ri[reaction] = r[j]
            self.r.append(ri)
        return self.U, self.dU, self.r

    def finalize(self):
        self.initialized = False
#        self.ffinalize()
    
    def copy(self, initialize=True):
        if initialize:
            U0 = self.U0
        else:
            U0 = None
        return Model(self.reactions, self.vacancy, self.T, self.Asite, self.rhoref, \
                self.coverage, self.z, self.nz, self.shape, \
                self.steady_state, self.fixed, self.D, self.solvent, U0, self.fortran)
