"""Microkinetic modeling objects"""

from __future__ import print_function

import os
import tempfile
import shutil
import warnings
import time

import numpy as np
import sympy as sym

from copy import copy
from ase.units import kB, _hplanck, kg, _k, _Nav, mol

from micki.reactants import _Thermo, _Fluid, _Reactants, Gas, Liquid, Adsorbate
from micki.reactants import DummyFluid, DummyAdsorbate, Electron


class Reaction(object):
    def __init__(self, reactants, products, ts=None, method=None, S0=1., \
            adsorption=False, dG_act=None, dground=True):

        # Wrap reactants and products in _Reactants type
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

        # Determine the number of sites on the LHS and the RHS of the reaction,
        # then add "bare" sites as necessary to balance the site number.
        vacancies = {}
        self.species = []
        for species in self.reactants:
            if species not in self.species:
                self.species.append(species)
            if species.sites is None:
                continue
            for site in species.sites:
                if site in vacancies:
                    vacancies[site] -= 1
                else:
                    vacancies[site] = -1
        for species in self.products:
            if species not in self.species:
                self.species.append(species)
            if species.sites is None:
                continue
            for site in species.sites:
                if site in vacancies:
                    vacancies[site] += 1
                else:
                    vacancies[site] = 1

        # If the user supplied "bare" sites, count them too
        for species in self.reactants:
            if species in vacancies:
                vacancies[species] -= 1
        for species in self.products:
            if species in vacancies:
                vacancies[species] += 1

        for vacancy, nvac in vacancies.items():
            # There are extra sites on the RHS, so add some to the LHS
            if nvac > 0:
                self.reactants += nvac * vacancy
            # There are extra sites on the LHS, so add some to the RHS
            elif nvac < 0:
                self.products += abs(nvac) * vacancy

        self.ts = None
        # The user supplied a transition state species
        if ts is not None:
            assert dG_act is None, "Cannot specify both barrier height and transition state!"
            # Wrap the TS in the _Reactants class
            if isinstance(ts, _Thermo):
                self.ts = _Reactants([ts])
            elif isinstance(ts, _Reactants):
                self.ts = ts
            # Fail if the user supplies something other than a _Thermo or _Reactants
            else:
                raise NotImplementedError
        # FIXME: Add stoichiometry checking to ensure logical reactions.
        # Caveat: Don't fail on unbalanced adsorption sites, since some
        # species take up more than one site.
#        # Mass balance requires that each element in the reactant is preserved
#        # in the product and in any transition states.
#        for element in self.reactants.elements:
#            assert self.reactants.elements[element] == self.products.elements[element]
#            if self.ts is not None:
#                assert self.reactants.elements[element] == self.ts.elements[element]
        # FIXME: This code can probably be reworked to not require a
        # user-supplied "adsorption" argument.
        self.adsorption = adsorption

        self.method = method
        if self.method is None:
            if self.ts is not None:
                self.method = 'TST'
            else:
                self.method = 'EQUIL'

        if isinstance(self.method, str):
            self.method = self.method.upper()

        # FIXME: this might fail on valid reactions, I haven't thought about it fully yet
        if self.adsorption and self.method not in ['EQUIL', 'CT', 'ER']:
            raise ValueError("Method {} unsupported for adsorption reactions!".format(self.method))

        self.S0 = S0
        self.keq = None
        self.kfor = None
        self.krev = None
        self.T = None
        self.Asite = None
        self.scale_params=['dH', 'dS', 'dH_act', 'dS_act', 'kfor', 'krev']

        # Scaling for sensitivity analysis, defaults to 1 (no scaling)
        self.scale = {}
        for param in self.scale_params:
            self.scale[param] = 1.0
        self.scale_old = self.scale.copy()

        # If the user supplied a TS, this should be None.
        self.dG_act = dG_act

        # If all reactants are Liquid species, then this reaction can occur
        # at any point of the diffusion grid, not just near the catalyst
        # surface.
        self.all_liquid = True

        # Count up the number of Fluid and Adsorbate species on either side
        # of the reaction. This is necessary to construct a proper Jacobian
        # for the rate of change of Fluid species vs Adsorbate species.
        # The Jacobian is related to the concentration in M of catalytic sites
        # in the model (defaults to 1 M in the Model class).
        self.Nreact_fluid = 0
        self.Nreact_ads = 0
        for species in self.reactants:
            if not isinstance(species, Liquid):
                self.all_liquid = False
            if isinstance(species, _Fluid):
                self.Nreact_fluid += 1
            elif isinstance(species, Adsorbate):
                self.Nreact_ads += 1

        self.Nprod_fluid = 0
        self.Nprod_ads = 0
        for species in self.products:
            if not isinstance(species, Liquid):
                self.all_liquid = False
            if isinstance(species, _Fluid):
                self.Nprod_fluid += 1
            elif isinstance(species, Adsorbate):
                self.Nprod_ads += 1

        self.Nfluid = self.Nreact_fluid + self.Nprod_fluid
        self.Nads = self.Nreact_ads + self.Nprod_ads

        self.dground = dground

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
            if self.dground:
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
        if self.dG_act is not None:
            barr *= sym.exp(-self.dG_act / (kB * T)) \
                    / self.reactants.get_reference_state()
#                    * self.ts.get_reference_state() \
        if self.method == 'EQUIL':
            self.kfor = _k * T  * barr / _hplanck * self.scale['kfor']
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
            m = 0.
            for species in self.reactants:
                if isinstance(species, _Fluid):
                    m += species.atoms.get_masses().sum()
            self.kfor = barr * 1000 * self.S0 * _Nav * Asite \
                    * np.sqrt(_k * T * kg \
                    / (2 * np.pi * m)) * self.scale['kfor']
        elif self.method == 'ER':
            # Collision Theory
            # kfor = S0 * Asite / (sqrt(2 * pi * m * kB * T))
            m_react = 0.
            for species in self.reactants:
                if isinstance(species, _Fluid):
                    m_react += species.atoms.get_masses().sum()
            kfor1 = barr * 1000 * self.S0 * _Nav * Asite \
                    * np.sqrt(_k * T * kg \
                    / (2 * np.pi * m_react)) * self.scale['kfor']

            m_prod = 0.
            for species in self.products:
                if isinstance(species, _Fluid):
                    m_prod = species.atoms.get_masses().sum()
            #FIXME: barr should be different for reverse reaction
            krev2 = barr * 1000 * self.S0 * _Nav * Asite \
                    * np.sqrt(_k * T * kg \
                    / (2 * np.pi * m_prod)) * self.scale['krev']
            kfor2 = self.keq * krev2
            self.kfor = kfor1 * kfor2 / (kfor1 + kfor2)
        elif self.method == 'TST':
            #Transition State Theory
            self.kfor = (_k * T / _hplanck) * barr * self.scale['kfor']
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
    def __init__(self, reactions, T, Asite, rhoref=1, coverage=1.0, z=0., nz=0, \
            shape='FLAT', steady_state=[], fixed=[], D=None, solvent=None, U0=None, V=None):
        # Set up list of reactions and species
        self.reactions = []
        self.species = []
        self.vacancy = []
        self.vacspecies = {}
        for reaction in reactions:
            assert isinstance(reaction, Reaction)
            self.reactions.append(reaction)
            for species in reaction.species:
                self.add_species(species)

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
                if species not in self.steady_state:
                    newspecies.append(species)
        for species in self.species:
            if isinstance(species, Adsorbate) or isinstance(species, DummyAdsorbate):
                if species not in self.steady_state and species not in self.vacancy:
                    newspecies.append(species)
        for species in self.steady_state:
            newspecies.append(species)
        for species in self.vacancy:
            newspecies.append(species)
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
        if species in self.species:
            return
        self.species.append(species)
        if species.sites is not None:
            for site in species.sites:
                if site not in self.vacancy:
                    self.add_species(site)
                    self.vacancy.append(site)
                    self.vacspecies[site] = [species]
                else:
                    self.vacspecies[site].append(species)

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

            reaction.update(T=self.T, Asite=self.Asite, force=True)

            rate_for = reaction.get_kfor(self.T, self.Asite)
            rate_rev = reaction.get_krev(self.T, self.Asite)

            if reaction.method in ['CT', 'ER']:
                rate_for *= self.rhoref
                rate_rev *= self.rhoref
            elif reaction.Nfluid != 0:
                rate_for *= self.rhoref**(reaction.Nreact_fluid - 1)
                rate_rev *= self.rhoref**(reaction.Nprod_fluid - 1)

            rate_count = {}
            for species in self.species:
                rate_count[species] = 0
                if isinstance(species, Liquid):
                    for i in range(self.nz):
                        rate_count[(species, i)] = 0

            for species in reaction.reactants:
                if not isinstance(species, Electron):
                    rate_for *= self.symbols_dict[species]
                rate_count[species] -= 1

            for species in reaction.products:
                if not isinstance(species, Electron):
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
            rate = rate_for - rate_rev
            if reaction.adsorption and self.diffusion:
                rate *= self.dV[0] / self.V

            self.rates.append(rate)
            self.rate_count.append(rate_count)
            self.is_rate_ads.append(reaction.adsorption)

            if reaction.all_liquid:
                for i in range(self.nz):
                    new_rate_count = {}
                    for species in self.species:
                        new_rate_count[species] = 0
                        if isinstance(species, Liquid):
                            for j in range(self.nz):
                                new_rate_count[(species, j)] = 0
                    subs = {}
                    for species in self.species:
                        if isinstance(species, Liquid) and species is not self.solvent:
                            new_rate_count[(species, i)] = rate_count[species]
                            subs[self.symbols_dict[species]] = self.symbols_dict[(species, i)]

                    self.rates.append(rate.subs(subs))
                    self.rate_count.append(new_rate_count)
                    self.is_rate_ads.append(reaction.adsorption) # this should always be "False"

    def rate_calc(self, U):
        self.update(U)
        return self.rates_last

    def set_initial_conditions(self, U0):
        if self.initialized:
            self.finalize()

        # Start with the incomplete user-provided initial conditions
        self.U0 = U0

        # Initialize counter for vacancies
        occsites = {species: 0 for species in self.vacancy}

        for species in self.U0:
            # Tuple implies we're talking about a diffusive liquid species
            if type(species) is tuple:
                species = species[0]

            # Throw an error if the user provides the concentration for a species we don't know about
            if species not in self.species:
                raise ValueError("Unknown species {}!".format(species))

            # If the species occupies a site, add its concentration to the occupied sites counter
            if species.sites is not None:
                for site in species.sites:
                    occsites[site] += self.U0[species]

            # If we're specifying a liquid concentration, U0[species] must equal U0[(species, nz-1)]
            elif isinstance(species, Liquid):
                if (species, self.nz - 1) in self.U0:
                    assert abs(self.U0[species] - self.U0[(species, self.nz - 1)]) < 1e-6, \
                            "Liquid concentrations not consistent!"

        self.vactot = {}
        # Determine what the initial vacancy concentration should be
        for species in self.vacancy:
            # The site with the highest concentration is normalized to one, so no site should
            # have an occupancy of over 1
            assert occsites[species] <= 1., "Too many adsorbates on {}!".format(species)
            # If we didn't also specify an initial vacancy concentration, assume it is 1
            if species not in U0:
                self.vactot[species] = 1.
                U0[species] = 1. - occsites[species]
            else:
                self.vactot[species] = U0[species] + occsites[species]
#            else:
#                assert abs(1. - U0[self.vacancy] - occsites) < 1e-6, \
#                        "Vacancy concentration not consistent!"
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
                if species in self.steady_state or species in self.vacancy:
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
#        algvar = np.zeros(self.nsymbols, dtype=bool)
        algvar = np.zeros(self.nsymbols, dtype=float)
        for i, symboli in enumerate(self.symbols_all):
            for j, symbolj in enumerate(self.symbols):
                if symboli == symbolj:
                    self.M[j, j] = M[i]
                    algvar[j] = M[i]

        # Volume of grid thickness per ads. site
        if self.diffusion:
            self.dV = self.zi * self.rhoref * self.V * self.Asite * 1000 * _Nav / self.coverage
#            self.dV = mol * self.zi * self.nsites * 1000 * self.Asite / self.coverage
            self.dV[-1] = self.V - self.dV.sum()
            assert self.dV[-1] > 0, "Volume is too small/quiescent layer is too thick! dV[-1] = {}".format(self.dV[-1])
            self.zi[-1] = self.dV[-1] * self.coverage / (self.rhoref * self.V * self.Asite * 1000)
#            self.zi[-1] = self.coverage * self.dV[-1] \
#                    / (1000 * mol * self.nsites * self.Asite)
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
                    for j, rate in enumerate(self.rates):
                        f += self.rate_count[j][(species, i)] * rate
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
                                f += self.rate_count[j][species] * rate * self.V / self.dV[0]
                    if not (i == self.nz - 1 and species in self.fixed):
                        self.f_sym.append(f)
                else:
                    continue
            elif species not in self.vacancy and species not in self.fixed:
                for i, rate in enumerate(self.rates):
                    f += self.rate_count[i][species] * rate
            elif species in self.vacancy:
                f = self.vactot[species] - self.symbols_dict[species]
                for a in self.vacspecies[species]:
                    f -= self.symbols_dict[a]
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

        self.finitialize(U0, 1e-8, [1e-8]*self.nsymbols, [], [], algvar)
        self.initialized = True

    def setup_execs(self):
        expressions = []
        expressions.extend(self.f_sym)
        expressions.extend(self.jac_sym.reshape(self.nsymbols**2))
        expressions.extend(self.rates)

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

        f2py.compile(program, modulename=modname, extra_args='--quiet --f90flags="-Wno-unused-dummy-argument -Wno-unused-variable" -lsundials_fida -lsundials_ida -lsundials_fnvecserial -lsundials_nvecserial -lmkl_rt ' +
                     os.path.join(dname, pyfname), source_fn=os.path.join(dname, fname), verbose=0)

        shutil.rmtree(dname)

        solve_ida = __import__(modname)

        self.finitialize = solve_ida.initialize
        self.fsolve = solve_ida.solve
        self.ffinalize = solve_ida.finalize

        os.remove(modname + '.so')

    def solve(self, t, ncp):
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
                if isinstance(species, Liquid) and species is not self.solvent:
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

            j = 0
            for reaction in self.reactions:
                ri[reaction] = r1[j][i]
                j += 1
                if reaction.all_liquid:
                    for i in range(self.nz):
                        ri[(reaction, i)] = r1[j][i]
                        j += 1
#            for j, reaction in enumerate(self.reactions):
#                ri[reaction] = r1[j][i]
            self.U.append(Ui)
            self.dU.append(dUi)
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
        return Model(self.reactions, self.T, self.Asite, self.rhoref, \
                self.coverage, self.z, self.nz, self.shape, \
                self.steady_state, self.fixed, self.D, self.solvent, U0, self.V)
