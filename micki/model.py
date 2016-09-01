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
            adsorption=False, dG_act=None, dground=False):

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
        if self.adsorption and self.method not in ['EQUIL', 'CT', 'ER', 'DIFF']:
            raise ValueError("Method {} unsupported for adsorption reactions!".format(self.method))

        self.S0 = S0
        self.keq = None
        self.kfor = None
        self.krev = None
        self.T = None
        self.Asite = None
        self.L = None
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

    def update(self, T=None, Asite=None, L=None, force=False):
        if not force and not self.is_update_needed(T, Asite, L):
            return

        self.T = T
        self.Asite = Asite
        self.L = L
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
        self._calc_kfor(T, Asite, L)
        self._calc_krev(T, Asite, L)
        self.scale_old = self.scale.copy()

    def is_update_needed(self, T, Asite, L):
        for species in self.species:
            if species.is_update_needed(T):
                return True
        if self.keq is None:
            return True
        if T is not None and T != self.T:
            return True
        if Asite is not None and Asite != self.Asite:
            return True
        if L is not None and L != self.L:
            return True
        for param in self.scale_params:
            if self.scale[param] != self.scale_old[param]:
                return True
        return False

    def get_keq(self, T, Asite, L=None):
        self.update(T, Asite, L)
        return self.keq

    def get_kfor(self, T, Asite, L=None):
        self.update(T, Asite, L)
        return self.kfor

    def get_krev(self, T, Asite, L=None):
        self.update(T, Asite, L)
        return self.krev

    def _calc_keq(self, T):
        self.keq = sym.exp(-self.dG / (kB * T)) \
                * self.products.get_reference_state() \
                / self.reactants.get_reference_state() \
                * self.scale['kfor'] / self.scale['krev']

    def _calc_kfor(self, T, Asite, L=None):
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
#            m = 0.
#            for species in self.reactants:
#                if isinstance(species, _Fluid):
#                    m += species.atoms.get_masses().sum()
#            self.kfor = barr * 1000 * self.S0 * _Nav * Asite \
#                    * np.sqrt(_k * T * kg \
#                    / (2 * np.pi * m)) * self.scale['kfor']
            # New implementation
            found_fluid = False
            for species in self.reactants:
                if isinstance(species, _Fluid):
                    if found_fluid:
                        raise ValueError, "At most one fluid can react with CT!"
                    found_fluid = True
                    fluid = species
            Sfluid = fluid.get_S(T)
            fluid._calc_qtrans2D(T, Asite)
            Strans = Sfluid - fluid.S['elec'] - fluid.S['rot'] - fluid.S['vib']
            Slost = Strans / fluid.S['trans']
            dS = (fluid.S['trans2D'] - fluid.S['trans']) * Slost
            dG = fluid.E['trans2D'] - fluid.E['trans'] - T * dS
            self.kfor = barr * _k * T / _hplanck * np.exp(-dG / (kB * T))
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
        elif self.method == 'DIFF':
            if L is None:
                raise ValueError, "Must provide diffusion length for diffusion reactions!"
            found_fluid = False
            for species in self.reactants:
                if isinstance(species, _Fluid):
                    if found_fluid:
                        raise ValueError, "Diffusion reaction must have exactly 1 fluid!"
                    found_fluid = True
                    D = species.D
            sites = 1
            for species in self.products:
                if isinstance(species, Adsorbate):
                    for site in species.sites:
                        sites *= site.symbol
                elif not isinstance(species, Electron):
                    raise ValueError, "All products must be adsorbates in diffusion reaction!"
            self.kfor = 1000 * D * self.Asite * mol * barr * self.scale['kfor'] / (L * sites)
        elif self.method == 'TST':
            #Transition State Theory
            self.kfor = (_k * T / _hplanck) * barr * self.scale['kfor']
        else:
            raise ValueError("Method {} is not recognized!".format(self.method))

    def _calc_krev(self, T, Asite, L=None):
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
    def __init__(self, reactions, T, Asite, z=None, nz=1, \
            shape='FLAT', steady_state=[], fixed=[], solvent=None, U0=None):
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

        # Load in steady-state approximated species
        self.steady_state = steady_state
        # Solvent will not diffuse even in diffusion system
        self.solvent = solvent
        # Fixed species are removed from the differential equations
        self.fixed = fixed
        # Solvent must be fixed
        if self.solvent is not None and self.solvent not in self.fixed:
            self.fixed.append(self.solvent)
        # Raise error if fixed species is not found in reactions (is this necessary?)
        for species in self.fixed:
            assert species in self.species, "Unknown fixed species {}".format(species)

        # Other model parameters
        self.T = T # System temperature
        self.Asite = Asite # Area of adsorption site -- 1 per microkinetic model!
        self.z = z # Diffusion length
        self.nz = nz # Number of diffusion grid points
        self.shape = shape # Distribution of diffusion grid points

#        if self.z is None:
#            self.V = 1
#        else:
#            #self.V = _Nav * self.Asite * self.z * 2000
#            self.V = 100

        # Do we need to consider diffusion?
        self.diffusion = False
        if nz > 1: # No numerical diffusion if there is only one grid point
            # No diffusion if there are no Liquid species in the system
            for species in self.species:
                if isinstance(species, Liquid):
                    assert self.z > 0, "Must specify boundary layer thickness for diffusion!"
                    self.diffusion = True
                    break

        # This is better than what I was doing above, but I need to debug this for now.
        if self.diffusion:
            self.V = _Nav * self.Asite * self.z * 2000
        else:
            self.V = 1

        # Reorder species such that Liquid -> Gas -> Adsorbate -> Vacancy
        # Steady-state species go to the end.
        newspecies = []
        self.nliquid = 0
        for species in self.species:
            if isinstance(species, Liquid):
                # As long as we are reordering the system, check to ensure all
                # Liquid species have a diffusion constant if we are doing
                # diffusion
                if self.diffusion:
                    assert species.D is not None, \
                            "Specify diffusion constant for {}!".format(species)
                    assert species not in self.steady_state, \
                            "Can't have Liquid in steady state with diffusion!"
                if species not in self.steady_state:
                    newspecies.append(species)
                self.nliquid += 1
        for species in self.species:
            if isinstance(species, (Gas, DummyFluid, Electron)):
                if species not in self.steady_state:
                    newspecies.append(species)
        for species in self.species:
            if isinstance(species, (Adsorbate, DummyAdsorbate)):
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
        # If the user supplied initial conditions, create and compile differential equations
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
        for i in range(self.nz - 1):
            self.zi[i] += self.dz[i]/2.
            if i > 0:
                self.zi[i] += self.dz[i-1]/2.
        self.dV = _Nav * self.Asite * self.zi * 1000
        self.dV[-1] = self.V - self.dV.sum()
        assert self.dV[-1] > 0, "Volume is too small/quiescent layer is too thick! dV[-1] = {}".format(self.dV[-1]) # THIS SHOULD NEVER HAPPEN ANYMORE
        self.zi[-1] = self.dV[-1] / (_Nav * self.Asite * 1000)
        self.dz = list(self.dz)
        self.dz.append(2 * self.zi[-1] - self.dz[-2])
        self.dz = np.array(self.dz)

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

            # If we're specifying a Liquid concentration with diffusion enabled, U0[species]
            # must equal U0[(species, nz-1)]
            if self.diffusion and isinstance(species, Liquid):
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

        # Populate dictionary of initial conditions for all species
        for species in self.species:
            # Assume concentration of unnamed species is 0
            if species not in self.U0:
                self.U0[species] = 0.
            # If we are doing diffusion, liquid species should have initial
            # concentrations for ALL diffusion grid points. We will assume a
            # flat concentration profile initially at the user specified
            # bulk concentration (or 0 if the user didn't specify anything)
            if self.diffusion and isinstance(species, Liquid):
                U0i = self.U0[species]
                if species is not self.solvent:
                    for i in range(self.nz):
                        # The user can also specify concentration at individual
                        # grid points
                        if (species, i) not in self.U0:
                            self.U0[(species, i)] = U0i
                        else:
                            U0i = self.U0[(species, i)]

        # The number of variables that will be in our differential equations
        size = len(self.species)
        # Each diffusion grid point adds nliquid variables
        if self.diffusion:
            size += (self.nz - 1) * self.nliquid
            # If the solvent is a liquid, don't count it
            if self.solvent is not None and isinstance(self.solvent, Liquid):
                size -= self.nz - 1
        # Initialize "mass matrix", which is a diagonal matrix with 1s for
        # differential elements and 0s for algebraic elements (steady-state)
        M = np.ones(size, dtype=int)
        i = 0
        for species in self.species:
            if self.diffusion and isinstance(species, Liquid) and species is not self.solvent:
                i += self.nz
            else:
                if species in self.steady_state or species in self.vacancy:
                    M[i] = 0
                i += 1

        # This creates a symbol for each species named modelparamX where X
        # is a three-digit numerical identifier that corresponds to its position
        # in the order of species
        self.symbols_all = sym.symbols(' '.join(['modelparam{}'.format(str(i).zfill(3)) for i in range(size)]))
        # symbols_dict a Thermo object and returns its corresponding symbol
        self.symbols_dict = {}
        # symbols ONLY includes species that will be in the differential equations.
        # Fixed species are not included in this list
        self.symbols = []
        i = 0
        for species in self.species:
            if self.diffusion and isinstance(species, Liquid):
                self.symbols_dict[species] = self.symbols_all[i]
                if species is not self.solvent:
                    for j in range(self.nz):
                        # (species, 0) and species should have the same symbol,
                        # as the correspond to the same variable
                        self.symbols_dict[(species, j)] = self.symbols_all[i]
                        # If we are doing diffusion and a Liquid species is
                        # fixed, we only exclude the *last* gridpoint from
                        # the differential equations.
                        if not (j == self.nz - 1 and species in self.fixed):
                            self.symbols.append(self.symbols_all[i])
                        i += 1
                else:
                    self.symbols_dict[(species, 0)] = self.symbols_all[i] # XXX what is this?
                    self.symbols_dict[(species, 9)] = self.symbols_all[i] # XXX same question
                    i += 1
            else:
                self.symbols_dict[species] = self.symbols_all[i]
                if species not in self.fixed:
                    self.symbols.append(self.symbols_all[i])
                i += 1

        # nsymbols is the size of the differential equations
        self.nsymbols = len(self.symbols)
        # trans_cov_symbols converts the species assigned symbol (species.symbol)
        # to the model's internal symbol for that species
        self.trans_cov_symbols = {}
        for species in self.species:
            if species.symbol is not None:
                self.trans_cov_symbols[species.symbol] = self.symbols_dict[species]

        # Create the final mass matrix of the proper dimensions
        self.M = np.zeros((self.nsymbols, self.nsymbols), dtype=int)
        # algvar tells the solver which variables are differential
        # and which are algebraic. It is the diagonal of the mass matrix.
        algvar = np.zeros(self.nsymbols, dtype=float)
        for i, symboli in enumerate(self.symbols_all):
            for j, symbolj in enumerate(self.symbols):
                if symboli == symbolj:
                    self.M[j, j] = M[i]
                    algvar[j] = M[i]

        # Initialize all rate expressions based on the above symbols
        self._rate_init()
        # f_sym is the SYMBOLIC master equation for all species
        self.f_sym = []

        
        for species in self.species:
            f = 0
            # DETAILED BALANCE NOTE: See discussion in _rate_init
            if self.diffusion and isinstance(species, Liquid) and species is not self.solvent:
                for i in range(self.nz):
                    f = 0
                    # At each grid point, add all-liquid reaction rates
                    for j, rate in enumerate(self.rates):
                        f += self.rate_count[j][(species, i)] * rate
                    diff = species.D / self.zi[i]
                    # For all grid points except the first, add diffusion away
                    # from the surface
                    if i > 0:
                        f += diff * (self.symbols_dict[(species, i-1)] \
                                - self.symbols_dict[(species, i)]) / self.dz[i-1]
                    # For all grid points except the last, add diffusion towards
                    # the surface
                    if i < self.nz - 1:
                        f += diff * (self.symbols_dict[(species, i+1)] \
                                - self.symbols_dict[(species, i)]) / self.dz[i]
                    # At the surface, account for any adsorption reactions
                    if i == 0:
                        for j, rate in enumerate(self.rates):
                            if self.is_rate_ads[j]:
                                f += self.rate_count[j][species] * rate * self.V / self.dV[0]
                    # If we were looking at the last grid point and the species is
                    # fixed, discard what we just calculated
                    if not (i == self.nz - 1 and species in self.fixed):
                        self.f_sym.append(f)
                else:
                    # This is necessary to avoid double-counting non-fixed
                    # liquid species. continue skips the below "if" statement
                    continue
            elif species not in self.vacancy and species not in self.fixed:
                # Reactions involving on-surface species
                for i, rate in enumerate(self.rates):
                    f += self.rate_count[i][species] * rate
            elif species in self.vacancy:
                # Vacancies are algebraic, we ensure here that the number
                # of a given type of adsorption site does not change over time
                f = self.vactot[species] - self.symbols_dict[species]
                for a in self.vacspecies[species]:
                    f -= self.symbols_dict[a]
            # Discard rates for fixed species
            if species not in self.fixed:
                self.f_sym.append(f)
            else:
                assert f == 0, "Fixed species rate of change not zero!"

        # Fixed species must have their symbols replaced by their fixed
        # initial values. subs is a dictionary whose keys are internal species
        # symbols and whose values are the initial concentration of that species.
        subs = {}
        for species in self.species:
            if species in self.fixed:
                liquid = False
                if self.diffusion and isinstance(species, Liquid) and species is not self.solvent:
                    species = (species, self.nz - 1)
                    liquid = True
                U0i = self.U0[species]
                if liquid or isinstance(species, (_Fluid, DummyFluid)):
                    U0i *= self.V
                subs[self.symbols_dict[species]] = U0i 

        # sym.sympify ensures that subs will not fail (if the expression has no
        # sympy symbols in it, this would normally fail)
        for i, f in enumerate(self.f_sym):
            self.f_sym[i] = sym.sympify(f).subs(subs)
        
        # jac_sym is the SYMBOLIC Jacobian matrix, that is df/dc, where f is a
        # row of the master equation and c is a species.
        self.jac_sym = np.zeros((self.nsymbols, self.nsymbols), dtype=object)
        for i, f in enumerate(self.f_sym):
            for j, symbol in enumerate(self.symbols):
                self.jac_sym[i, j] = sym.diff(f, symbol)

        # Additionally, insert fixed species concentrations into rate expressions
        for i, r in enumerate(self.rates):
            self.rates[i] = sym.sympify(r).subs(subs)
        
        # Sets up and compiles the Fortran differential equation solving module
        self.setup_execs()

        # Convert the dictionary U0 of initial conditions into a list that can
        # be used with the Fortran module.
        U0 = []
        for symbol in self.symbols:
            for species, isymbol in self.symbols_dict.items():
                if symbol == isymbol:
                    U0i = self.U0[species]
                    if type(species) is tuple or isinstance(species, (_Fluid, DummyFluid)):
                        U0i *= self.V
                    U0.append(U0i)
                    break

        # Pass initial values to the fortran module
        self.finitialize(U0, 1e-10, [1e-10]*self.nsymbols, [], [], algvar)

        self.initialized = True

    def _rate_init(self):
        # List of symbolic rate expressions
        self.rates = []
        # List of dicts. For a given reaction, the change in concentration
        # of a species for "one unit" of reaction occuring in the forward
        # direction
        self.rate_count = []
        # Keeps track of adsorption reactions
        self.is_rate_ads = []

        for reaction in self.reactions:
            # Calculate kfor, krev, and keq
            # If we are diffusing and the species is a liquid, the diffusion
            # length is the distance to the first grid point, NOT the overall
            # stationary layer thickness.
            liquid_diffusion = False
            if self.diffusion:
                for species in reaction.reactants:
                    if isinstance(species, Liquid):
                        liquid_diffusion = True
                        break
            if liquid_diffusion:
                rate_for = reaction.get_kfor(self.T, self.Asite, self.zi[0])
                rate_rev = reaction.get_krev(self.T, self.Asite, self.zi[0])
            else:
                rate_for = reaction.get_kfor(self.T, self.Asite, self.z)
                rate_rev = reaction.get_krev(self.T, self.Asite, self.z)
            
            # For adsorption reactions, scale the rate by the concentration
            # of catalytic sites (normally 1 for reactions without diffusion)
            if reaction.method in ['CT', 'ER', 'DIFF']:
                rate_for /= self.V
            # For reactions entirely in solution, scale rate by volume
            elif reaction.Nfluid != 0:
                rate_for *= self.V**(-reaction.Nreact_fluid + 1)
                rate_rev *= self.V**(-reaction.Nprod_fluid + 1)

            # IMPORTANT NOTE PERTAINING TO ABOVE:
            # As far as the user can tell, all fluids (Gas, Liquid) are represented
            # in units of concentration (i.e. mol/L) and all adsorbates are
            # represented in coverage (i.e. N/M, where M is the total number of sites).
            # INTERNALLY, this is not the case. Internally, ALL species are number
            # fractions relative to the default number of catalytic sites
            # (though this can be changed). This means that in order to achieve
            # detailed balance, the reaction rates must be modulated by the volume.
            # Further sections of the code that pertain to this behavior will
            # be highlighted with "DETAILED BALANCE NOTE"

            # Initialize dictionary for reaction stiochiometry to 0 for all
            # species in the model
            rate_count = {}
            for species in self.species:
                rate_count[species] = 0
                if self.diffusion and isinstance(species, Liquid):
                    for i in range(self.nz):
                        rate_count[(species, i)] = 0
            
            # Reactants are consumed in the forward direction
            for species in reaction.reactants:
                # Multiply the rate by the species' symbol UNLESS it is an
                # electron. All reactions are 0th order in electrons, even
                # if they consume/produce electrons.
                if not isinstance(species, Electron):
                    rate_for *= self.symbols_dict[species]
                rate_count[species] -= 1
            
            # Products are created in the forward direction
            for species in reaction.products:
                if not isinstance(species, Electron):
                    rate_rev *= self.symbols_dict[species]
                rate_count[species] += 1

            # If there are symbols in a species energy or rate, convert
            # those symbols to the Model's internal symbols
            if self.trans_cov_symbols is not None:
                # subs fails if there are no Sympy objects in an expression,
                # so we just check to make sure subs won't fail before using it
                if isinstance(rate_for, sym.Basic):
                    rate_for = rate_for.subs(self.trans_cov_symbols)
                if isinstance(rate_rev, sym.Basic):
                    rate_rev = rate_rev.subs(self.trans_cov_symbols)

            # Overall reaction rate (flux)
            rate = rate_for - rate_rev

            # For adsorption reactions, scale rates by the volume of the
            # first layer
            if reaction.adsorption and self.diffusion:
                rate *= self.dV[0] / self.V
            
            # Append all data to global rate lists
            self.rates.append(rate)
            self.rate_count.append(rate_count)
            self.is_rate_ads.append(reaction.adsorption)

            # If a reaction is occuring between all liquid species, then we
            # have to perform the same reaction at all grid points. This is
            # an exact mirror of what occurred above, but at different grid
            # points.
            if self.diffusion and reaction.all_liquid:
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


    def setup_execs(self):
        from micki.fortran import f90_template, pyf_template
        from numpy import f2py

        # y_vec is an array symbol that will represent the species concentrations
        # provided by the differential equation solver inside the Fortran code
        # (that is, y_vec is an INPUT to the functions that calculate the
        # residual, Jacobian, and rate)
        y_vec = sym.IndexedBase('yin', shape=(self.nsymbols,))
        # Map y_vec elements (1-indexed, of course) onto 'modelparam' symbols
        trans = {self.symbols[i]: y_vec[i + 1] for i in range(self.nsymbols)}
        # Map string represntation of 'modelparam' symbols onto string
        # representation of y-vec elements
        str_trans = {sym.fcode(self.symbols[i], source_format='free'): sym.fcode(y_vec[i + 1], source_format='free') for i in range(self.nsymbols)}

        # these will contain lists of strings, with each element being one
        # Fortran assignment for the master equation, Jacobian, and
        # rate expressions
        rescode = []
        jaccode = []
        ratecode = []

        # Convert symbolic master equation into a valid Fortran string
        for i in range(self.nsymbols):
            fcode = sym.fcode(self.f_sym[i], source_format='free')
            # Replace modelparam symbols with their y_vec counterpart
            for key, val in str_trans.items():
                fcode = fcode.replace(key, val)
            # Create actual line of code for calculating residual
            rescode.append('   res({}) = '.format(i + 1) + fcode)

        # Effectively the same as above, except on the two-dimensional Jacobian
        # matrix.
        for i in range(self.nsymbols):
            for j in range(self.nsymbols):
                expr = self.jac_sym[j, i]
                # Unlike the residual, some elements of the Jacobian can be 0.
                # We don't need to bother writing 'jac(x,y) = 0' a hundred times
                # in Fortran, so we omit those.
                if expr != 0:
                    fcode = sym.fcode(expr, source_format='free')
                    for key, val in str_trans.items():
                        fcode = fcode.replace(key, val)
                    jaccode.append('   jac({}, {}) = '.format(j + 1, i + 1) + fcode)

        # See residual above
        for i, rate in enumerate(self.rates):
            fcode = sym.fcode(rate, source_format='free')
            for key, val in str_trans.items():
                fcode = fcode.replace(key, val)
            ratecode.append('   rates({}) = '.format(i + 1) + fcode)

        # We insert all of the parameters of this differential equation into
        # the prewritten Fortran template, including the residual, Jacobian,
        # and rate expressions we just calculated.
        program = f90_template.format(neq=self.nsymbols, nx=1,
                nrates=len(self.rates), rescalc='\n'.join(rescode),
                jaccalc='\n'.join(jaccode), ratecalc='\n'.join(ratecode))

        # Generate a randomly-named temp directory for compiling the module.
        # We will name the actual module file after the directory.
        dname = tempfile.mkdtemp()
        modname = os.path.split(dname)[1]
        fname = modname + '.f90'
        pyfname = modname + '.pyf'

        # For debugging purposes, write out the generated module
        with open('solve_ida.f90', 'w') as f:
            f.write(program)
        
        # Write the pertinent data into the temp directory
        with open(os.path.join(dname, pyfname), 'w') as f:
            f.write(pyf_template.format(modname=modname, neq=self.nsymbols, nrates=len(self.rates)))
    
        # Compile the module with f2py
        f2py.compile(program, modulename=modname, extra_args='--quiet --f90flags="-Wno-unused-dummy-argument -Wno-unused-variable -w" -lsundials_fida -lsundials_ida -lsundials_fnvecserial -lsundials_nvecserial -lmkl_rt ' +
                     os.path.join(dname, pyfname), source_fn=os.path.join(dname, fname), verbose=0)
        
        # Delete the temporary directory
        shutil.rmtree(dname)
        
        # Import the module on-the-fly with __import__. This is kind of a hack.
        solve_ida = __import__(modname)
        
        # The Fortran module's initialize, solve, and finalize routines
        # are mapped onto finitialize, fsolve, and ffinalize inside the Model
        # object. We don't want users touching these manually
        self.finitialize = solve_ida.initialize
        self.fsolve = solve_ida.solve
        self.ffinalize = solve_ida.finalize
        
        # Delete the module file. We've already imported it, so it's in memory.
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
                if self.diffusion and isinstance(species, Liquid) and species is not self.solvent:
                    species = (species, self.nz - 1)
                dUi[species] = 0.
                Ui[species] = self.U0[species]
            for j, symbol in enumerate(self.symbols):
                for species, isymbol in self.symbols_dict.items():
                    if symbol == isymbol:
                        Uij = self.U1[i][j]
                        dUij = self.dU1[i][j]
                        if type(species) is tuple or isinstance(species, (_Fluid, DummyFluid)):
                            Uij /= self.V
                            dUij /= self.V
                        Ui[species] = Uij
                        dUi[species] = dUij

            j = 0
            for reaction in self.reactions:
                ri[reaction] = r1[j][i]
                j += 1
                if self.diffusion and reaction.all_liquid:
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
        raise NotImplementedError
#        if initialize:
#            U0 = self.U0
#        else:
#            U0 = None
#        return Model(self.reactions, self.T, self.Asite, self.rhoref, \
#                self.coverage, self.z, self.nz, self.shape, \
#                self.steady_state, self.fixed, self.D, self.solvent, U0, self.V)
