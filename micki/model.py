"""Microkinetic modeling objects"""

from __future__ import print_function

import os
import tempfile
import shutil
import warnings

from collections import OrderedDict

import numpy as np
import sympy as sym

from copy import copy
from ase.units import kB, _hplanck, kg, _k, _Nav, mol

from micki.reactants import _Thermo, _Fluid, _Reactants, Gas, Liquid, Adsorbate
from micki.reactants import Electron

from micki.lattice import Lattice


class Reaction(object):
    def __init__(self, reactants, products, ts=None, method=None, S0=1.,
                 dG_act=None, dground=False, reversible=True):

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
        vacancies = OrderedDict() 
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
            assert dG_act is None, \
                    "Cannot specify both barrier height and transition state!"
            # Wrap the TS in the _Reactants class
            if isinstance(ts, _Thermo):
                self.ts = _Reactants([ts])
            elif isinstance(ts, _Reactants):
                self.ts = ts
            # Fail if the user supplies something other than a _Thermo
            # or _Reactants
            else:
                raise NotImplementedError
        # FIXME: Add stoichiometry checking to ensure logical reactions.
        # Caveat: Don't fail on unbalanced adsorption sites, since some
        # species take up more than one site.
        self.adsorption = False

        adscheck = 0
        for species in self.reactants:
            if isinstance(species, _Fluid):
                adscheck += 1
        for species in self.products:
            if isinstance(species, _Fluid):
                break
        else:
            if adscheck == 1:
                self.adsorption = True

        self.method = method
        if self.method is None:
            if self.ts is not None:
                self.method = 'TST'
            else:
                self.method = 'EQUIL'

        if isinstance(self.method, str):
            self.method = self.method.upper()

        self.S0 = S0
        self.keq = None
        self.kfor = None
        self.krev = None
        self.T = None
        self.Asite = None
        self.L = None
        self.scale_params = ['dH', 'dS', 'dH_act', 'dS_act', 'kfor', 'krev']
        self.alpha = None
        self.reversible = reversible

        # Scaling for sensitivity analysis, defaults to 1 (no scaling)
        self.scale = OrderedDict() 
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
#        self.dH *= self.scale['dH']
        self.dS = self.products.get_S(T) - self.reactants.get_S(T)
#        self.dS *= self.scale['dS']
        self.dG = self.dH - self.T * self.dS
        if self.ts is not None:
            self.dH_act = self.ts.get_H(T) - self.reactants.get_H(T)
#            self.dH_act *= self.scale['dH_act']
            self.dS_act = self.ts.get_S(T) - self.reactants.get_S(T)
#            self.dS_act *= self.scale['dS_act']
            self.dG_act = self.dH_act - self.T * self.dS_act

            self.ts_dE = np.sum([species.coverage for species in self.ts])
            self.reactants_dE = np.sum([species.coverage for species in self.reactants])
            self.products_dE = np.sum([species.coverage for species in self.products])

            self.ts_dE += np.sum([species.dE for species in self.ts])
            self.reactants_dE += np.sum([species.dE for species in self.reactants])
            self.products_dE += np.sum([species.dE for species in self.products])

            G_react = np.sum([species.get_G(T) for species in self.reactants])
            G_prod = np.sum([species.get_G(T) for species in self.products])
            G_ts = np.sum([species.get_G(T) for species in self.ts])

            all_symbols = set()
            all_symbols.update(sym.sympify(G_react).atoms(sym.Symbol))
            all_symbols.update(sym.sympify(G_prod).atoms(sym.Symbol))
            all_symbols.update(sym.sympify(G_ts).atoms(sym.Symbol))

            G_react = sym.sympify(G_react).subs({symbol: 0 for symbol in all_symbols})
            G_prod = sym.sympify(G_prod).subs({symbol: 0 for symbol in all_symbols})
            G_ts = sym.sympify(G_ts).subs({symbol: 0 for symbol in all_symbols})
            reactants_dE = sym.sympify(self.reactants_dE)
            reactants_dE = reactants_dE.subs({symbol: 0 for symbol in all_symbols})
            products_dE = sym.sympify(self.products_dE)
            products_dE = products_dE.subs({symbol: 0 for symbol in all_symbols})

            A = 2*reactants_dE - 2*products_dE
            B = 2*G_ts - G_react - G_prod - 2*reactants_dE + 2*products_dE
            C = -G_ts + G_prod

            if abs(A) < 1e-8:
                dG_for = G_ts - G_react
                dG_rev = G_ts - G_prod
                self.alpha = dG_rev / (dG_rev + dG_for)
            else:

                alpha1 = (-B + sym.sqrt(B**2 - 4*A*C))/(2*A)
                alpha2 = (-B - sym.sqrt(B**2 - 4*A*C))/(2*A)

                if 0 <= alpha1 <= 1:
                    self.alpha = alpha1
                elif 0 <= alpha2 <= 1:
                    self.alpha = alpha2
                elif alpha1 < 0 and alpha2 < 0:
                    self.alpha = 0.
                elif alpha1 > 1 and alpha2 > 1:
                    self.alpha = 1.
                else:
                    print(self, alpha1, alpha2)
                    raise ValueError("Failed to find alpha parameter!")

            self.dG_act -= self.ts_dE
            self.dG_act += self.alpha * self.reactants_dE
            self.dG_act += (1 - self.alpha) * self.products_dE

            # If there is a coverage dependence, assume everything has
            # coverage 0
            if self.dground:
                dG_act = self.dG_act
                if isinstance(dG_act, sym.Basic):
                    subs = {}
                    for atom in dG_act.atoms(sym.Symbol):
                        subs[atom] = 0.
                    dG_act = dG_act.subs(subs)

                if dG_act < 0.:
                    warnings.warn('Negative activation energy found for {}. '
                                  'Rounding to 0.'.format(self),
                                  RuntimeWarning, stacklevel=2)
                    self.dG_act = 0.

                dG_rev = self.dG_act - self.dG
                if isinstance(dG_rev, sym.Basic):
                    subs = {}
                    for atom in dG_rev.atoms(sym.Symbol):
                        subs[atom] = 0.
                    dG_rev = dG_rev.subs(subs)

                if dG_rev < 0.:
                    warnings.warn('Negative activation energy found for {}. '
                                  'Rounding to {}'.format(self, self.dG),
                                  RuntimeWarning, stacklevel=2)
                    self.dG_act = self.dG
        self._calc_keq()
        self._calc_kfor()
        self._calc_krev()
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

    def get_keq(self, T=None, Asite=None, L=None):
        self.update(T, Asite, L)
        return self.keq

    def get_kfor(self, T=None, Asite=None, L=None):
        self.update(T, Asite, L)
        return self.kfor

    def get_krev(self, T=None, Asite=None, L=None):
        self.update(T, Asite, L)
        return self.krev

    def _calc_keq(self):
        self.keq = sym.exp(-self.dG / (kB * self.T)) \
                * self.products.get_reference_state() \
                / self.reactants.get_reference_state() \
                * self.scale['kfor'] / self.scale['krev']

    def _calc_kfor(self):
        barr = 1
        if self.dG_act is not None:
            barr *= sym.exp(-self.dG_act / (kB * self.T)) \
                    / self.reactants.get_reference_state()
#                    * self.ts.get_reference_state() \
        if self.method == 'EQUIL':
            self.kfor = _k * self.T * barr / _hplanck * self.scale['kfor']
            if isinstance(self.keq, sym.Basic):
                subs = {}
                for atom in self.keq.atoms(sym.Symbol):
                    subs[atom] = 0.
                keq = self.keq.subs(subs)
            else:
                keq = self.keq
            if keq < 1:
                self.kfor *= self.keq * self.scale['krev'] / self.scale['kfor']
        elif self.method == 'DIEQUIL':
            kfor1 = _k * self.T * barr / _hplanck * self.scale['kfor']
            kfor2 = kfor1 * self.keq * self.scale['krev'] / self.scale['kfor']
            self.kfor = kfor1 * kfor2 / (kfor1 + kfor2)
        elif self.method == 'CT':
            # CT is TST with the transition state being a non-interacting
            # 2D ideal gas.
            found_fluid = False
            for species in self.reactants:
                if isinstance(species, _Fluid):
                    if found_fluid:
                        raise ValueError("At most one fluid "
                                         "can react with CT!")
                    found_fluid = True
                    fluid = species
            Sfluid = fluid.get_S(self.T)
            fluid._calc_qtrans2D(self.T, self.Asite)
            Strans = Sfluid - fluid.S['elec'] - fluid.S['rot'] - fluid.S['vib']
            Slost = Strans / fluid.S['trans']
            dS = (fluid.S['trans2D'] - fluid.S['trans']) * Slost
            dG = fluid.E['trans2D'] - fluid.E['trans'] - self.T * dS
            self.kfor = barr * _k * self.T / _hplanck * np.exp(-dG / (kB * self.T))
            self.kfor *= self.scale['kfor']
        elif self.method == 'ER':
            # Collision Theory
            # kfor = S0 * Asite / (sqrt(2 * pi * m * kB * T))
            m_react = 0.
            for species in self.reactants:
                if isinstance(species, _Fluid):
                    m_react += species.atoms.get_masses().sum()
            kfor1 = barr * 1000 * self.S0 * _Nav * self.Asite \
                * np.sqrt(_k * self.T * kg / (2 * np.pi * m_react)) \
                * self.scale['kfor']

            m_prod = 0.
            for species in self.products:
                if isinstance(species, _Fluid):
                    m_prod = species.atoms.get_masses().sum()
            # FIXME: barr should be different for reverse reaction
            krev2 = barr * 1000 * self.S0 * _Nav * self.Asite \
                * np.sqrt(_k * self.T * kg / (2 * np.pi * m_prod)) \
                * self.scale['krev']
            kfor2 = self.keq * krev2
            self.kfor = kfor1 * kfor2 / (kfor1 + kfor2)
        elif self.method == 'DIFF':
            if self.L is None:
                raise ValueError("Must provide diffusion length "
                                 "for diffusion reactions!")
            found_fluid = False
            for species in self.reactants:
                if isinstance(species, _Fluid):
                    if found_fluid:
                        raise ValueError("Diffusion reaction must "
                                         "have exactly 1 fluid!")
                    found_fluid = True
                    D = species.D
            sites = 1
            for species in self.reactants:
                if isinstance(species, Adsorbate):
                    sites *= species.symbol
            for species in self.products:
                if not isinstance(species, (Adsorbate, Electron)):
                    raise ValueError("All products must be adsorbates "
                                     "in diffusion reaction!")
            self.kfor = 1000 * D * self.Asite * mol * barr \
                * self.scale['kfor'] / (self.L * sites)
        elif self.method == 'TST':
            # Transition State Theory
            self.kfor = (_k * self.T / _hplanck) * barr * self.scale['kfor']
        else:
            raise ValueError("Method {} is not recognized!".format(
                self.method))

    def _calc_krev(self):
        self.krev = self.kfor / self.keq

    def __repr__(self):
        string = self.reactants.__repr__() + ' <-> '
        if self.ts is not None:
            string += self.ts.__repr__() + ' <-> '
        string += self.products.__repr__()
        return string


class Model(object):
    def __init__(self, T, Asite, z=0, nz=0, shape='FLAT', lattice=None):
        self.reactions = OrderedDict()
        self._reactions = []
        self._species = []
        self.species = OrderedDict()
        self.vacancy = []
        self.vacspecies = OrderedDict()
        self.solvent = None
        self.fixed = []
        self.initialized = False
        self.U0 = None

        self.T = T  # System temperature
        self.Asite = Asite  # Area of adsorption site
        self._z = z  # Diffusion length
        self.nz = nz  # Number of diffusion grid points
        self.shape = shape  # Distribution of diffusion grid points
        self.lattice = lattice

    def check_diffusion(self):
        # Do we need to consider diffusion?
        self.diffusion = False
        self.V = 1
        if self.nz > 0:  # No numerical diffusion if there is only one grid point
            if self.z <= 0:
                raise ValueError('Must specify boundary layer thickness for'
                                 'diffusion!')
            self.diffusion = True
            self.V = _Nav * self.Asite * self.z * 2000

    def add_reactions(self, reactions):
        # Set up list of reactions and species
        for name, reaction in reactions.items():
            assert isinstance(reaction, Reaction)
            if reaction in self._reactions:
                return
            self._reactions.append(reaction)
            self.reactions[name] = reaction
            for species in reaction.species:
                self._add_species(species)
            if reaction.ts is not None:
                for ts in reaction.ts:
                    ts.lattice = self.lattice
            reaction.update(T=self.T, Asite=self.Asite, L=self.z)

    def set_solvent(self, solvent):
        # Solvent will not diffuse even in diffusion system
        if solvent is not None:
            if self.solvent is not None:
                warnings.warn('Overriding old solvent {} with {}.'
                              ''.format(solvent, self.solvent),
                              RuntimeWarning, stacklevel=2)
            if not isinstance(self.species[solvent], Liquid):
                raise ValueError("Solvent must be a Liquid!")
            self.solvent = solvent

    def set_fixed(self, fixed):
        # Fixed species are removed from the differential equations
        if isinstance(fixed, str):
            fixed = [fixed]
        for name in fixed:
            if name not in self.fixed:
                self.fixed.append(name)

    def _add_species(self, species):
        assert isinstance(species, _Thermo)
        # Do nothing if we already know about the species
        if species in self._species or species in self.vacancy:
            return
        # Add the species to the list of known species
        species.lattice = self.lattice
        self.species[species.label] = species
        self._species.append(species)
        # Add the sites that species occupies to the list of known vacancies.
        if species.sites is not None:
            for site in species.sites:
                if site not in self.vacancy:
                    if site in self._species:
                        self._species.remove(site)
                    self.vacancy.append(site)
                    self.vacspecies[site] = [species]
                else:
                    self.vacspecies[site].append(species)

    def set_T(self, T):
        self._T = T
        for reaction in self._reactions:
            reaction.update(T=T)
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_T(self):
        return self._T

    T = property(get_T, set_T, doc='Model temperature')

    def set_Asite(self, Asite):
        self._Asite = Asite
        for reaction in self._reactions:
            reaction.update(Asite=Asite)
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_Asite(self):
        return self._Asite

    Asite = property(get_Asite, set_Asite, doc='Area of an adsorption site')

    def set_z(self, z):
        self._z = z
        self.check_diffusion()
        for reaction in self._reactions:
            reaction.update(L=z)
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_z(self):
        return self._z

    z = property(get_z, set_z, doc='Diffusion length')

    def set_nz(self, nz):
        self._nz = nz
        self.check_diffusion()
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_nz(self):
        return self._nz

    nz = property(get_nz, set_nz, doc='Number of diffusion grid points')

    def set_lattice(self, lattice):
        if isinstance(lattice, Lattice) or lattice is None:
            self._lattice = lattice
        elif isinstance(lattice, dict):
            self._lattice = Lattice(lattice)
        else:
            raise ValueError('Unable to parse lattice!')
        for species in self._species:
            species.set_lattice(self.lattice)
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_lattice(self):
        return self._lattice

    lattice = property(get_lattice, set_lattice, doc='Model lattice')

    def set_up_grid(self):
        if self.shape.upper() != 'FLAT':
            raise NotImplementedError
        self.dz = self.z / self.nz
        self.dV = _Nav * self.Asite * self.dz * 1000
        assert self.V - self.dV * self.nz > 0, "Quiescent layer is too thick!"

    def set_initial_conditions(self, U0):
        if self.initialized:
            self.finalize()

        # Reorder species such that Liquid -> Gas -> Adsorbate -> Vacancy
        # Steady-state species go to the end.
        newspecies = []
        self.nliquid = 0
        for species in self._species:
            if isinstance(species, Liquid):
                # As long as we are reordering the system, check to ensure all
                # Liquid species have a diffusion constant if we are doing
                # diffusion
                if self.diffusion:
                    assert species.D is not None, \
                        "Specify diffusion constant for {}".format(species)
                newspecies.append(species)
                self.nliquid += 1
        for species in self._species:
            if isinstance(species, (Gas, Electron)):
                newspecies.append(species)
        for species in self._species:
            if isinstance(species, Adsorbate):
                newspecies.append(species)
        self._species = newspecies

        # Set up diffusion grid, if necessary
        if self.diffusion:
            self.set_up_grid()

        # Start with the incomplete user-provided initial conditions
        self.U0 = U0.copy()

        # Initialize counter for vacancies
        occsites = {species: 0 for species in self.vacancy}

        for name in self.U0:
            try:
                species = self.species[name]
            except KeyError:
                for species in self.vacancy:
                    if species.label == name:
                        break
                else:
                    raise ValueError('Species {} is unknown!'.format(name))
            # Ignore all initial conditions for the number of empty sites
            if species in self.vacancy:
                warnings.warn('Initial condition for vacancy concentration '
                              'ignored.', RuntimeWarning, stacklevel=2)
                continue

            # Tuple implies we're talking about a diffusive liquid species
            if type(species) is tuple:
                species = species[0]

            # Throw an error if the user provides the concentration for a
            # species we don't know about
            if species not in self._species:
                raise ValueError("Unknown species {}!".format(species))

            # If the species occupies a site, add its concentration to the
            # occupied sites counter
            if species.sites is not None:
                for site in species.sites:
                    occsites[site] += self.U0[name]

            # If we're specifying a Liquid concentration with diffusion
            # enabled, U0[species]
            # must equal U0[(species, nz-1)]
            if self.diffusion and isinstance(species, Liquid):
                if (species, self.nz - 1) in self.U0:
                    assert abs(self.U0[name] -
                               self.U0[(name, self.nz - 1)]) < 1e-6, \
                               "Liquid concentrations not consistent!"

        self.vactot = {}
        # Determine what the initial vacancy concentration should be
        for species in self.vacancy:
            # If a vacancy species is part of the lattice, get its maximum
            # concentration from its relative abundance. Otherwise, assume
            # it is 1.
            if self.lattice is not None and species in self.lattice.sites:
                self.vactot[species] = self.lattice.ratio[species]
            else:
                self.vactot[species] = 1.
            # Make sure there isn't too much stuff occupying each kind of
            # site on the surface.
            assert occsites[species] <= self.vactot[species], \
                    "Too many adsorbates on {}!".format(species)
            # Normalize the concentration of empty sites to match the
            # appropriate site ratio from the lattice.
            self.U0[name] = self.vactot[species] - occsites[species]
#            if species not in U0:
#                self.vactot[species] = 1.
#                U0[species] = 1. - occsites[species]
#            else:
#                self.vactot[species] = U0[species] + occsites[species]

        # Populate dictionary of initial conditions for all species
        for name, species in self.species.items():
            # Assume concentration of unnamed species is 0
            if name not in self.U0:
                self.U0[name] = 0.
            # If we are doing diffusion, liquid species should have initial
            # concentrations for ALL diffusion grid points. We will assume a
            # flat concentration profile initially at the user specified
            # bulk concentration (or 0 if the user didn't specify anything)
            if self.diffusion and isinstance(species, Liquid):
                U0i = self.U0[name]
                if species.label != self.solvent:
                    for i in range(self.nz):
                        # The user can also specify concentration at individual
                        # grid points
                        if (name, i) not in self.U0:
                            self.U0[(name, i)] = U0i
                        else:
                            U0i = self.U0[(name, i)]

        # The number of variables that will be in our differential equations
        size = len(self._species)
        # Each diffusion grid point adds nliquid variables
        if self.diffusion:
            size += (self.nz - 1) * self.nliquid
            # If the solvent is a liquid, don't count it
            if self.solvent is not None:
                size -= self.nz - 1
        # Initialize "mass matrix", which is a diagonal matrix with 1s for
        # differential elements and 0s for algebraic elements (steady-state)
        M = np.ones(size, dtype=int)

        # This creates a symbol for each species named modelparamX where X
        # is a three-digit numerical identifier that corresponds to its
        # position in the order of species
        self.symbols_all = []
        # symbols_dict a Thermo object and returns its corresponding symbol
        self.symbols_dict = OrderedDict()
        # symbols ONLY includes species that will be in the differential
        # equations. Fixed species are not included in this list
        self.symbols = []

        for species in self._species:
            self.symbols_all.append(species.symbol)
            self.symbols_dict[species] = species.symbol
            if self.diffusion and isinstance(species, Liquid):
                if species.label == self.solvent:
                    continue
                self.symbols_dict[(species, self.nz - 1)] = species.symbol
                for j in range(self.nz - 1):
                    newsymbol = sym.Symbol(str(species.symbol) + str(j).zfill(3))
                    self.symbols_all.append(newsymbol)
                    self.symbols_dict[(species, j)] = newsymbol
                    self.symbols.append(newsymbol)
                if species.label not in self.fixed:
                    self.symbols.append(species.symbol)
            else:
                if species.label not in self.fixed and species.label != self.solvent:
                    self.symbols.append(species.symbol)

        # subs converts a species symbol to either its initial value if
        # it is fixed or to a constraint (such as constraining the total
        # number of adsorption sites)
        subs = {}

        # A vacancy will be represented by the total number of sites
        # minus the symbol of each species that occupies one of its sites.
        for vacancy in self.vacancy:
            vacsymbols = self.vactot[vacancy]
            for species in self.vacspecies[vacancy]:
                vacsymbols -= species.symbol
            subs[vacancy.symbol] = vacsymbols

        # nsymbols is the size of the differential equations
        self.nsymbols = len(self.symbols)

        # known_symbols keeps track of user-provided symbols that the
        # model has seen, so that symbols referring to species not in
        # the model can be later removed.
        known_symbols = set()
        for species in self._species + self.vacancy:
            known_symbols.add(species.symbol)

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

        for species in self._species:
            f = 0
            # DETAILED BALANCE NOTE: See discussion in _rate_init
            if self.diffusion and isinstance(species, Liquid) \
                    and species.label != self.solvent:
                for i in range(self.nz):
                    if i == self.nz - 1 and species.label in self.fixed:
                        continue
                    f = 0
                    # At each grid point, add all-liquid reaction rates
                    for j, rate in enumerate(self.rates):
                        f += self.rate_count[j][(species, i)] * rate
                    diff = species.D / self.dz
                    # For all grid points except the first, add diffusion away
                    # from the surface
                    if i > 0:
                        f += diff * (self.symbols_dict[(species, i-1)] -
                                     self.symbols_dict[(species, i)]) \
                                             / self.dz
                    # For all grid points except the last, add diffusion
                    # towards the surface
                    if i < self.nz - 1:
                        f += diff * (self.symbols_dict[(species, i+1)] -
                                     self.symbols_dict[(species, i)]) \
                                             / self.dz
                    # At the surface, account for any adsorption reactions
                    if i == 0:
                        for j, rate in enumerate(self.rates):
                            if self.is_rate_ads[j]:
                                f += self.rate_count[j][species] * rate * \
                                        self.V / self.dV
                    # If we were looking at the last grid point and the species
                    # is fixed, discard what we just calculated
                    self.f_sym.append(f)
                else:
                    # This is necessary to avoid double-counting non-fixed
                    # liquid species. continue skips the below "if" statement
                    continue
            elif species not in self.vacancy and species.label not in self.fixed:
                # Reactions involving on-surface species
                for i, rate in enumerate(self.rates):
                    f += self.rate_count[i][species] * rate
            # Discard rates for fixed species
            if species.label not in self.fixed and species.label != self.solvent:
                self.f_sym.append(f)
            else:
                assert f == 0, "Fixed species rate of change not zero!"

        # subs is a dictionary whose keys are internal species symbols and
        # whose values are the initial concentrations of that species if it
        # is known, or 0 otherwise.

        # All symbols referring to unknown species are going to be replaced
        # by 0
        unknown_symbols = set()
        for f in self.f_sym:
            unknown_symbols.update(f.atoms(sym.Symbol))
        unknown_symbols -= known_symbols
        unknown_symbols -= set(self.symbols_all)
        subs.update({symbol: 0 for symbol in unknown_symbols})

        # Fixed species must have their symbols replaced by their fixed
        # initial values.
        for species in self._species:
            if species.label in self.fixed or species.label == self.solvent:
                liquid = False
                label = species.label
                if self.diffusion and isinstance(species, Liquid) \
                        and species.label != self.solvent:
                    label = (species.label, self.nz - 1)
                    liquid = True
                U0i = self.U0[label]
                if liquid or isinstance(species, _Fluid):
                    U0i *= self.V
                subs[species.symbol] = U0i

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

        # Additionally, insert fixed species concentrations into rate
        # expressions
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
                    if isinstance(species, tuple):
                        U0i = self.U0[(species[0].label, species[1])]
                    else:
                        U0i = self.U0[species.label]
                    if type(species) is tuple \
                            or isinstance(species, _Fluid):
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

        for reaction in self._reactions:
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
                rate_for = reaction.get_kfor(self.T, self.Asite, self.dz/2.)
                rate_rev = reaction.get_krev(self.T, self.Asite, self.dz/2.)
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
            # As far as the user can tell, all fluids (Gas, Liquid) are
            # represented in units of concentration (i.e. mol/L) and all
            # adsorbates are represented in coverage (i.e. N/M, where M is the
            # total number of sites). INTERNALLY, this is not the case.
            # Internally, ALL species are number fractions relative to the
            # default number of catalytic sites (though this can be changed).
            # This means that in order to achieve detailed balance, the
            # reaction rates must be modulated by the volume. Further sections
            # of the code that pertain to this behavior will be highlighted
            # with "DETAILED BALANCE NOTE"

            # Initialize dictionary for reaction stiochiometry to 0 for all
            # species in the model
            rate_count = {}
            for species in self._species:
                rate_count[species] = 0
                if self.diffusion and isinstance(species, Liquid):
                    for i in range(self.nz):
                        rate_count[(species, i)] = 0
            for vacancy in self.vacancy:
                rate_count[vacancy] = 0

            # Reactants are consumed in the forward direction
            for species in reaction.reactants:
                # Multiply the rate by the species' symbol UNLESS it is an
                # electron. All reactions are 0th order in electrons, even
                # if they consume/produce electrons.
                if not isinstance(species, Electron):
                    rate_for *= species.symbol
                rate_count[species] -= 1

            # Products are created in the forward direction
            for species in reaction.products:
                if not isinstance(species, Electron):
                    rate_rev *= species.symbol
                rate_count[species] += 1

            # Overall reaction rate (flux)
            rate = rate_for
            if reaction.reversible:
                rate -= rate_rev

            # For adsorption reactions, scale rates by the volume of the
            # first layer
            if reaction.adsorption and self.diffusion:
                rate *= self.dV / self.V

            # Append all data to global rate lists
            self.rates.append(rate)
            self.rate_count.append(rate_count)
            self.is_rate_ads.append(reaction.adsorption)

            # If a reaction is occuring between all liquid species, then we
            # have to perform the same reaction at all grid points. This is
            # an exact mirror of what occurred above, but at different grid
            # points.
            if self.diffusion and reaction.all_liquid:
                for i in range(self.nz - 1):
                    new_rate_count = {}
                    for species in self._species:
                        new_rate_count[species] = 0
                        if isinstance(species, Liquid):
                            for j in range(self.nz):
                                new_rate_count[(species, j)] = 0
                    subs = {}
                    for species in self._species:
                        if isinstance(species, Liquid) \
                                and species.label != self.solvent:
                            new_rate_count[(species, i)] = rate_count[species]
                            subs[self.symbols_dict[species]] = \
                                self.symbols_dict[(species, i)]

                    self.rates.append(rate.subs(subs))
                    self.rate_count.append(new_rate_count)
                    # This should always be False:
                    self.is_rate_ads.append(reaction.adsorption)

    def setup_execs(self):
        from micki.fortran import f90_template, pyf_template
        from numpy import f2py

        # y_vec is an array symbol that will represent the species
        # concentrations provided by the differential equation solver inside
        # the Fortran code (that is, y_vec is an INPUT to the functions that
        # calculate the residual, Jacobian, and rate)
        y_vec = sym.IndexedBase('yin', shape=(self.nsymbols,))
        # Map y_vec elements (1-indexed, of course) onto 'modelparam' symbols
        trans = {self.symbols[i]: y_vec[i + 1] for i in range(self.nsymbols)}
        # Map string represntation of 'modelparam' symbols onto string
        # representation of y-vec elements
        str_trans = {}
        for i in range(self.nsymbols):
            str_trans[sym.fcode(self.symbols[i], source_format='free')] = \
                    sym.fcode(y_vec[i + 1], source_format='free')
        
        str_list = [key for key in str_trans]
        str_list.sort(key=len, reverse=True)

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
            for key in str_list:
                fcode = fcode.replace(key, str_trans[key])
            # Create actual line of code for calculating residual
            rescode.append('   res({}) = '.format(i + 1) + fcode)

        # Effectively the same as above, except on the two-dimensional Jacobian
        # matrix.
        for i in range(self.nsymbols):
            for j in range(self.nsymbols):
                expr = self.jac_sym[j, i]
                # Unlike the residual, some elements of the Jacobian can be 0.
                # We don't need to bother writing 'jac(x,y) = 0' a hundred
                # times in Fortran, so we omit those.
                if expr != 0:
                    fcode = sym.fcode(expr, source_format='free')
                    for key in str_list:
                        fcode = fcode.replace(key, str_trans[key])
                    jaccode.append('   jac({}, {}) = '.format(j + 1, i + 1) +
                                   fcode)

        # See residual above
        for i, rate in enumerate(self.rates):
            fcode = sym.fcode(rate, source_format='free')
            for key in str_list:
                fcode = fcode.replace(key, str_trans[key])
            ratecode.append('   rates({}) = '.format(i + 1) + fcode)

        # We insert all of the parameters of this differential equation into
        # the prewritten Fortran template, including the residual, Jacobian,
        # and rate expressions we just calculated.
        program = f90_template.format(neq=self.nsymbols, nx=1,
                                      nrates=len(self.rates),
                                      rescalc='\n'.join(rescode),
                                      jaccalc='\n'.join(jaccode),
                                      ratecalc='\n'.join(ratecode))

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
            f.write(pyf_template.format(modname=modname, neq=self.nsymbols,
                    nrates=len(self.rates)))

        # Compile the module with f2py
        f2py.compile(program, modulename=modname,
                     extra_args='--quiet '
                                '--f90flags="-Wno-unused-dummy-argument '
                                '-Wno-unused-variable -w -fopenmp" ' 
                                '-lsundials_fcvode '
                                '-lsundials_cvode -lsundials_fnvecopenmp '
                                '-lsundials_nvecopenmp -lmkl_rt -lgomp ' +
                                os.path.join(dname, pyfname),
                     source_fn=os.path.join(dname, fname), verbose=0)

        # Delete the temporary directory
        shutil.rmtree(dname)

        # Import the module on-the-fly with __import__. This is kind of a hack.
        solve_ida = __import__(modname)

        # The Fortran module's initialize, solve, and finalize routines
        # are mapped onto finitialize, fsolve, and ffinalize inside the Model
        # object. We don't want users touching these manually
        self.finitialize = solve_ida.initialize
        self.ffind_steady_state = solve_ida.find_steady_state
        self.fsolve = solve_ida.solve
        self.ffinalize = solve_ida.finalize

        # Delete the module file. We've already imported it, so it's in memory.
        os.remove(modname + '.so')

    def _out_array_to_dict(self, U, dU, r):
        Ui = {}
        dUi = {}
        ri = {}
        fixed = self.fixed
        if self.solvent is not None:
            fixed += [self.solvent]
        for name in fixed:
            if self.diffusion and isinstance(self.species[name], Liquid) \
                    and name != self.solvent:
                name = (name, self.nz - 1)
            dUi[name] = 0.
            Ui[name] = self.U0[name]
        for j, symbol in enumerate(self.symbols):
            for species, isymbol in self.symbols_dict.items():
                if symbol == isymbol:
                    if type(species) is tuple:
                        label = (species[0].label, species[1])
                        species = species[0]
                    else:
                        label = species.label

                    Uij = U[j]
                    dUij = dU[j]
                    if isinstance(species, _Fluid):
                        Uij /= self.V
                        dUij /= self.V

                    Ui[label] = Uij
                    dUi[label] = dUij
        for vacancy in self.vacancy:
            Ui[vacancy.label] = self.vactot[vacancy]
            for species in self.vacspecies[vacancy]:
                Ui[vacancy.label] -= Ui[species.label]
            dUi[vacancy.label] = 0

        j = 0
        rxn_to_name = {}
        for name, reaction in self.reactions.items():
            rxn_to_name[reaction] = name
        for reaction in self._reactions:
            ri[rxn_to_name[reaction]] = r[j]
            j += 1
            if self.diffusion and reaction.all_liquid:
                for i in range(self.nz - 1):
                    ri[(rxn_to_name[reaction], i)] = r[j]
                    j += 1

        return Ui, dUi, ri

    def find_steady_state(self, dt=60, maxiter=2000, epsilon=1e-6):
        t, U1, dU1, r1 = self.ffind_steady_state(self.nsymbols,
                                                 len(self.rates),
                                                 dt,
                                                 maxiter,
                                                 epsilon)
        self.t = t
        self.U = []
        self.dU = []
        self.r = []
        U, dU, r = self._out_array_to_dict(U1.T, dU1.T, r1.T)
        self.U.append(U)
        self.dU.append(dU)
        self.r.append(r)
        return t, U, r

    def solve(self, t, ncp):
        self.t, U1, dU1, r1 = self.fsolve(self.nsymbols,
                                          len(self.rates), ncp, t)
        self.U1 = U1.T
        self.dU1 = dU1.T
        self.r1 = r1.T
        self.U = []
        self.dU = []
        self.r = []
        for i, t in enumerate(self.t):
            Ui, dUi, ri = self._out_array_to_dict(self.U1[i], self.dU1[i],
                                                  self.r1[i])
            self.U.append(Ui)
            self.dU.append(dUi)
            self.r.append(ri)
        return self.U, self.r

    def finalize(self):
        self.initialized = False
#        self.ffinalize()

    def copy(self, initialize=True):
        newmodel = Model(self.T, self.Asite, self.z, self.nz, self.shape, self.lattice)
        newmodel.add_reactions(self.reactions)
        newmodel.set_fixed(self.fixed)
        newmodel.set_solvent(self.solvent)
        if initialize:
            newmodel.set_initial_conditions(self.U0)
        return newmodel
