"""Module for doing sensitivity analysis of microkinetic model"""

import collections

import numpy as np
import sympy as sym

from ase.units import kB

from micki.reactants import Adsorbate, _Fluid
from micki.model import Model


class ModelAnalysis(object):
    def __init__(self, model, product_reaction, Uequil, tol=1e-3, dt=3600):
        self.model = model
        self.reaction_name = product_reaction
        self.product_reaction = model.reactions[product_reaction]
        self.Uequil = Uequil
        self.tol = tol
        self.dt = dt

        self.model.set_initial_conditions(self.Uequil)

        self.U, self.dU, self.r = self.model.solve(self.dt, 100)
        model.finalize()

        self.check_converged(self.U, self.dU, self.r)
        self.species_symbols = []
        for species in self.model._species:
            if species.symbol is not None:
                self.species_symbols.append(species)

    def campbell_rate_control(self, rxn_name, scale=0.001):
        reaction = self.model.reactions[rxn_name]

        rmid = self.r[-1][self.reaction_name]
        keq = reaction.get_keq(self.model.T,
                               self.model.Asite,
                               self.model.z)

        subs = {}
        for species in self.species_symbols:
            subs[species.symbol] = self.U[-1][species.label]

        if isinstance(keq, sym.Basic):
            keq = keq.subs(subs)
        if keq >= 0:
            kmid = reaction.get_kfor(self.model.T,
                                     self.model.Asite,
                                     self.model.z)
        else:
            kmid = reaction.get_krev(self.model.T,
                                     self.model.Asite,
                                     self.model.z)

        if isinstance(kmid, sym.Basic):
            kmid = kmid.subs(subs)

        reaction.set_scale('kfor', 1.0 - scale)
        reaction.set_scale('krev', 1.0 - scale)
        if keq >= 0:
            klow = reaction.get_kfor(self.model.T,
                                     self.model.Asite,
                                     self.model.z)
        else:
            klow = reaction.get_krev(self.model.T,
                                     self.model.Asite,
                                     self.model.z)
        model = self.model.copy()

        try:
            U1, dU1, r1 = model.solve(self.dt, 100)
        finally:
            reaction.set_scale('kfor', 1.0)
            reaction.set_scale('krev', 1.0)

        model.finalize()
        self.check_converged(U1, dU1, r1)
        rlow = r1[-1][self.reaction_name]
        if isinstance(klow, sym.Basic):
            subs = {}
            for species in self.species_symbols:
                subs[species.symbol] = U1[-1][species.label]
            klow = klow.subs(subs)

        reaction.set_scale('kfor', 1.0 + scale)
        reaction.set_scale('krev', 1.0 + scale)
        if keq >= 0:
            khigh = reaction.get_kfor(self.model.T,
                                      self.model.Asite,
                                      self.model.z)
        else:
            khigh = reaction.get_krev(self.model.T,
                                      self.model.Asite,
                                      self.model.z)
        model = self.model.copy()

        try:
            U2, dU2, r2 = model.solve(self.dt, 100)
        finally:
            reaction.set_scale('kfor', 1.0)
            reaction.set_scale('krev', 1.0)

        model.finalize()
        self.check_converged(U2, dU2, r2)
        rhigh = r2[-1][self.reaction_name]
        if isinstance(khigh, sym.Basic):
            subs = {}
            for species in self.species_symbols:
                subs[species.symbol] = U2[-1][species.label]
            khigh = khigh.subs(subs)
        reaction.set_scale('kfor', 1.0)
        reaction.set_scale('krev', 1.0)

        return kmid * (rhigh - rlow) / (rmid * (khigh - klow))

    def thermodynamic_rate_control(self, names, scale=0.01):

        if not isinstance(names, (list, tuple)):
            species = [self.model.species[names]]
        else:
            species = [self.model.species[name] for name in names]

        rmid = self.r[-1][self.reaction_name]
        T = self.model.T
        gmid = species[0].get_H(T) - T * species[0].get_S(T)
        if isinstance(gmid, sym.Basic):
            subs = {}
            for sp in self.species_symbols:
                subs[sp.symbol] = self.U[-1][sp.label]
            gmid = gmid.subs(subs)
        dg = abs(gmid * 2 * scale)

        def set_dg(species, dg):
            species.dE += dg

        for sp in species:
            set_dg(sp, -dg)

        for reaction in self.model._reactions:
            reaction.update(T=self.model.T,
                            Asite=self.model.Asite,
                            L=self.model.z,
                            force=True)

        model = self.model.copy()

        try:
            U1, dU1, r1 = model.solve(self.dt, 100)
        finally:
            for sp in species:
                set_dg(sp, dg)

        model.finalize()
        self.check_converged(U1, dU1, r1)
        rlow = r1[-1][self.reaction_name]

        for sp in species:
            set_dg(sp, dg)

        for reaction in self.model._reactions:
            reaction.update(T=self.model.T,
                            Asite=self.model.Asite,
                            L=self.model.z,
                            force=True)

        model = self.model.copy()

        try:
            U2, dU2, r2 = model.solve(self.dt, 100)
        finally:
            for sp in species:
                set_dg(sp, -dg)

        for reaction in self.model._reactions:
            reaction.update(T=self.model.T,
                            Asite=self.model.Asite,
                            L=self.model.z,
                            force=True)

        model.finalize()
        self.check_converged(U2, dU2, r2)
        rhigh = r2[-1][self.reaction_name]

        return (rlow - rhigh) * kB * T / (rmid * dg)

    def activation_barrier(self, dT=0.01):
        rmid = self.r[-1][self.reaction_name]
        T = self.model.T

        model = self.model.copy(initialize=False)
        model.set_temperature(T - dT)
        model.set_initial_conditions(self.Uequil)
        U1, dU1, r1 = model.solve(self.dt, 100)
        model.finalize()
        self.check_converged(U1, dU1, r1)

        rlow = r1[-1][self.reaction_name]

        model = self.model.copy(initialize=False)
        model.set_temperature(T + dT)
        model.set_initial_conditions(self.Uequil)
        U2, dU2, r2 = model.solve(self.dt, 100)
        model.finalize()
        self.check_converged(U2, dU2, r2)

        rhigh = r2[-1][self.reaction_name]

        return kB * T**2 * (rhigh - rlow) / (rmid * 2 * dT)

    def rate_order(self, name, drho=0.05):
        species = self.model.species[name]

        rhomid = self.Uequil[species.label]
        assert rhomid > 0

        rmid = self.r[-1][self.reaction_name]
        U0 = self.Uequil.copy()
        rholow = rhomid * (1.0 - drho)
        U0[species.label] = rholow
        model = self.model.copy(initialize=False)
        model.set_initial_conditions(U0)
        U1, dU1, r1 = model.solve(self.dt, 100)
        model.finalize()
        self.check_converged(U1, dU1, r1)
        rlow = r1[-1][self.reaction_name]

        rhohigh = rhomid * (1.0 + drho)
        U0[species.label] = rhohigh
        model.set_initial_conditions(U0)
        U2, dU2, r2 = model.solve(self.dt, 100)
        model.finalize()
        self.check_converged(U2, dU2, r2)
        rhigh = r2[-1][self.reaction_name]

        return (rhomid / rmid) * (rhigh - rlow) / (rhohigh - rholow)

    def drate_order_dg(self, fluid, adsorbates, rho_scale=0.01, g_scale=0.01):
        assert isinstance(fluid, _Fluid)

        if not isinstance(adsorbates, collections.Iterable):
            adsorbates = [adsorbates]

        assert isinstance(adsorbates[0], Adsorbate)

        rhomid = self.U[-1][fluid]
        assert rhomid > 0
        rmid = self.r[-1][self.reaction_name]
        gmid = adsorbates[0].get_G(self.model.T)
        if isinstance(gmid, sym.Basic):
            trans = {}
            for species in self.model._species:
                if isinstance(species, Adsorbate) \
                        and species.symbol is not None:
                    trans[species.symbol] = self.U[-1][species.label]
            gmid = gmid.subs(trans)
        dg = np.abs(gmid * g_scale * 2)
        drho = rhomid * rho_scale * 2

        dr = 0

        def set_dg(species, dg):
            species.dE += dg

        for i in [-1, 1]:
            for adsorbate in adsorbates:
                set_dg(adsorbate, i * dg)

            for reaction in self.model._reactions:
                reaction.update(T=self.model.T,
                                Asite=self.model.Asite,
                                L=self.model.z,
                                force=True)

            for j in [-1, 1]:
                U0 = self.Uequil.copy()
                U0[fluid] = rhomid + j * drho

                model = self.model.copy(initialize=False)
                model.set_initial_conditions(U0)

                Ui, dUi, ri = model.solve(self.dt, 100)
                dr += i * j * ri[-1][self.product_reaction]

            for adsorbate in adsorbates:
                set_dg(adsorbate, -i * dg)

        for reaction in self.model._reactions:
            reaction.update(T=self.model.T,
                            Asite=self.model.Asite,
                            L=self.model.z,
                            force=True)

        return (rhomid / rmid) * dr / (dg * drho)

    def check_converged(self, *vals):
        for val in vals:
            for i, key in enumerate(val[0]):
                if np.abs(val[-1][key] - val[-2][key]) > self.tol:
                    print(key, val[-1][key], val[-1][key] - val[-2][key])
                    raise ValueError("Calculation not converged! Increase "
                                     "dt or use better initial guess.")
