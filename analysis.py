"""Module for doing sensitivity analysis of microkinetic model"""

import numpy as np
from ase.units import kB
from micki.reactants import Adsorbate, _Fluid
from micki.model import Model

class ModelAnalysis(object):
    def __init__(self, model, product_reaction, Uequil, tol=1e-3, dt=3600):
        self.model = model
        self.product_reaction = product_reaction
        self.Uequil = Uequil
        self.tol = tol
        self.dt = dt

        assert self.product_reaction in self.model.reactions

        self.model.set_initial_conditions(self.Uequil)

        self.U, self.dU, self.r = self.model.solve(self.dt, 4)
        
        self.check_converged(self.U, self.dU, self.r)

    def campbell_rate_control(self, test_reaction, scale=0.001):
        assert test_reaction in self.model.reactions

        rmid = self.r[-1][self.product_reaction]
        kmid = test_reaction.get_kfor(self.model.T, self.model.N0)

        test_reaction.set_scale('kfor', 1.0 - scale)
        test_reaction.set_scale('krev', 1.0 - scale)
        klow = test_reaction.get_kfor(self.model.T, self.model.N0)
        model = self.model.copy()
        try:
            U1, dU1, r1 = model.solve(self.dt, 4)
        except:
            test_reaction.set_scale('kfor', 1.0)
            test_reaction.set_scale('krev', 1.0)
            raise
        self.check_converged(U1, dU1, r1)
        rlow = r1[-1][self.product_reaction]

        test_reaction.set_scale('kfor', 1.0 + scale)
        test_reaction.set_scale('krev', 1.0 + scale)
        khigh = test_reaction.get_kfor(self.model.T, self.model.N0)
        model = self.model.copy()
        try:
            U2, dU2, r2 = model.solve(self.dt, 4)
        except:
            test_reaction.set_scale('kfor', 1.0)
            test_reaction.set_scale('krev', 1.0)
            raise
        self.check_converged(U2, dU2, r2)
        rhigh = r2[-1][self.product_reaction]
        test_reaction.set_scale('kfor', 1.0)
        test_reaction.set_scale('krev', 1.0)

        return kmid * (rhigh - rlow) / (rmid * (khigh - klow))

    def thermodynamic_rate_control(self, test_species, scale=0.0001):
        assert isinstance(test_species, Adsorbate)
        assert test_species in self.model.species
        assert test_species not in self.model.fixed

        rmid = self.r[-1][self.product_reaction]
        T = self.model.T

        test_species.scale['S']['tot'] = 1.0 - scale
        test_species.scale['H']['tot'] = 1.0 - scale
        model = self.model.copy()
        try:
            U1, dU1, r1 = model.solve(self.dt, 4)
        except:
            test_species.scale['S']['tot'] = 1.0
            test_species.scale['H']['tot'] = 1.0
            raise
        glow = test_species.get_H(T) - T * test_species.get_S(T)
        self.check_converged(U1, dU1, r1)
        rlow = r1[-1][self.product_reaction]

        test_species.scale['S']['tot'] = 1.0 + scale
        test_species.scale['H']['tot'] = 1.0 + scale
        model = self.model.copy()
        try:
            U2, dU2, r2 = model.solve(self.dt, 4)
        except:
            test_species.scale['S']['tot'] = 1.0
            test_species.scale['H']['tot'] = 1.0
        ghigh = test_species.get_H(T) - T * test_species.get_S(T)
        self.check_converged(U2, dU2, r2)
        rhigh = r2[-1][self.product_reaction]

        test_species.scale['S']['tot'] = 1.0
        test_species.scale['H']['tot'] = 1.0

        return (rhigh - rlow) * kB * T / (rmid * (glow - ghigh))

    def activation_barrier(self, dT=0.01):
        rmid = self.r[-1][self.product_reaction]
        T = self.model.T

        model = self.model.copy()
        model.set_temperature(T - dT)
        U1, dU1, r1 = model.solve(self.dt, 4)
        self.check_converged(U1, dU1, r1)

        rlow = r1[-1][self.product_reaction]

        model.set_temperature(T + dT)
        U2, dU2, r2 = model.solve(self.dt, 4)
        self.check_converged(U2, dU2, r2)

        rhigh = r2[-1][self.product_reaction]

        return kB * T**2 * (rhigh - rlow) / (rmid * 2 * dT)

    def rate_order(self, test_species, drho=0.001):
        assert isinstance(test_species, _Fluid)

        rhomid = self.Uequil[test_species]
        assert rhomid > 0

        rmid = self.r[-1][self.product_reaction]
        U0 = self.Uequil.copy()
        rholow = rhomid * (1.0 - drho)
        U0[test_species] = rholow
        model = self.model.copy()
        model.set_initial_conditions(U0)
        U1, dU1, r1 = model.solve(self.dt, 4)
        self.check_converged(U1, dU1, r1)
        rlow = r1[-1][self.product_reaction]

        rhohigh = rhomid * (1.0 + drho)
        U0[test_species] = rhohigh
        model.set_initial_conditions(U0)
        U2, dU2, r2 = model.solve(self.dt, 4)
        self.check_converged(U2, dU2, r2)
        rhigh = r2[-1][self.product_reaction]

        return (rhomid / rmid) * (rhigh - rlow) / (rhohigh - rholow)
        
    def check_converged(self, *vals):
        for val in vals:
            for i, key in enumerate(val[0]):
                if np.abs(val[-1][key] - val[-2][key]) > self.tol:
                    raise ValueError, "Calculation not converged! Increase dt or use better initial guess."
