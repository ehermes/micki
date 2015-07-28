"""Module for doing sensitivity analysis of microkinetic model"""

import numpy as np
from ase.units import kB
from mkm.model import Model

class ModelAnalysis(object):
    def __init__(self, model, product_reaction, Uequil, tol=1e-6, dt=3600):
        self.model = model
        self.product_reaction = product_reaction
        self.Uequil = Uequil
        self.tol = tol
        self.dt = dt

        assert self.product_reaction in self.model.reactions

        self.model.set_initial_conditions(self.Uequil)

        self.U, self.dU, self.r = self.model.solve(self.dt, self.dt)
        
        self.check_converged(self.U, self.dU, self.r)

    def campbell_rate_control(self, test_reaction, scale=0.05):
        assert test_reaction in self.model.reactions

        rmid = self.r[self.product_reaction]
        kmid = test_reaction.get_kfor(self.model.T, self.model.N0)

        test_reaction.set_scale('kfor', 1 - scale)
        test_reaction.set_scale('krev', 1 - scale)
        klow = test_reaction.get_kfor(self.model.T, self.model.N0)
        self.model.set_initial_conditions(self.Uequil)
        U1, dU1, r1 = self.model.solve(self.dt, 3)
        self.check_converged(U1, dU1, r1)
        rlow = r1[self.product_reaction]

        test_reaction.set_scale('kfor', 1 + scale)
        test_reaction.set_scale('krev', 1 + scale)
        khigh = test_reaction.get_kfor(self.model.T, self.model.N0)
        self.model.set_initial_conditions(self.Uequil)
        U2, dU2, r2 = self.model.solve(self.dt, 3)
        self.check_converged(U2, dU2, r2)
        rhigh = r2[self.product_reaction]

        return kmid * (rhigh - rlow) / (rmid * (khigh - klow))

    def thermodynamic_rate_control(self, test_species, scale=0.05):
        assert test_species in self.model.species
        assert test_species not in self.model.fixed

        rmid = self.r[self.product_reaction]
        T = self.model.T

        test_species.set_scale('Stot', 1 - scale)
        test_species.set_scale('Htot', 1 - scale)
        glow = test_species.get_H(T) - T * test_species.get_S(T)
        self.model.set_initial_conditions(self.Uequil)
        U1, dU1, r1 = self.model.solve(self.dt, 3)
        self.check_converged(U1, dU1, r1)
        rlow = r1[self.product_reaction]

        test_species.set_scale('Stot', 1 + scale)
        test_species.set_scale('Htot', 1 + scale)
        ghigh = test_species.get_H(T) - T * test_species.get_S(T)
        self.model.set_initial_conditions(self.Uequil)
        U2, dU2, r2 = self.model.solve(self.dt, 3)
        self.check_converged(U2, dU2, r2)
        rhigh = r2[self.product_reaction]

        return (rhigh - rlow) * kB * T / (rmid * (glow - ghigh))

    def check_converged(self, *vals):
        for val in vals:
            for i, key in enumerate(val[0]):
                if np.abs(val[-1][key] - val[-2][key]) > self.tol:
                    raise ValueError, "Calculation not converged! Increase dt or use better initial guess."
