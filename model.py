"""Microkinetic modeling objects"""

import numpy as np

from scipy.integrate import odeint

from ase.units import kB, _hplanck, kg, _k, _Nav, mol

from odespy import Radau5Implicit

import sympy as sym

from mkm.reactants import _Thermo, _Reactants


class Reaction(object):
    def __init__(self, reactants, products, ts=None, method=None, S0=1., \
            dh_scale=1.0, ds_scale=1.0, dh_act_scale=1.0, ds_act_scale=1.0):
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
        self.method = method
        if isinstance(self.method, str):
            self.method =self.method.upper()
        if self.method not in [None, 'CT']:
            raise ValueError, "Method {} unrecognized!".format(self.method)
        self.S0 = S0
        self.dh_scale = dh_scale
        self.ds_scale = ds_scale
        self.dh_act_scale = dh_act_scale
        self.ds_act_scale = ds_act_scale
        self.keq = None
        self.kfor = None
        self.krev = None
        self.T = None
        self.N0 = None

    def update(self, T=None, N0=None):
        if self.keq is not None and self.kfor is not None and self.krev is not None:
            if T is None or T == self.T:
                if N0 is None or N0 == self.N0:
                    return

        self.T = T
        self.N0 = N0
        self._calc_keq(T)
        self._calc_kfor(T, N0)
        self._calc_krev(T, N0)

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
        self.keq = self.products.get_q(T) / self.reactants.get_q(T)

    def _calc_kfor(self, T, N0):
        if self.ts is None:
            if self.method is None:
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
            self.kfor = (_k * T / _hplanck) * self.ts.get_q(T) / self.reactants.get_q(T)

    def _calc_krev(self, T, N0):
        if self.ts is None:
            self.krev = _k * T / _hplanck
            if self.keq > 1:
                self.krev /= self.keq
            else:
                self.krev = self.kfor / self.keq
        else:
            self.krev = (_k * T / _hplanck) * self.ts.get_q(T) / self.products.get_q(T)

    def set_scale(self, dh=None, ds=None, dh_act=None, ds_act=None):
        if dh is not None:
            self.dh_scale = dh
        if ds is not None:
            self.ds_scale = ds
        if dh_act is not None:
            self.dh_act_scale = dh_act
        if ds_act is not None:
            self.ds_act_scale = ds_act

    def __repr__(self):
        string = self.reactants.__repr__() + ' <-> '
        if self.ts is not None:
            string += self.ts.__repr__() + ' <-> '
        string += self.products.__repr__()
        return string


class Model(object):
    def __init__(self, reactions, T, V, nsites, coverage=1.0):
        self.reactions = []
        self.species = []
        for reaction in reactions:
            assert isinstance(reaction, Reaction)
            self.reactions.append(reaction)
            for species in reaction.reactants:
                self.add_species(species)
            for species in reaction.products:
                self.add_species(species)
        self.T = T
        self.V = V
        self.nsites = nsites
        self.coverage = coverage

    def add_reaction(self, reaction):
        assert isinstance(reaction, Reaction)
        self.reactions.append(reaction)
        for species in reaction.reactants:
            self.add_species(species)

    def add_species(self, species):
        assert isinstance(species, _Thermo)
        if species not in self.species:
            self.species.append(species)

    def set_vacancy(self, species, N0):
        assert species.gas is False
        if species not in self.species:
            self.species.append(species)
        self.vacancy = species
        newspecies = []
        self.ngas = 0
        for species in self.species:
            if species.gas:
                self.ngas += 1
                newspecies.append(species)
        for species in self.species:
            if not species.gas and not species.transient and species is not self.vacancy:
                newspecies.append(species)
        for species in self.species:
            if not species.gas and species.transient and species is not self.vacancy:
                newspecies.append(species)
        newspecies.append(self.vacancy)
        self.species = newspecies
        self.N0 = N0

    def set_initial_conditions(self, U0):
        self.U0 = []
        for species in U0:
            if species not in self.species:
                raise ValueError, "Unknown species!"
        for species in self.species:
            if species in U0:
                self.U0.append(U0[species])
            else:
                self.U0.append(0.)
        self._initialize()

    def _initialize(self):
        raise NotImplementedError

    def _rate_calc(self):
        self.rates = []
        self.rate_count = []
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
            self.rates.append(rate_for - rate_rev)
            self.rate_count.append(rate_count)

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

    def solve_scipy(self, t):
        self.t = t
        self.U1 = odeint(self.f, self.U0, t, Dfun=self.jac, rtol=1e-8, mxstep=5000000)
        return self._results()

    def _results(self):
        self.U = {species: [] for species in self.species}
        for i, t in enumerate(self.t):
            for j, species in enumerate(self.species):
                self.U[species].append(self.U1[i][j])
        return self.U, self.t


class GasPhaseModel(Model):
    def _initialize(self):
        # Initialize the mass matrix
        self.M = np.identity(len(self.species), dtype=int)
        for i, species in enumerate(self.species):
            if species.transient or species is self.vacancy:
                self.M[i, i] = 0
        # Create sympy symbols for creating f and jac
        self.symbols = sym.symbols('x0:{}'.format(len(self.species)))
        self.symbols_dict = {}
        for i, species in enumerate(self.species):
            self.symbols_dict[species] = self.symbols[i]
        # Set up rates
        self._rate_calc()
        self.f_sym = []
        self.f_exec = []
        for species in self.species:
            f = 0
            if species is self.vacancy:
                f = 1
                for a in self.species:
                    if not a.gas:
                        f -= self.symbols_dict[a] * a.coord
            elif species.gas:
                if not species.fixed:
                    for i, rate in enumerate(self.rates):
                        f += self.rate_count[i][species] * rate * self.nsites / self.V
            else:
                for i, rate in enumerate(self.rates):
                    f += self.rate_count[i][species] * rate
            self.f_sym.append(f)
            self.f_exec.append(sym.lambdify(self.symbols, f))

        self.jac_exec = []
        for f in self.f_sym:
            jac_exec = []
            for species in self.species:
                jac_exec.append(sym.lambdify(self.symbols, sym.diff(f, \
                        self.symbols_dict[species])))
            self.jac_exec.append(jac_exec)
        self.model = Radau5Implicit(f=self.f, jac=self.jac, mas=self.mas, \
                rtol=1e-8)
        self.model.set_initial_condition(self.U0)


class LiquidPhaseModel(Model):
    def __init__(self, reactions, T, V, nsites, z, nz, shape='FLAT', coverage=1.0):
        self.z = z
        self.nz = nz
        self.shape = shape
        assert self.nz >= 3, "Too few grid points!"
        if self.shape.upper() == 'FLAT':
            self.dz = np.ones(self.nz - 1, dtype=float)
        elif self.shape.upper() == 'GAUSSIAN':
            self.dz = 1. / (np.exp(10 * (0.5 - np.arange(self.nz - 1, dtype=float) \
                    / (self.nz - 2))) + 1.)
        elif self.shape.upper() == 'EXP':
            self.dz = np.exp(7. * np.arange(self.nz - 1, dtype=float) / (self.nz - 2))
        self.dz *= self.z / self.dz.sum()
        self.zi = np.zeros(self.nz, dtype=float)
        for i in xrange(nz):
            if i < self.nz - 1:
                self.zi[i] += self.dz[i]/2.
                if i > 0:
                    self.zi[i] += self.dz[i-1]/2.
        super(LiquidPhaseModel, self).__init__(reactions, T, V, nsites, coverage)

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
            if species.gas:
                for i in xrange(self.nz):
                    self.U0.append(U0i)
            else:
                self.U0.append(U0i)
        self._initialize()

    def _initialize(self):
        size = (self.nz - 1) * self.ngas + len(self.species)
        self.M = np.identity(size, dtype=int)
        i = 0
        for species in self.species:
            if species.gas:
                i += self.nz
            else:
                if species.transient or species is self.vacancy:
                    self.M[i, i] = 0
                i += 1
#        self.M[0, 0] = 0
#        self.M = np.zeros((size, size), dtype=int)
#        self.M[0, 0] = 1
#        self.M[-2, -2] = 1
        self.symbols = sym.symbols('x0:{}'.format(size))
        self.symbols_dict = {}
        j = 0
        for species in self.species:
            if species.gas:
                self.symbols_dict[species] = self.symbols[j]
                for i in xrange(self.nz):
                    self.symbols_dict[(species, i)] = self.symbols[j]
                    j += 1
            else:
                self.symbols_dict[species] = self.symbols[j]
                j += 1

        # Volume of grid thickness per ads. site
        self.dV = mol * self.zi * self.nsites * 1000 / (self.N0 * self.coverage)
        self.dV[-1] = self.V - self.dV.sum()
        self.zi[-1] = self.N0 * self.coverage * self.dV[-1] / (1000 * mol * self.nsites)
        self.dz = list(self.dz)
        self.dz.append(2 * self.zi[-1] - self.dz[-2])
        self.dz = np.array(self.dz)
        self._rate_calc()
        self.f_sym = []
        self.f_exec = []
        for species in self.species:
            if species.gas:
                for i in xrange(self.nz):
                    f = 0
                    diff = 2 * species.D / self.zi[i]
                    if i > 0:
                        if not (i == self.nz - 1 and species.fixed):
                            f += diff * (self.symbols_dict[(species, i-1)] \
                                    - self.symbols_dict[(species, i)]) / self.dz[i-1]
                    if i < self.nz - 1:
                        f += diff * (self.symbols_dict[(species, i+1)] \
                                - self.symbols_dict[(species, i)]) / self.dz[i]
                    if i == 0:
                        for j, rate in enumerate(self.rates):
                            f += self.rate_count[j][species] * rate * self.nsites \
                                    / self.dV[0]
                    self.f_sym.append(f)
                    self.f_exec.append(sym.lambdify(self.symbols, f))
                else:
                    continue
            elif species is not self.vacancy:
                f = 0
                for i, rate in enumerate(self.rates):
                    f += self.rate_count[i][species] * rate
            else:
                f = 1
                for a in self.species:
                    if not a.gas:
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
        self.U =[]
        for i, t in enumerate(self.t):
            Ui = {}
            j = 0
            for species in self.species:
                if species.gas:
                    for k in xrange(self.nz):
                        Ui[(species, k)] = self.U1[i][j]
                        j += 1
                else:
                    Ui[species] = self.U1[i][j]
                    j += 1
            self.U.append(Ui)
        return self.U, self.t

class DummyDiffusionModel(Model):
    def __init__(self, species, V, z, nz, shape='FLAT'):
        self.reactions = []
        self.species = [species]
        self.V = V
        self.z = z
        self.nz = nz
        assert self.nz >= 3, "Too few grid points!"
        if shape.upper() == 'FLAT':
            self.dz = np.ones(self.nz) * self.z / self.nz
        elif shape.upper() == 'GAUSSIAN':
            self.dz = self._gauss(np.arange(self.nz) * self.z / (self.nz - 1), \
                    self.z, self.z/10.)
            self.dz *= self.z / self.dz.sum()
        elif shape.upper() == 'EXP':
            self.dz = np.exp(7. * np.arange(self.nz) / (self.nz - 1))
            self.dz *= self.z / self.dz.sum()
        self.dV = self.V * self.dz / self.z

    def _gauss(self, x, xmax, sigma):
        return 1. / (np.exp((xmax/2. - x)/sigma) + 1.)

    def set_initial_conditions(self, U0):
        self.U0 = []
        assert len(U0) == 1
        assert self.species[0] in U0
        self.U0.append(U0[self.species[0]])
        for i in xrange(self.nz - 1):
            self.U0.append(0.)
        self._initialize()

    def _initialize(self):
        species = self.species[0]
        size = self.nz
        self.M = np.identity(size, dtype=int)
        self.M[-1, -1] = 0
        self.symbols = sym.symbols('x0:{}'.format(size))
        self.symbols_dict = {}
        for i in xrange(self.nz):
            self.symbols_dict[(species, i)] = self.symbols[i]

        self.f_sym = []
        self.f_exec = []
        for i in xrange(self.nz):
            Si = self.symbols_dict[(species, i)]
            f = 0
            diff = 2 * species.D
            if i == 0:
                diff /= self.dz[i]
            else:
                diff /= self.dz[i-1] + self.dz[i]
            if i > 0:
                if not (i == self.nz - 1 and species.fixed):
                    f += diff * (self.symbols_dict[(species, i-1)] \
                            - self.symbols_dict[(species, i)]) / self.dz[i-1]
            if i < self.nz - 1:
                f += diff * (self.symbols_dict[(species, i+1)] \
                        - self.symbols_dict[(species, i)]) / self.dz[i]
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
                rtol=1e-10)
        self.model.set_initial_condition(self.U0)

    def _results(self):
        self.U =[]
        for i, t in enumerate(self.t):
            Ui = {}
            for j in xrange(self.nz):
                Ui[j] = self.U1[i][j]
                j += 1
            self.U.append(Ui)
        return self.U, self.t
