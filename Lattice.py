"""
Creates a class which defines a lattice for calculation
"""

import numpy as np

class Lattice():

    def __init__(self, length, p_max, p_newspacing, init_params, params):

        self.length = length
        self.p_max  = p_max
        self.deltap = p_max / length

        self.alphas = init_params[0]
        self.Qs     = init_params[1]
        self.f0     = init_params[2]

        self.Nc = params[0]
        self.Nf = params[1]
        self.CF = (self.Nc**2 - 1) / (2*self.Nc)

        self.lattice  = self.construct(p_newspacing)
        self.lattice_ = self.construct2()

        # Gluon distribution functions
        self.function_ = np.array([self.f0/self.alphas]*len(self.lattice_)) * \
                            np.heaviside(self.Qs - self.lattice_, 0.5)
        self.function  = np.array([self.f0/self.alphas]*(len(self.lattice_)-1)) * \
                            np.heaviside(self.Qs - self.lattice, 0.5)
        # Quark distribution function
        self.Function_ = np.array([self.f0/self.alphas]*len(self.lattice_)) * \
                            np.heaviside(self.Qs - self.lattice_, 0.5)
        self.Function  = np.array([self.f0/self.alphas]*(len(self.lattice_)-1)) * \
                            np.heaviside(self.Qs - self.lattice, 0.5)
        # Antiquark distribution function
        self.Fbunction_ = np.zeros(len(self.lattice_))
        self.Fbunction  = np.zeros(len(self.lattice))

        return

    def construct(self, p_newspacing):

        """
        Creates the lattice according to the __init__ parameters
        """

        lat1 = np.linspace(p_newspacing, self.Qs, int(self.length*self.Qs/self.p_max))
        lat2 = np.arange(self.Qs, self.p_max, self.deltap)

        if p_newspacing != 0:
            #lat_aux = np.arange(0, p_newspacing, self.deltap*0.2)
            lat_aux = np.logspace(-3.5, np.log10(p_newspacing), 10)
            return np.concatenate((lat_aux[:-1], lat1, lat2[1:]))
        else:
            return lat

    def construct2(self):

        lat_ = (self.lattice[1:] + self.lattice[:-1]) / 2

        return np.concatenate(([0], lat_, [2*lat_[-1] - lat_[-2]]))

    def number(self, i, pf, particle):

        """
        Computes the number of particles in the volume of the phase space which
        momentum is in the interval (p_lattice[i], p_lattice[i+1])
        :param i: int, index of the lattice on which we want to compute the number
        :param pf: function distribution of momentum times momentum
        :param particle: string with the name of particle (gluon or quark)
        :return: float
        """

        if particle == 'gluon':
            return 4 * self.Nc * (self.lattice_[i+1]**2 - self.lattice_[i]**2) * \
                    pf[i] * self.CF / (4 * np.pi**2)
        elif particle == 'quark':
            return 2 * self.Nc * (self.lattice_[i+1]**2 - self.lattice_[i]**2) * \
                    pf[i] * self.Nf / (4 * np.pi**2)
        else:
            print('Particle %s is not included in computation' %particle)


    def energy(self, i, pf, particle):

        """
        Computes the energy density in the volume of the phase space which momentum
        is in the interval (p_lattice[i], p_lattice[i+1])
        :param i: int, index of the lattice on which we want to compute the number
        :param pf: function distribution of momentum times momentum
        :param particle: string with the name of particle (gluon or quark)
        :return: float
        """

        if particle == 'gluon':
            return 4 * self.Nc * (self.lattice_[i+1]**3 - self.lattice_[i]**3) * \
                    pf[i] * self.CF / (6 * np.pi**2)
        elif particle == 'quark':
            return 2 * self.Nc * (self.lattice_[i+1]**3 - self.lattice_[i]**3) * \
                    pf[i] * self.Nf / (6 * np.pi**2)
        else:
            print('Particle %s is not included in computation' %particle)

    def entropy(self, i, f, particle):

        """
        Computes the entropy in the volume of the phase space which momentum is
        in the interval (p_lattice[i-1], p_lattice[i])
        Notice that in f=0, there is an indetermination in the term f*log(f),
        so we take the limit 0*log(0)=0
        :param i: int, index of the lattice on which we want to compute the number
        :param f: function distribution of momentum
        :param particle: string with the name of particle (gluon or quark)
        :return: float
        """

        if particle == 'gluon':
            if f[i] <= 0:
                return 4 * self.Nc * self.CF * (self.lattice_[i + 1] ** 3 - self.lattice_[i] ** 3) * \
                        (1 + f[i]) * np.log(1 + f[i]) / (6 * np.pi ** 2)
            else:
                return 4 * self.Nc * self.CF * (self.lattice_[i + 1] ** 3 - self.lattice_[i] ** 3) * (
                        (1 + f[i]) * np.log(1 + f[i]) - f[i] * np.log(f[i])) / \
                        (6 * np.pi ** 2)

        if particle == 'quark':
            if f[i] <= 0:
                return -2 * self.Nc * self.Nf * (self.lattice_[i + 1] ** 3 - self.lattice_[i] ** 3) * \
                        (1 - f[i]) * np.log(1 - f[i]) / (6 * np.pi ** 2)
            else:
                return -2 * self.Nc * self.Nf * (self.lattice_[i + 1] ** 3 - self.lattice_[i] ** 3) * (
                        (1 - f[i]) * np.log(1 - f[i]) + f[i] * np.log(f[i])) / \
                        (6 * np.pi ** 2)

        else:
            print('Particle %s is not included in computation' %particle)

    def save_lattice(self):

        """
        Save the lattice in a .txt file
        :return:
        """

        np.savetxt('data/lattice.txt', self.lattice)
        np.savetxt('data/lattice_.txt', self.lattice_)
