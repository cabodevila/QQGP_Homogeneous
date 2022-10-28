"""
Defines a class to evolve the system
"""

import numpy as np
import matplotlib.pyplot as plt
import Kernel

from Elastic import Elastic_kernel as ek
from Inelastic import Inelastic_kernel as ik

import os

from Lattice import Lattice

class Evolution(Lattice):

    def __init__(self, compute, deltat, length, p_max, p_newspacing=0,
                 init_params=[0.1, 0.1, 0.01],
                 params=[3, 3], save=1000, plot=1000):

        super().__init__(length, p_max, p_newspacing, init_params, params)

        self.compute = compute
        self.deltat  = deltat
        self.save    = save   # Save the data each 'save' iterations
        self.plot    = plot   # Plot the data each 'plot' iterations

        self.pf   = self.lattice  * self.function   # gluon
        self.pf_  = self.lattice_ * self.function_  # gluon
        self.pF   = self.lattice  * self.Function   # quark
        self.pF_  = self.lattice_ * self.Function_  # quark
        self.pFb  = self.lattice  * self.Fbunction  # antiquark
        self.pFb_ = self.lattice_ * self.Fbunction_ # antiquark

        # Save the lattice in a text file
        os.mkdir('data')
        self.save_lattice()

        # Get necesary values for the inelastic kernel computation
        self.inelastic_parameters()

        return

    def inelastic_parameters(self):

        """
        Compute some constant parameters needed for the inelastic kernel
        """

        x_ = np.logspace(-5, np.log10(0.5), 2000)
        self.x = np.append(x_, np.flip(1-x_))
        self.factors = [1/self.x, (1-self.x)/self.x, self.x, 1-self.x]

        return

    def get_kernels(self):

        """
        Get the values for each kernel in the current time step
        """

        ker = Kernel.Kernel(self)
        self.integrals = ker.integrals

        gluon_kernel  = 0
        quark_kernel  = 0
        quarkb_kernel = 0

        if self.compute[0] == True:

            elastic_kernel = ek(ker)
            self.ek_gluon, self.ek_quark, self.ek_quarkb = elastic_kernel.kernels()

            gluon_kernel  += self.ek_gluon
            quark_kernel  += self.ek_quark
            quarkb_kernel += self.ek_quarkb

        if self.compute[1] == True:

            inelastic_kernel = ik(ker, [self.x, self.factors])
            self.ik_gluon, self.ik_quark, self.ik_quarkb = inelastic_kernel.kernels()

            gluon_kernel  += self.ik_gluon
            quark_kernel  += self.ik_quark
            quarkb_kernel += self.ik_quarkb


        return gluon_kernel, quark_kernel, quarkb_kernel

    def next_step(self):

        """
        Evolve the system to the next step
        """

        volume = 0.5 * (self.lattice_[1:]**2 - self.lattice_[:-1]**2) / (2 * np.pi**2)

        self.pf = self.pf + self.deltat * self.gluon_kernel / volume
        self.pF = self.pF + self.deltat * self.quark_kernel / volume
        self.pFb = self.pFb + self.deltat * self.quarkb_kernel / volume
        self.function = self.pf / self.lattice
        self.Function = self.pF / self.lattice
        self.Fbunction = self.pFb / self.lattice

        return

    def evolve(self, steps):

        """
        Performs the numerical evolution of the system
        """

        if self.plot != False:
            self.fig, self.axs = plt.subplots(1, 3, figsize=(21,7))

        for i in range(steps):

            self.gluon_kernel, self.quark_kernel, self.quarkb_kernel = self.get_kernels()

            if self.save != False and i % self.save == 0:
                self.save_results(i)
                print('========== Iteration %i ===========' %i)

                if self.plot != False and i % self.plot == 0 and i != 0:
                    self.plot_results(i)

            self.next_step()

        return

    def save_results(self, iter):

        """
        Saves the data of the current step in different text files
        :param iter: current iteration of the evolution
        :param additional: additional parameters to save. Must be False or a
                           [Ia, Ib, T_star, Jp, deriv] list
        :return:
        """

        number  = np.array([self.number(i, self.pf, 'gluon')
                           for i in range(len(self.pf))])
        Number  = np.array([self.number(i, self.pF, 'quark')
                           for i in range(len(self.pF))])
        Numberb = np.array([self.number(i, self.pFb, 'quark')
                           for i in range(len(self.pFb))])

        os.makedirs('data/function_gluon', exist_ok=True)
        os.makedirs('data/function_quark', exist_ok=True)
        os.makedirs('data/function_quarkb', exist_ok=True)
        os.makedirs('data/number_gluon', exist_ok=True)
        os.makedirs('data/number_quark', exist_ok=True)
        os.makedirs('data/number_quarkb', exist_ok=True)
        os.makedirs('data/kern_gluon', exist_ok=True)
        os.makedirs('data/kern_quark', exist_ok=True)
        os.makedirs('data/kern_quarkb', exist_ok=True)

        np.savetxt('data/function_gluon/iteration_%i.txt' %iter,
                    np.array([self.lattice, self.function]).T)
        np.savetxt('data/function_quark/iteration_%i.txt' %iter,
                    np.array([self.lattice, self.Function]).T)
        np.savetxt('data/function_quarkb/iteration_%i.txt' %iter,
                    np.array([self.lattice, self.Fbunction]).T)
        np.savetxt('data/number_gluon/iteration_%i.txt' %iter,
                    np.array([self.lattice, number]).T)
        np.savetxt('data/number_quark/iteration_%i.txt' %iter,
                    np.array([self.lattice, Number]).T)
        np.savetxt('data/number_quarkb/iteration_%i.txt' %iter,
                    np.array([self.lattice, Numberb]).T)

        total_number  = sum(number)
        total_energy  = sum([self.energy(i, self.pf, 'gluon')
                             for i in range(len(self.pf)-1)])
        total_entropy = sum([self.entropy(i, self.function, 'gluon')
                             for i in range(len(self.function)-1)])

        total_Number  = sum(Number)
        total_Energy  = sum([self.energy(i, self.pF, 'quark')
                             for i in range(len(self.pF)-1)])
        total_Entropy = sum([self.entropy(i, self.Function, 'quark')
                             for i in range(len(self.Function)-1)])

        total_Numberb  = sum(Numberb)
        total_Energyb  = sum([self.energy(i, self.pFb, 'quark')
                             for i in range(len(self.pFb)-1)])
        total_Entropyb = sum([self.entropy(i, self.Fbunction, 'quark')
                             for i in range(len(self.Fbunction)-1)])

        integrals = open('data/integrals.txt', 'a')
        Number    = open('data/Number.txt', 'a')
        Energy    = open('data/Energy.txt', 'a')
        Entropy   = open('data/Entropy.txt', 'a')
        time      = open('data/iterations.txt', 'a')

        integrals.write('%.16e %.16e %.16e %.16e %.16e %.16e\n' %(*self.integrals,
                                        self.integrals[0]/self.integrals[1],
                                        iter))
        Number.write('%.16e %.16e %.16e %.16e %.16e\n'
                     %(total_number, total_Number, total_Numberb,
                       total_number + total_Number + total_Numberb,
                       iter))
        Energy.write('%.16e %.16e %.16e %.16e %.16e\n'
                     %(total_energy, total_Energy, total_Energyb,
                       total_energy + total_Energy + total_Energyb,
                       iter))
        Entropy.write('%.16e %.16e %.16e %.16e %.16e\n'
                      %(total_entropy, total_Entropy, total_Entropyb,
                        total_entropy + total_Entropy + total_Entropyb,
                        iter))
        time.write('%i\n' %iter)

        integrals.close()
        Number.close()
        Energy.close()
        Entropy.close()
        time.close()

        np.savetxt('data/kern_gluon/iteration_%i.txt' %iter, self.gluon_kernel)
        np.savetxt('data/kern_quark/iteration_%i.txt' %iter, self.quark_kernel)
        np.savetxt('data/kern_quark/iteration_%i.txt' %iter, self.quarkb_kernel)

        return

    def plot_results(self, iter):

        self.fig.suptitle('Iteration: %i     Gluons: red     Quarks: green     Antiquarks: blue  ' %iter, fontsize=16)

        self.axs[0].clear()
        self.axs[1].clear()
        self.axs[2].clear()

        self.axs[0].plot(self.lattice, self.pf, 'r.-', label='Gluons')
        self.axs[0].plot(self.lattice, self.pF, 'g.-', label='Quarks')
        self.axs[0].plot(self.lattice, self.pFb, 'b.-', label='Antiquarks')

        self.axs[0].set_xscale('log')
        self.axs[0].grid()
        self.axs[0].legend()

        if self.compute[0] == True:
            self.axs[1].plot(self.lattice, self.ek_gluon, label='ek_gluon')
            self.axs[1].plot(self.lattice, self.ek_quark, label='ek_quark')
            self.axs[1].plot(self.lattice, self.ek_quarkb, label='ek_quarkb')
        if self.compute[1] == True:
            self.axs[1].plot(self.lattice, self.ik_gluon, label='ik_gluon')
            self.axs[1].plot(self.lattice, self.ik_quark, label='ik_quark')
            self.axs[1].plot(self.lattice, self.ik_quarkb, label='ik_quarkb')

        self.axs[1].set_xscale('log')
        self.axs[1].grid()
        self.axs[1].legend()
        #self.axs[1].set_ylim((-0.2e-9, 0.2e-9))


        time = np.loadtxt('data/iterations.txt')
        Number = np.loadtxt('data/Number.txt')
        Energy = np.loadtxt('data/Energy.txt')
        Entropy = np.loadtxt('data/Entropy.txt')
        """
        self.axs[2].plot(time, Number[:,0], 'r-', label='Gluon number')
        self.axs[2].plot(time, Energy[:,0], 'r--', label='Gluon energy')
        self.axs[2].plot(time, Entropy[:,0], 'r-.', label='Gluon entropy')

        self.axs[2].plot(time, Number[:,1], 'g-', label='Quark number')
        self.axs[2].plot(time, Energy[:,1], 'g--', label='Quark energy')
        self.axs[2].plot(time, Entropy[:,1], 'g-.', label='Quark entropy')

        self.axs[2].plot(time, Number[:,2], 'b-', label='Antiquark number')
        self.axs[2].plot(time, Energy[:,2], 'b--', label='Antiquark energy')
        self.axs[2].plot(time, Entropy[:,2], 'b-.', label='Antiquark entropy')
        """
        #self.axs[2].plot(time, Number[:,3], 'k-', label='Total number')
        self.axs[2].plot(time, Energy[:,3], 'k--', label='Total energy')
        #self.axs[2].plot(time, Entropy[:,3], 'k-.', label='Total entropy')

        # Equilibrium values
        T_eq = ((60 * self.f0/self.alphas) / (16 + 0.5*21*self.Nf))**(1/4) * (self.Qs / np.pi)
        n_eq = (1.20206 / np.pi**2) * (16 + 9*self.Nf) * T_eq**3
        s_eq = (np.pi**2 / 15) * (32/3 + 7*self.Nf) * T_eq**3

        #self.axs[2].hlines(n_eq, time[0], time[-1], colors='k', linestyles='dotted')
        #self.axs[2].hlines(s_eq, time[0], time[-1], colors='k', linestyles='dotted')

        self.axs[2].grid()
        self.axs[2].legend(loc='upper left')
        #plt.legend(loc='upper center', title='Iteration: %i\nBlue: gluons\nRed:quarks' %iter)
        plt.pause(0.1)
        #plt.show()

        return
