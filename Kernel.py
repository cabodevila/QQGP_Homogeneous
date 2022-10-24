"""
Computes the 2 elastic collision kernels for a system of quarks and gluons
"""

import numpy as np
import matplotlib.pyplot as plt

class Kernel():

    def __init__(self, kern):

        self.lattice  = kern.lattice
        self.lattice_ = kern.lattice_

        self.function  = kern.function
        self.Function  = kern.Function
        self.Fbunction = kern.Fbunction

        self.pf  = kern.pf
        self.pF  = kern.pF
        self.pFb = kern.pFb

        deltap = (kern.lattice[1:] - kern.lattice[:-1])
        self.pf_deriv  = (self.pf[1:] - self.pf[:-1]) / deltap
        self.pF_deriv  = (self.pF[1:] - self.pF[:-1]) / deltap
        self.pFb_deriv = (self.pFb[1:] - self.pFb[:-1]) / deltap

        self.F_deriv = (self.Function[1:] - self.Function[:-1]) / deltap
        self.Fb_deriv = (self.Fbunction[1:] - self.Fbunction[:-1]) / deltap

        self.alphas = kern.alphas
        self.Qs     = kern.Qs
        self.f0     = kern.f0

        self.Nc = kern.Nc
        self.Nf = kern.Nf
        self.CF = (self.Nc**2 - 1) / (2*self.Nc)

        self.compute_integrals()
        #self.compute_pt2()

        self.function_  = np.interp(self.lattice_[1:-1], self.lattice, self.function)
        self.Function_  = np.interp(self.lattice_[1:-1], self.lattice, self.Function)
        self.Fbunction_ = np.interp(self.lattice_[1:-1], self.lattice, self.Function)
        self.pf_  = np.interp(self.lattice_[1:-1], self.lattice, self.pf)
        self.pF_  = np.interp(self.lattice_[1:-1], self.lattice, self.pF)
        self.pFb_ = np.interp(self.lattice_[1:-1], self.lattice, self.pFb)

        self.parameters()

        return

    def compute_integrals(self):

        Ia_integrand = (self.Nc * self.pf * (self.lattice + self.pf) +
                        0.5 * self.Nf * self.pF * (self.lattice - self.pF) +
                        0.5 * self.Nf * self.pFb * (self.lattice - self.pFb))
        self.Ia = np.trapz(Ia_integrand, x=self.lattice) / (2*np.pi**2)
        print(self.Ia)

        Ib_integrand = 2 * (self.Nc * self.pf +
                            0.5 * self.Nf * self.pF +
                            0.5 * self.Nf * self.pFb)
        self.Ib = np.trapz(Ib_integrand, x=self.lattice) / (2*np.pi**2)

        Ic_integrand  = self.pf + self.pF \
                      + self.pf*(self.Function - self.Fbunction)
        Icb_integrand = self.pf + self.pFb \
                      + self.pf*(self.Fbunction - self.Function)
        self.Ic  = np.trapz(Ic_integrand, x=self.lattice) / (2*np.pi**2)
        self.Icb = np.trapz(Icb_integrand, x=self.lattice) / (2*np.pi**2)

        self.integrals = [self.Ia, self.Ib, self.Ic, self.Icb]

        return

    def compute_pt2(self):

        """
        Compute the average value of the transverse momentum squared
        """

        integrand = (self.Nc * self.pf +
                     0.5 * self.Nf * self.pF +
                     0.5 * self.Nf * self.pFb)
        p_average = np.trapz(self.lattice**3 * integrand, self.lattice) / (2*np.pi**2)
        #print(p_average)

        self.pt2 = p_average * 2 / 3

        return

    def parameters(self):

        """ Compute the current values for the jet quenching parameter and the
            Debye screening mass
        """

        self.mD2  = 8 * np.pi * self.alphas * self.Ib
        self.qhat = 8 * np.pi * self.alphas**2 * self.Nc * self.Ia \
                    #* np.log(self.pt2 / self.mD2)

        return
