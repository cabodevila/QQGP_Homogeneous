import numpy as np
import scipy.interpolate as scpi
import matplotlib.pyplot as plt

import multiprocessing as mp

class Inelastic_kernel():

    def __init__(self, kernel, inelastic):

        self.k = kernel

        self.lattice_aux = np.append(self.k.lattice, 2 * self.k.lattice[-1] - self.k.lattice[-2])
        self.function_aux = np.append(self.k.pf, 0)
        self.Function_aux = np.append(self.k.Function, 0)
        self.Fbunction_aux = np.append(self.k.Fbunction, 0)
        self.extrf  = scpi.InterpolatedUnivariateSpline(self.lattice_aux,
                                                        self.function_aux, ext=3)
        self.extrF  = scpi.InterpolatedUnivariateSpline(self.lattice_aux,
                                                        self.Function_aux, ext=3)
        self.extrFb = scpi.InterpolatedUnivariateSpline(self.lattice_aux,
                                                        self.Fbunction_aux, ext=3)
        self.pf  = self.extrf(self.k.lattice_)
        self.pF  = self.extrF(self.k.lattice_)
        self.pFb = self.extrFb(self.k.lattice_)

        # Compute a x grid distributed as the splitting function
        self.x = inelastic[0]
        self.factors = inelastic[1]
        self.spltr = self.split_rates()

        return

    def split_rates(self):

        """
        Computes the (LPM) splitting rates for the following processes:
            - g <-> gg
            - g <-> qq_bar
            - q <-> gq
        times sqrt(p)
        """

        # g <-> gg
        a0 = (self.k.alphas * self.k.Nc / np.pi) * np.sqrt(self.k.qhat)
        a  = (1-self.x+self.x**2)**(5/2) / (self.x-self.x**2)**(3/2)
        Ig_gg = a * a0

        # g <-> qq_bar
        c0 = self.k.alphas * np.sqrt(self.k.qhat) / (4 * np.pi)
        c  = np.sqrt((self.k.CF/self.k.Nc) - (self.x - self.x**2)) \
            * (self.x**2 + (1 - self.x)**2) / np.sqrt(self.x - self.x**2)
        Ig_qq = c * c0

        # q <-> gq
        b0 = (self.k.alphas * self.k.CF) * np.sqrt(self.k.qhat) / (2 * np.pi)
        b1 = np.sqrt((1-self.x) + (self.k.CF/self.k.Nc) * self.x**2) \
             * (1 + (1-self.x)**2) / (self.x * np.sqrt(self.x - self.x**2))
        Iq_gq = b1 * b0

        # q <-> qg (change x <-> (1-x))
        b2 = np.sqrt((self.x) + (self.k.CF/self.k.Nc) * (1-self.x)**2) \
             * (1 + self.x**2) / ((1-self.x) * np.sqrt(self.x - self.x**2))
        Iq_qg = b2 * b0

        return [Ig_gg, Ig_qq, Iq_gq, Iq_qg]

    def new_grid(self, p):

        self.new_f  = []
        self.new_F  = []
        self.new_Fb = []

        for i, factor in enumerate(self.factors):
            new_grid = factor * p
            self.new_f.append(self.extrf(new_grid) / factor)
            self.new_F.append(self.extrF(new_grid))
            self.new_Fb.append(self.extrFb(new_grid))

        return

    def statistical_factors(self, i, p):

        """
        Compute the statistical factors for each posible splittings
        """

        self.Fggg_a = (self.new_f[0] * (p + self.pf[i]) * (p + self.new_f[1]) -
                       self.pf[i] * self.new_f[1] * (p + self.new_f[0]))
        self.Fggg_b = (self.pf[i] * (p + self.new_f[2]) * (p + self.new_f[3]) -
                       self.new_f[2] * self.new_f[3] * (p + self.pf[i]))

        self.Fgqqb_a = (self.new_f[0] * (1 - self.pF[i]) * (1 - self.new_Fb[1]) -
                        self.pF[i] * self.new_Fb[1] * (p + self.new_f[0])) * p**2
        self.Fgqqb_b = (self.pf[i] * (1 - self.new_F[2]) * (1 - self.new_Fb[3]) -
                        self.new_F[2] * self.new_Fb[3] * (p + self.pf[i])) * p**2

        self.Fgqbq_a = (self.new_f[0] * (1 - self.pFb[i]) * (1 - self.new_F[1]) -
                        self.pFb[i] * self.new_F[1] * (p + self.new_f[0])) * p**2
        self.Fgqbq_b = (self.pf[i] * (1 - self.new_Fb[2]) * (1 - self.new_F[3]) -
                        self.new_Fb[2] * self.new_F[3] * (p + self.pf[i])) * p**2

        self.Fqgq_a = (self.new_F[0] * (p + self.pf[i]) * (1 - self.new_F[1]) -
                       self.pf[i] * self.new_F[1] * (1 - self.new_F[0])) * p**2
        self.Fqgq_b = (self.pF[i] * (p + self.new_f[2]) * (1 - self.new_F[3]) -
                       self.new_f[2] * self.new_F[3] * (1 - self.pF[i])) * p**2

        self.Fqqg_a = (self.new_F[0] * (1 - self.pF[i]) * (p + self.new_f[1]) -
                       self.pF[i] * self.new_f[1] * (1 - self.new_F[0])) * p**2
        self.Fqqg_b = (self.pF[i] * (1 - self.new_F[2]) * (p + self.new_f[3]) -
                       self.new_F[2] * self.new_f[3] * (1 - self.pF[i])) * p**2

        self.Fqbgqb_a = (self.new_Fb[0] * (p + self.pf[i]) * (1 - self.new_Fb[1]) -
                         self.pf[i] * self.new_Fb[1] * (1 - self.new_Fb[0])) * p**2
        self.Fqbgqb_b = (self.pFb[i] * (p + self.new_f[2]) * (1 - self.new_Fb[3]) -
                         self.new_f[2] * self.new_Fb[3] * (1 - self.pFb[i])) * p**2

        self.Fqbqbg_a = (self.new_Fb[0] * (1 - self.pFb[i]) * (p + self.new_f[1]) -
                         self.pFb[i] * self.new_f[1] * (1 - self.new_Fb[0])) * p**2
        self.Fqbqbg_b = (self.pFb[i] * (1 - self.new_Fb[2]) * (p + self.new_f[3]) -
                         self.new_Fb[2] * self.new_f[3] * (1 - self.pFb[i])) * p**2

        return

    def integrand(self, i, p):

        self.new_grid(p)
        self.statistical_factors(i, p)

        integrand_gluon = 0
        integrand_quark = 0
        integrand_quarkb = 0

        integrand_gluon += self.spltr[0] * (self.Fggg_a/self.x**(5/2) - 0.5*self.Fggg_b)
        integrand_gluon -= self.spltr[1] * self.k.Nf * self.Fgqqb_b
        integrand_gluon += self.spltr[2] * (self.k.Nf/(2*self.k.CF)) * (self.Fqgq_a + self.Fqbgqb_a) / self.x**(5/2)

        #integrand_quark += self.spltr[1] * 2 * self.k.CF * self.Fgqqb_a / self.x**(5/2)
        integrand_quark -= self.spltr[3] * self.Fqqg_b
        integrand_quark += self.spltr[3] * (self.Fqqg_a/self.x**(5/2))# - 0.5 * self.Fqqg_b)

        #integrand_quarkb += self.spltr[1] * 2 * self.k.CF * self.Fgqqb_a / self.x**(5/2)
        integrand_quarkb -= self.spltr[3] * self.Fqbqbg_b
        integrand_quarkb += self.spltr[3] * (self.Fqbqbg_a/self.x**(5/2))# - 0.5 * self.Fqbqbg_b)
        """

        integrand_gluon = self.spltr[0] * (self.Fggg_a/self.x**(5/2) - 0.5*self.Fggg_b) \
                        - self.spltr[1] * self.k.Nf * self.Fgqqb_b \
                        + self.spltr[2] * (self.k.Nf/(2*self.k.CF)) * (self.Fqgq_a + self.Fqbgqb_a) / self.x**(5/2)

        integrand_quark = self.spltr[1] * 2 * self.k.CF * self.Fgqqb_a / self.x**(5/2) \
                        - self.spltr[3] * self.Fqqg_b \
                        + self.spltr[3] * (self.Fqqg_a/self.x**(5/2))# - 0.5 * self.Fqqg_b)

        integrand_quarkb = self.spltr[1] * 2 * self.k.CF * self.Fgqbq_a / self.x**(5/2) \
                         - self.spltr[3] * self.Fqbqbg_b \
                         + self.spltr[3] * (self.Fqbqbg_a/self.x**(5/2))# - 0.5 * self.Fqbqbg_b)
        """

        integral_gluon  = np.trapz(integrand_gluon, self.x)
        integral_quark  = np.trapz(integrand_quark, self.x)
        integral_quarkb = np.trapz(integrand_quarkb, self.x)

        return integral_gluon, integral_quark, integral_quarkb

    def kernels(self):

        lattice_aux = np.copy(self.k.lattice_)
        lattice_aux[0] = self.k.lattice[0]
        # Execute the integration in x in parallel
        # Using Pool
        lista = [[i, p] for i, p in enumerate(lattice_aux)]
        with mp.Pool(8) as pool:
            integrand = pool.starmap(self.integrand, lista)

        integrand = np.array(integrand)
        integrand_gluons  = integrand[:,0]
        integrand_quarks  = integrand[:,1]
        integrand_quarksb = integrand[:,2]
        integrand_gluons  = integrand_gluons * lattice_aux**(-3/2)
        integrand_quarks  = integrand_quarks * lattice_aux**(-3/2)
        integrand_quarksb = integrand_quarksb * lattice_aux**(-3/2)

        # Integrate in momentum in order to compute the derivative
        derivative_gluons = np.array(
            [np.trapz(integrand_gluons[i:i+2], x=self.k.lattice_[i:i+2])
            for i in range(1, len(self.k.lattice))]
            )
        derivative_quarks = np.array(
            [np.trapz(integrand_quarks[i:i+2], x=self.k.lattice_[i:i+2])
            for i in range(1, len(self.k.lattice))]
            )
        derivative_quarksb = np.array(
            [np.trapz(integrand_quarksb[i:i+2], x=self.k.lattice_[i:i+2])
            for i in range(1, len(self.k.lattice))]
            )

        derivative_gluons = np.insert(derivative_gluons,
                                      0,
                                      integrand_gluons[1] * self.k.lattice_[1])
        derivative_quarks = np.insert(derivative_quarks,
                                      0,
                                      integrand_quarks[1] * self.k.lattice_[1])
        derivative_quarksb = np.insert(derivative_quarksb,
                                      0,
                                      integrand_quarksb[1] * self.k.lattice_[1])

        derivative_gluons  /= (2 * np.pi**2)
        derivative_quarks  /= (2 * np.pi**2)
        derivative_quarksb /= (2 * np.pi**2)

        print(sum(derivative_quarks))
        print(sum(derivative_quarksb))
        print(sum(derivative_quarks - derivative_quarksb))


        return derivative_gluons, derivative_quarks, derivative_quarksb
