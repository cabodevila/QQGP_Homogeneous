import numpy as np
import scipy.interpolate as scpi

import multiprocessing as mp

class Elastic_kernel():

    def __init__(self, kernel):

        # Pass all the parameters computed in the Kernel class
        self.k = kernel

        return

    def currents(self):

        """
        Compute the p^2 times the currents for gluons (p2Jg), quarks (p2Jq) and
        antiquarks (p2Jqb)
        """

        p2Jg  = -self.k.Ia * (self.k.lattice_[1:-1] * self.k.pf_deriv - self.k.pf_) \
                -self.k.Ib * self.k.pf_ * (self.k.lattice_[1:-1] + self.k.pf_)
        p2Jq  = -self.k.Ia * (self.k.lattice_[1:-1]**2 * self.k.F_deriv) \
                -self.k.Ib * self.k.pF_ * (self.k.lattice_[1:-1] - self.k.pF_)
        p2Jqb = -self.k.Ia * (self.k.lattice_[1:-1]**2 * self.k.Fb_deriv) \
                -self.k.Ib * self.k.pFb_ * (self.k.lattice_[1:-1] - self.k.pFb_)

        # Boundary conditions
        p2Jg  = np.append(p2Jg, 0)
        p2Jg  = np.insert(p2Jg, 0, 0)
        p2Jq  = np.append(p2Jq, 0)
        p2Jq  = np.insert(p2Jq, 0, 0)
        p2Jqb = np.append(p2Jqb, 0)
        p2Jqb = np.insert(p2Jqb, 0, 0)

        return 2 * np.pi * self.k.alphas**2 * self.k.Nc * p2Jg, \
               2 * np.pi * self.k.alphas**2 * self.k.CF * p2Jq, \
               2 * np.pi * self.k.alphas**2 * self.k.CF * p2Jqb #* np.log(self.k.pt2 / self.k.mD2)

    def sources(self):

        """
        Compute the p^2 times the source terms for gluons (p2Sg), quarks (p2Sq)
        and antiquarks (p2Sq)
        """

        Sq = self.k.Ic  * (self.k.pf * (1 - self.k.Function)) - \
             self.k.Icb * (self.k.Function * (self.k.lattice + self.k.pf))
        p2Sq = 2 * np.pi * self.k.alphas**2 * self.k.CF**2 * Sq #* np.log(self.k.pt2 / self.k.mD2)

        Sqb = self.k.Icb * (self.k.pf * (1 - self.k.Fbunction)) - \
              self.k.Ic  * (self.k.Fbunction * (self.k.lattice + self.k.pf))
        p2Sqb = 2 * np.pi * self.k.alphas**2 * self.k.CF**2 * Sqb #* np.log(self.k.pt2 / self.k.mD2)

        p2Sg = - (self.k.Nf / (2 * self.k.CF)) * (p2Sq + p2Sqb)

        return p2Sg, p2Sq, p2Sqb

    def kernels(self):

        p2Jg, p2Jq, p2Jqb = self.currents()
        p2Sg, p2Sq, p2Sqb = self.sources()

        Ag  = p2Jg[1:] - p2Jg[:-1]
        Aq  = p2Jq[1:] - p2Jq[:-1]
        Aqb = p2Jqb[1:] - p2Jqb[:-1]

        integ_sourceq = [p2Sq[i] * (self.k.lattice_[i+1] - self.k.lattice_[i])
                         for i in range(len(p2Sq))]
        Bq            = np.array(integ_sourceq)
        integ_sourceqb = [p2Sqb[i] * (self.k.lattice_[i+1] - self.k.lattice_[i])
                         for i in range(len(p2Sqb))]
        Bqb            = np.array(integ_sourceqb)

        Bg  = - (self.k.Nf / (2*self.k.CF)) * (Bq + Bqb)

        gluons_kernel = (-Ag + Bg) / (2 * np.pi**2)
        quarks_kernel = (-Aq + Bq) / (2 * np.pi**2)
        quarksb_kernel = (-Aqb + Bqb) / (2 * np.pi**2)

        return gluons_kernel, quarks_kernel, quarksb_kernel
