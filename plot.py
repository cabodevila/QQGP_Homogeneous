import numpy as np
import matplotlib.pyplot as plt

import os
import re

os.makedirs('figures/function_gluon', exist_ok=True)
os.makedirs('figures/function_quark', exist_ok=True)
os.makedirs('figures/function_quarkb', exist_ok=True)

def plot_fun(lattice, steps):

    for n in ['gluon', 'quark', 'quarkb']:
        fun_arx = sorted(os.listdir('data/function_' + n),
                         key=lambda x: int(re.search('iteration_(.*).txt', x).group(1)))
        for i, name in enumerate(fun_arx):
            if i % 10 == 0:
                plt.clf()
                plt.plot(lattice, lattice * np.loadtxt('data/function_' + n + '/' + name)[:,1],
                         label='distribution')
                plt.xscale('log')
                plt.title(name)
                plt.legend()
                plt.grid()
                plt.savefig('figures/function_' + n + '/fun_' + name[:-4] + '.png')

def plot_integrals(data, time_step, steps, data_save):

    Ia = data[:,0]
    Ib = data[:,1]
    Ic = data[:,2]
    Icb = data[:,3]
    T = Ia / Ib

    time = np.loadtxt('data/iterations.txt')
    x_axis = time_step * time


    T_thermal = 0.0216516 # It depends on the initial conditions and simulation parameters

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(x_axis, Ia, label=r'$I_a$')
    plt.plot(x_axis, Ib, label=r'$I_b$')
    plt.plot(x_axis, Ic, label=r'$I_c$')
    plt.plot(x_axis, Icb, label=r'$\bar{I_c}$')
    plt.plot(x_axis, T, label=r'$T_*$')
    plt.hlines(T_thermal, x_axis[0], x_axis[-1], 'b', '--', label=r'$T_{th}$')

    plt.grid()
    plt.legend(loc='lower left')
    plt.savefig('figures/integrals.png')

    return

def plot_stats(Number, Energy, Entropy, time_step, steps, data_save):

    num = Number[:,0]
    ene = Energy[:,0]
    ent = Entropy[:,0]

    Num = Number[:,1]
    Ene = Energy[:,1]
    Ent = Entropy[:,1]

    Numb = Number[:,2]
    Eneb = Energy[:,2]
    Entb = Entropy[:,2]

    time = np.loadtxt('data/iterations.txt')
    x_axis = time_step * time

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(x_axis, num, 'r-', label='Gluon number')
    plt.plot(x_axis, ene, 'r--', label='Gluon energy')
    plt.plot(x_axis, ent, 'r-.', label='Gluon entropy')

    plt.plot(x_axis, Num, 'g-', label='Quark number')
    plt.plot(x_axis, Ene, 'g--', label='Quark energy')
    plt.plot(x_axis, Ent, 'g-.', label='Quark entropy')

    plt.plot(x_axis, Numb, 'b-', label='Antiquark number')
    plt.plot(x_axis, Eneb, 'b--', label='Antiquark energy')
    plt.plot(x_axis, Entb, 'b-.', label='Antiquark entropy')

    plt.grid()
    plt.legend()
    plt.savefig('figures/stats.png')

    return

def plot_stats_total(Number, Energy, Entropy, time_step, steps, data_save):

    num = Number[:,3]
    ene = Energy[:,3]
    ent = Entropy[:,3]

    time = np.loadtxt('data/iterations.txt')
    x_axis = time_step * time

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(x_axis, num, 'r-', label='Number')
    plt.plot(x_axis, ene, 'r--', label='Energy')
    plt.plot(x_axis, ent, 'r-.', label='Entropy')

    # Equilibrium values
    #f0 = 0.01; alphas = 0.1; Qs = 0.1; Nf = 3
    T_eq = 0.0193881
    #n_eq = (1.20206 / np.pi**2) * (16 + 9*Nf) * T_eq**3
    #s_eq = (np.pi**2 / 15) * (32/3 + 7*Nf) * T_eq**3

    #plt.hlines(n_eq, x_axis[0], x_axis[-1], colors='k', linestyles='dotted')
    #plt.hlines(s_eq, x_axis[0], x_axis[-1], colors='k', linestyles='dotted')

    plt.grid()
    plt.legend()
    plt.savefig('figures/stats_global.png')

    return

def plot_energy_total(Energy, time_step, steps, data_save):

    ene = Energy[:,3]

    time = np.loadtxt('data/iterations.txt')
    x_axis = time_step * time

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(x_axis, ene, 'r--', label='Energy')

    plt.grid()
    plt.legend()
    plt.savefig('figures/energy_global.png')

    return

def plot_number_total(Number, time_step, steps, data_save):

    num  = Number[:,0]
    Num  = Number[:,1]
    Numb = Number[:,2]
    NUM  = Number[:,3]

    time = np.loadtxt('data/iterations.txt')
    x_axis = time_step * time

    n_thermal = 0.0000642772

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(x_axis, num+Num+Numb, 'r--', label='Number')
    plt.hlines(n_thermal, x_axis[0], x_axis[-1], colors='k', linestyles='dotted')

    plt.grid()
    plt.legend()
    plt.savefig('figures/number_global.png')

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(x_axis, Num - Numb, 'r--', label='Baryon Number')
    plt.grid()
    plt.legend()
    plt.savefig('figures/BaryonNumber.png')

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(x_axis, num, 'r--', label='Gluon Number')
    plt.plot(x_axis, Num, 'g--', label='Quark Number')
    plt.plot(x_axis, Numb, 'b--', label='Antiquark Number')

    plt.hlines(16*1.23622e-6, x_axis[0], x_axis[-1], colors='r', linestyles='dotted')
    plt.hlines(18*2.08039e-6, x_axis[0], x_axis[-1], colors='g', linestyles='dotted')
    plt.hlines(18*3.91703e-7, x_axis[0], x_axis[-1], colors='b', linestyles='dotted')
    plt.grid()
    plt.legend()
    plt.savefig('figures/Numbers.png')
    #plt.show()

    return

def plot_entropy_total(Entropy, time_step, steps, data_save):

    ent = Entropy[:,0]
    Ent = Entropy[:,1]
    Entb = Entropy[:,2]
    ENT = Entropy[:,3]

    time = np.loadtxt('data/iterations.txt')
    x_axis = time_step * time

    ent_thermal = 0.00023723

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(x_axis, ENT, 'r--', label='Entropy')
    plt.hlines(ent_thermal, x_axis[0], x_axis[-1], colors='k', linestyles='dotted')

    plt.grid()
    plt.legend()
    plt.savefig('figures/entropy_global.png')
    #plt.show()

    return



lattice = np.loadtxt('data/lattice.txt')
integrals = np.loadtxt('data/integrals.txt')
Number = np.loadtxt('data/Number.txt')
Energy = np.loadtxt('data/Energy.txt')
Entropy = np.loadtxt('data/Entropy.txt')


time_step = 5e-3
steps = int(1e6)
data_save = 200


plot_integrals(integrals, time_step, steps, data_save)
plot_stats(Number, Energy, Entropy, time_step, steps, data_save)
plot_stats_total(Number, Energy, Entropy, time_step, steps, data_save)
plot_energy_total(Energy, time_step, steps, data_save)
plot_number_total(Number, time_step, steps, data_save)
plot_entropy_total(Entropy, time_step, steps, data_save)
#plot_fun(lattice, steps)
