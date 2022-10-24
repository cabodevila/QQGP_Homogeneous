import Evolution as ev

import shutil
import os
import time

# Define the parameters
start = time.time()

deltat = 1e-3
lenght = 200
p_max = 1
p_newspacing = 0.01

alphas = 0.1
Qs = 0.1
f0 = 0.01
init_params = [alphas, Qs, f0]

Nc = 3
Nf = 3
params = [Nc, Nf]

elastic   = True
inelastic = True

gluons     = True
quarks     = True
antiquarks = True

compute = [elastic, inelastic, gluons, quarks, antiquarks]

save = 1
plot = 1

# Check if data directory exists and ask for remove
if os.path.isdir('data'):
    inp = input("Some data are already saved. Type 'y' to remove it.\n")
    if inp == 'y':
        shutil.rmtree('data')
    else:
        exit()


# Execute the simulation

evol = ev.Evolution(compute, deltat, lenght, p_max, p_newspacing, init_params,
                    params, save=save, plot=plot)

evol.evolve(int(1e6))

ti = open('time.txt', 'w')
ti.write('Time needed: %f s' %(time.time() - start))
ti.close()

# Shutdown the computer
time.sleep(60)
os.system('shutdown now')
