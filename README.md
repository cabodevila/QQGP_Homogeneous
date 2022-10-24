# QQGP_Homogeneous
Implements the simulation of the BEDA for a homogeneous system composed by gluons, quarks and antiquarks

All the parameters of the evolution can be controled in the main.py file. There is a few exemptions for this:

  - Initial conditions are defined in the __init__() function of [Lattice.py](Lattice.py).
  - The x grid must be controlled in the inelastic_parameters() function of [Evolution.py](Evolution.py).
