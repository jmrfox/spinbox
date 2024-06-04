from quap import *

ket = ProductState(n_particles=5, isospin=True).randomize(seed=1729)
bra = ket.copy().dagger()

potential = ArgonnePotential(n_particles=5) # AV7 + Coulomb
potential.sigma.generate(scale=2.0)
potential.coulomb.generate(scale=0.1)

propagator = ProductPropagatorRBM(n_particles=5, dt=0.0001, isospin=True)
