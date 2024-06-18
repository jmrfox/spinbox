from spinbox import *

n_particles = 2



# Asig = SigmaCoupling(2, file="./data/h2/fort.7701" )
# print(Asig)
pot = ArgonnePotential(n_particles)
pot.read_sigma("./data/h2/fort.7701")

