from quap import *


n_particles = 2
Asig = SigmaCoupling(2, np.zeros(shape=(3,n_particles,3,n_particles)))
Asig.read("./data/h2/fort.7701")
print(Asig)
x = Asig.copy()
print(x)
