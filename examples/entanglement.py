from spinbox import *

n_particles = 3
isospin = False
seed = 2
seeds = np.arange(3)
maxiter = 100

print("======== PRODUCT =======")
# test for pure product state
ket = ProductState(n_particles, isospin=isospin).randomize(seed)
fit = ket.to_manybody_basis().nearest_product_state(seeds=seeds, maxiter=maxiter)
print("product state:", ket)
print("fit: ", fit)
print("overlap:", ket.dagger() * fit )
print("entanglement = ", 1-abs(fit.dagger() * ket))

print("======== ENTANGLED =======")
# test for entangled state
ket = HilbertState(n_particles, isospin=isospin).randomize(seed)
fit = ket.nearest_product_state(seeds=seeds, maxiter=maxiter)
print("hilbert state:", ket)
print("fit:", fit.to_manybody_basis())
print("overlap:", ket.dagger() * fit )
print("entanglement = ", 1-abs(fit.to_manybody_basis().dagger() * ket))
print("fit product state:", fit)