from spinbox import *

n_particles = 3
isospin = False
seed = 2
seeds = np.arange(3)
maxiter = 100

print("======== PRODUCT =======")
# test for pure product state
ket = ProductState(n_particles, isospin=isospin).random(seed)
fit = ket.to_full_basis().nearest_product_state(seeds=seeds, maxiter=maxiter)
print("product state:", ket)
print("fit: ", fit)
print("overlap:", ket.dagger() * fit )
print("entanglement = ", 1-abs(fit.dagger() * ket))
print("VN entropy = ", ket.to_full_basis().entropy())


print("======== ENTANGLED =======")
# test for entangled state
ket = HilbertState(n_particles, isospin=isospin).random(seed)
fit = ket.nearest_product_state(seeds=seeds, maxiter=maxiter).to_full_basis()
print("hilbert state:", ket)
print("nearest product state:", fit)
print("overlap:", ket.dagger() * fit )
print("entanglement = ", 1-abs(fit.dagger() * ket))
print("VN entropy = ", ket.entropy())