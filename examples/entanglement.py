from spinbox import *

n_particles = 3
isospin=True
seed = 2
seeds = np.arange(3)
maxiter = 100


# print("======== PRODUCT =======")
# # test for pure product state
# ket = ProductState(n_particles, isospin=isospin).randomize(seed)

# print(ket.coefficients)

# fit = ket.to_manybody_basis().nearest_product_state(seeds=seeds, maxiter=maxiter)
# print("product state:", ket)
# print("fit: ", fit)
# print("overlap:", ket.dagger().inner(fit))
# print("entanglement = ", 1-abs(fit.dagger().inner(ket)))

# print("======== ENTANGLED =======")
# # test for entangled state
# ket = HilbertState(n_particles, isospin=isospin).randomize(seed)

# print(ket.coefficients)

# fit = ket.nearest_product_state(seeds=seeds, maxiter=maxiter)
# print("hilbert state:", ket)
# print("fit:", fit.to_manybody_basis())
# print("entanglement = ", 1-abs(fit.to_manybody_basis().dagger().inner(ket)))
# print("fit product state:", fit)


SAFE = False

print("======== PRODUCT =======")
# test for pure product state
ket = ProductState(n_particles, isospin=isospin).randomize(seed)

print(ket.coefficients)

fit = ket.to_manybody_basis().nearest_product_state(seeds=seeds, maxiter=maxiter)
print("product state:", ket)
print("fit: ", fit)
print("overlap:", ket.dagger() * fit )
print("entanglement = ", 1-abs(fit.dagger() * ket))

print("======== ENTANGLED =======")
# test for entangled state
ket = HilbertState(n_particles, isospin=isospin).randomize(seed)

print(ket.coefficients)

fit = ket.nearest_product_state(seeds=seeds, maxiter=maxiter)
print("hilbert state:", ket)
print("fit:", fit.to_manybody_basis())
print("entanglement = ", 1-abs(fit.to_manybody_basis().dagger().inner(ket)))
print("fit product state:", fit)