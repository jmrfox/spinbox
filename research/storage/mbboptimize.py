from spinbox import *
from scipy.optimize import minimize
np.set_printoptions(linewidth=300)

num_particles = 2
sig = [[ManyBodyBasisSpinIsospinOperator(num_particles).sigma(i,a) for a in [0, 1, 2]] for i in range(num_particles)]

# hamiltonian 
rng = np.random.default_rng(12345)
v = rng.standard_normal(size=(num_particles, num_particles, 3))
# print('v:', v)
ham = ManyBodyBasisSpinIsospinOperator(num_particles).zeros()
for i in range(num_particles):
    for j in range(i):
        for a in range(3):
            ham += v[i,j,a]*sig[i][a]*sig[j][a]
ham = 0.5*(ham + ham.transpose())

print(ham)

# trial state
global_seed = 1312
bra, ket = random_spinisospin_bra_ket(num_particles, bra_seed=global_seed, ket_seed=global_seed)
bra = bra.to_many_body_state()
ket = ket.to_many_body_state()
trial_energy = bra * ham * ket
print('starting energy: ', trial_energy)


# optimize
def obj(coeffs):
    print(coeffs)
    bra = ManyBodyBasisSpinIsospinState(num_particles, 'bra', coeffs)
    ket = ManyBodyBasisSpinIsospinState(num_particles, 'ket', coeffs.reshape(1,-1))
    return bra * ham * ket

opt = minimize(obj, bra.coefficients.flatten())