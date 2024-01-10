from quap import *

sqrt = lambda x: np.sqrt(x, dtype=complex)

num_particles = 2
ident = OneBodyBasisSpinIsospinOperator(num_particles)
# list constructors make generating operators more streamlined
sig = [[OneBodyBasisSpinIsospinOperator(num_particles).sigma(i,a) for a in [0, 1, 2]] for i in range(num_particles)]
tau = [[OneBodyBasisSpinIsospinOperator(num_particles).tau(i,a) for a in [0, 1, 2]] for i in range(num_particles)]
# access like sig[particle][xyz]


def load_ket(filename):
    c = read_coeffs(filename)
    sp = OneBodyBasisSpinIsospinState(num_particles, 'ket', c.reshape(-1, 1))
    return sp

def g_sigma_1(dt, v, i, j):
    """for testing purposes
    want to apply exp(- dt/2 V sigma_i sigma_j)
     = cosh(k) + sinh(k) sigma_i sigma_j
     can we do this in 2 seperate lines?"""
    k = - 0.5 * v * dt
    ck, sk = np.cosh(k), np.sinh(k)
    out = ident.scalar_mult(i, ck).scalar_mult(j, ck) + sig[i][0].scalar_mult(i, sk) * sig[j][0].scalar_mult(j, sk)
    return out

def g_sigma_2(dt, v, i, j):
    """for testing purposes
    want to apply exp(- dt/2 V sigma_i sigma_j)
     = cosh(k) + sinh(k) sigma_i sigma_j
     can we do this in 2 seperate lines?"""
    k = - 0.5 * v * dt
    ck, sk = np.cosh(k), np.sinh(k)
    out = ident.copy()
    out = (ident.scalar_mult(i, sqrt(ck)) + sig[i][0].scalar_mult(i, sqrt(sk)) ) * out
    out = (ident.scalar_mult(j, sqrt(ck)) + sig[j][0].scalar_mult(j, sqrt(sk)) ) * out
    return out


if __name__ == "__main__":
    data_dir = './data/h2/'
    ket = load_ket(data_dir+'fort.770')
    print("INITIAL KET\n", ket.coefficients)

    dt = 0.01
    v = 1.0
    s1 = g_sigma_1(dt, v, 0, 1) * ket
    s2 = g_sigma_2(dt, v, 0, 1) * ket
    print(s1)
    print(s2)