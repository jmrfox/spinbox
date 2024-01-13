import nuctest as nt
from quap import *
from tqdm import tqdm
from cProfile import Profile
from pstats import SortKey, Stats
from multiprocessing.pool import Pool


ident = OneBodyBasisSpinIsospinOperator(nt.num_particles)
# list constructors make generating operators more streamlined
sig = [[OneBodyBasisSpinIsospinOperator(nt.num_particles).sigma(i,a) for a in [0, 1, 2]] for i in range(nt.num_particles)]
tau = [[OneBodyBasisSpinIsospinOperator(nt.num_particles).tau(i,a) for a in [0, 1, 2]] for i in range(nt.num_particles)]
# access like sig[particle][xyz]


def load_ket(filename):
    c = read_coeffs(filename)
    sp = OneBodyBasisSpinIsospinState(nt.num_particles, 'ket', c.reshape(-1, 1))
    return sp

def g_coul_onebody(dt, v, i, j):
    """just the one-body part of the factored coulomb propagator
    for use along with auxiliary field propagator for 2 body part
    although I call this one-body factor, it in fact acts on both particles, just individually"""
    k = - 0.125 * v * dt
    norm = np.exp(k)
    ck, sk = np.cosh(k), np.sinh(k)
    out = ident.scalar_mult(i, ck).scalar_mult(j, ck) + tau[i][2].scalar_mult(i, sk) * tau[j][2].scalar_mult(j, sk)
    return out.spread_scalar_mult(norm)

def g_ls_linear(gls, i, a):
    """just the one-body part of the factored LS propagator
    this is actually a 1-body operator. i is the particle index"""
    k = - 1.j * gls[a, i]
    out = ident + sig[i][a].scalar_mult(i, k)
    return out

def g_ls_onebody(gls, i, a):
    """just the one-body part of the factored LS propagator
    this is actually a 1-body operator. i is the particle index"""
    k = - 1.j * gls[a, i]
    ck, sk = np.cosh(k, dtype=complex), np.sinh(k, dtype=complex)
    out = ident.scalar_mult(i, ck) + sig[i][a].scalar_mult(i, sk)
    return out

def g_ls_twobody(gls, i, j, a, b):
    k = 0.5 * gls[a, i] * gls[b, j]
    ck, sk = np.cosh(k, dtype=complex), np.sinh(k, dtype=complex)
    out = ident.scalar_mult(i, ck).scalar_mult(j, ck) + sig[i][a].scalar_mult(i, sk) * sig[j][b].scalar_mult(j, sk)
    return out

def g_gauss_sample(dt: float, a: float, x, i: int, j: int, opi: OneBodyBasisSpinIsospinOperator, opj: OneBodyBasisSpinIsospinOperator):
    k = np.sqrt(-0.5 * dt * a, dtype=complex)
    norm = np.exp(0.5 * dt * a)
    out = ident.scalar_mult(i, np.cosh(k * x)).scalar_mult(j, np.cosh(k * x)) + opi.scalar_mult(i, np.sinh(k * x)) * opj.scalar_mult(j, np.sinh(k * x))
    return out.spread_scalar_mult(norm)


def g_rbm_sample(dt, a, h, i, j, opi, opj):
    norm = np.exp(-0.5 * dt * np.abs(a))
    W = np.arctanh(np.sqrt(np.tanh(0.5 * dt * np.abs(a))))
    arg = W * (2 * h - 1)
    out = ident.scalar_mult(i, np.cosh(arg)).scalar_mult(j, np.cosh(arg)) + opi.scalar_mult(i, np.sinh(arg)) * opj.scalar_mult(j, -np.sign(a) * np.sinh(arg))
    return out.spread_scalar_mult(norm)


def load_h2():
    data_dir = './data/h2/'
    c = read_coeffs(data_dir+'fort.770')
    ket = OneBodyBasisSpinIsospinState(nt.num_particles, 'ket', c.reshape(-1, 1)).to_many_body_state()    
    asig = np.loadtxt(data_dir+'fort.7701').reshape((3,2,3,2), order='F')
    asigtau = np.loadtxt(data_dir+'fort.7702').reshape((3,2,3,2), order='F')
    atau = np.loadtxt(data_dir+'fort.7703').reshape((2,2), order='F')
    vcoul = np.loadtxt(data_dir+'fort.7704').reshape((2,2), order='F')
    bls = nt.make_bls()
    return ket, asig, asigtau, atau, vcoul, bls


ls_test = True
if __name__ == "__main__" and ls_test:
    bra, ket = nt.make_test_states()
    pots = nt.make_all_potentials(scale = 0.1)
    gls = np.sum(pots['bls'], axis = 2)

    ket_0 = ket.copy()
    for i in range(nt.num_particles):
        for a in range(3):
            ket_0 = g_ls_linear(gls, i, a) * ket_0

    ket_1 = ket.copy()
    trace_factor = np.exp( - 0.5 * np.sum(gls**2))
    ket_1 = trace_factor * ket_1
    for i in range(nt.num_particles):
        for a in range(3):
            ket_1 = g_ls_onebody(gls, i, a) * ket_1
        for j in range(i):
            for b in range(3):
                ket_1 = g_ls_twobody(gls, i, j, a, b) * ket_1   
    
    # print("linear ket: \n", ket_0)
    # print("full ket: \n", ket_1)
    print("linear bracket: \n", bra * ket_0)
    print("full bracket: \n", bra * ket_1)
    
    
    print('DONE')


elif __name__ == "__main__" and not ls_test:
    ket, asig, asigtau, atau, vcoul, bls = load_h2()
    print("INITIAL KET\n", ket.coefficients)
            
    pairs_ij = [[0,1]]
    h = 1.0
    for i,j in pairs_ij:
        for a in range(3):
            for b in range(3):
                norm = np.exp(-nt.dt * 0.5 * np.abs(asig[a, i, b, j]))
                ket = (g_rbm_sample(nt.dt, asig[a, i, b, j], h, i, j, sig[i][a], sig[j][b]) * ket).spread_scalar_mult(1/norm)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    norm = np.exp(-nt.dt * 0.5 * np.abs(asigtau[a, i, b, j]))
                    ket = (g_rbm_sample(nt.dt, asigtau[a, i, b, j], h, i, j, sig[i][a] * tau[i][c], sig[j][b] * tau[j][c]) * ket).spread_scalar_mult(1/norm)
        for c in range(3):
            norm = np.exp(-nt.dt * 0.5 * np.abs(atau[i, j]))
            ket = (g_rbm_sample(nt.dt, atau[i, j], h, i, j, tau[i][c], tau[j][c]) * ket).spread_scalar_mult(1/norm)
        # coulomb
        norm = np.exp(-nt.dt * 0.125 * (vcoul[i, j] + np.abs(vcoul[i, j])))
        ket = (g_coul_onebody(nt.dt, vcoul[i, j], i, j) * g_rbm_sample(nt.dt, 0.25 * vcoul[i, j], h, i, j, tau[i][2], tau[j][2]) * ket).spread_scalar_mult(1/norm)
        # LS
        for a in range(3):
            ket = g_ls_onebody(bls, i, j, a) * ket
        for a in range(3):
            for b in range(3):
                asigls = - bls[a]* bls[b]
                # norm = np.exp(- 0.5 * (np.abs(asigls) + bls[a]**2))
                # ket = (g_rbm_sample(1, asigls, h, i, j, sig[i][a], sig[j][b]) * ket).spread_scalar_mult(1/norm)
                ket = g_rbm_sample(1, asigls, h, i, j, sig[i][a], sig[j][b]) * ket
    print("FINAL KET\n", ket.coefficients)
    print("FINAL KET in many-body basis\n", ket.to_many_body_state().coefficients)
    print('DONE')
