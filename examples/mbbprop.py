import nuctest as nt
from quap import *
from tqdm import tqdm
from cProfile import Profile
from pstats import SortKey, Stats
from multiprocessing.pool import Pool


ident = ManyBodyBasisSpinIsospinOperator(nt.num_particles)
# list constructors make generating operators more streamlined
sig = [[ManyBodyBasisSpinIsospinOperator(nt.num_particles).sigma(i,a) for a in [0, 1, 2]] for i in range(nt.num_particles)]
tau = [[ManyBodyBasisSpinIsospinOperator(nt.num_particles).tau(i,a) for a in [0, 1, 2]] for i in range(nt.num_particles)]
# sig[particle][xyz]


def g_pade_sig(dt, asig, i, j):
    out = ManyBodyBasisSpinIsospinOperator(nt.num_particles).zeros()
    for a in range(3):
        for b in range(3):
            out += asig[a, b] * sig[i][a] * sig[j][b]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_sigtau(dt, asigtau, i, j):
    out = ManyBodyBasisSpinIsospinOperator(nt.num_particles).zeros()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                out += asigtau[a, b] * sig[i][a] * sig[j][b] * tau[i][c] * tau[j][c]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_tau(dt, atau, i, j):
    out = ManyBodyBasisSpinIsospinOperator(nt.num_particles).zeros()
    for c in range(3):
        out += atau * tau[i][c] * tau[j][c]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_coul(dt, v, i, j):
    out = ident + tau[i][2] + tau[j][2] + tau[i][2] * tau[j][2]
    out = -0.125 * v * dt * out
    return out.exponentiate()


def g_coul_onebody(dt, v, i, j):
    """just the one-body part of the expanded coulomb propagator
    for use along with auxiliary field propagators"""
    out = - 0.125 * v * dt * (ident + tau[i][2] + tau[j][2])
    return out.exponentiate()


def g_ls_linear(gls, i):
    # linear approx to LS
    out = ident
    for a in range(3):
        out += - 1.j * gls[a, i] * sig[i][a] 
    return out

def g_ls_onebody(gls, i):
    # one-body part of the LS propagator factorization
    out = ManyBodyBasisSpinIsospinOperator(nt.num_particles).zeros()
    for a in range(3):
        out += - 1.j * gls[a, i] * sig[i][a]
    return out.exponentiate()

def g_ls_twobody(gls, i, j):
    # one-body part of the LS propagator factorization
    out = ManyBodyBasisSpinIsospinOperator(nt.num_particles).zeros()
    for a in range(3):
        for b in range(3):
            out += 0.5 * gls[a, i] * gls[b, j] * sig[i][a] * sig[j][b]
    return out.exponentiate()


def g_gauss_sample(dt, a, x, opi, opj):
    k = csqrt(-0.5 * dt * a)
    norm = np.exp(0.5 * dt * a)
    gi = np.cosh(k * x) * ident + np.sinh(k * x) * opi
    gj = np.cosh(k * x) * ident + np.sinh(k * x) * opj
    return norm * gi * gj


def g_rbm_sample(dt, a, h, opi, opj):
    norm = np.exp(-0.5 * dt * np.abs(a)) / 2
    W = np.arctanh(csqrt(np.tanh(0.5 * dt * np.abs(a))))
    arg = W * (2 * h - 1)
    qi = np.cosh(arg) * ident + np.sinh(arg) * opi
    qj = np.cosh(arg) * ident - np.sign(a) * np.sinh(arg) * opj
    return 2 * norm * qi * qj


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
    bra, ket = nt.make_test_states(manybody=True)
    
    pots = nt.make_all_potentials(scale = 0.1)
    gls = np.sum(pots['bls'], axis = 2)

    ket_0 = ket.copy()
    for i in range(nt.num_particles):
        ket_0 = g_ls_linear(gls, i) * ket_0
    
    ket_1 = ket.copy()
    trace_factor = np.exp( - 0.5 * np.sum(gls**2))
    ket_1 = trace_factor * ket_1
    for i in range(nt.num_particles):
        ket_1 = g_ls_onebody(gls, i) * ket_1
        for j in range(i):
            ket_1 = g_ls_twobody(gls, i, j) * ket_1
    
    # print("linear ket: \n", ket_0)
    # print("full ket: \n", ket_1)
    print("linear bracket: \n", bra * ket_0)
    print("full bracket: \n", bra * ket_1)
    
    print('DONE')

if __name__ == "__main__" and not ls_test:
    ket, asig, asigtau, atau, vcoul, bls = load_h2()
    print("INITIAL KET\n", ket.coefficients)

    pairs_ij = [[0,1]]
    h = 1.0
    for i,j in pairs_ij:
        for a in range(3):
            for b in range(3):
                norm = np.exp(-nt.dt * 0.5 * np.abs(asig[a, i, b, j]))
                ket = (1/norm) * g_rbm_sample(nt.dt, asig[a, i, b, j], h, sig[i][a], sig[j][b]) * ket
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    norm = np.exp(-nt.dt * 0.5 * np.abs(asigtau[a, i, b, j]))
                    ket = (1/norm) * g_rbm_sample(nt.dt, asigtau[a, i, b, j], h, sig[i][a] * tau[i][c], sig[j][b] * tau[j][c]) * ket
        for c in range(3):
            norm = np.exp(-nt.dt * 0.5 * np.abs(atau[i, j]))
            ket = (1/norm) * g_rbm_sample(nt.dt, atau[i, j], h, tau[i][c], tau[j][c]) * ket
        # coulomb
        norm = np.exp(-nt.dt * 0.125 * (vcoul[i, j] + np.abs(vcoul[i, j])))
        ket = (1/norm) * g_coul_onebody(nt.dt, vcoul[i, j], i, j) * g_rbm_sample(nt.dt, 0.25 * vcoul[i, j], h, tau[i][2], tau[j][2]) * ket
        # LS
        ls_type = 'linear'
        # ls_type = 'full'
        if ls_type is None:
            pass
        elif ls_type == 'linear':
            ket = g_ls_linear(bls, i, j) * ket
        elif ls_type == 'full':
            ket = g_ls_onebody(bls, i, j) * ket
            trace_factor = np.exp( 0.5 * np.sum(bls**2))
            ket = trace_factor * g_ls_twobody(bls, i, j) * ket                   

    print("FINAL KET\n", ket.coefficients)
    print('DONE')
