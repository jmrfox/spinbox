import nuctest as nt
from quap import *
from tqdm import tqdm
from cProfile import Profile
from pstats import SortKey, Stats
from multiprocessing.pool import Pool


num_particles = 2
ident = ManyBodyBasisSpinIsospinOperator(num_particles)
# list constructors make generating operators more streamlined
sig = [[ManyBodyBasisSpinIsospinOperator(num_particles).sigma(i,a) for a in [0, 1, 2]] for i in range(num_particles)]
tau = [[ManyBodyBasisSpinIsospinOperator(num_particles).tau(i,a) for a in [0, 1, 2]] for i in range(num_particles)]
# access like sig[particle][xyz]


def g_pade_sig(dt, asig, i, j):
    out = ManyBodyBasisSpinIsospinOperator(num_particles).zeros()
    for a in range(3):
        for b in range(3):
            out += asig[a, b] * sig[i][a] * sig[j][b]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_sigtau(dt, asigtau, i, j):
    out = ManyBodyBasisSpinIsospinOperator(num_particles).zeros()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                out += asigtau[a, b] * sig[i][a] * sig[j][b] * tau[i][c] * tau[j][c]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_tau(dt, atau, i, j):
    out = ManyBodyBasisSpinIsospinOperator(num_particles).zeros()
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


def g_linear_ls(bls, i, j):
    # linear approx to LS
    out = ident
    for a in range(3):
         out += - 0.5j * bls[a] * ( sig[i][a] + sig[j][a] ) 
    return out


def g_ls_onebody(bls, i, j):
    # one-body part of the LS propagator factorization
    out = ident.copy()
    for a in range(3):
        norm = np.exp(- 0.5 * bls[a]**2)
        out = norm * (- 1.j * bls[a] * (sig[i][a] + sig[j][a])).exponentiate() * out
    return out


def g_gauss_sample(dt, a, x, opi, opj):
    k = np.sqrt(-0.5 * dt * a, dtype=complex)
    norm = np.exp(0.5 * dt * a)
    gi = np.cosh(k * x) * ident + np.sinh(k * x) * opi
    gj = np.cosh(k * x) * ident + np.sinh(k * x) * opj
    return norm * gi * gj


def g_rbm_sample(dt, a, h, opi, opj):
    norm = np.exp(-0.5 * dt * np.abs(a)) / 2
    W = np.arctanh(np.sqrt(np.tanh(0.5 * dt * np.abs(a))))
    arg = W * (2 * h - 1)
    qi = np.cosh(arg) * ident + np.sinh(arg) * opi
    qj = np.cosh(arg) * ident - np.sign(a) * np.sinh(arg) * opj
    return 2 * norm * qi * qj


def load_ket(filename):
    c = read_coeffs(filename)
    sp = OneBodyBasisSpinIsospinState(num_particles, 'ket', c.reshape(-1, 1))
    return sp


if __name__ == "__main__":
    data_dir = './data/h2/'
    ket = load_ket(data_dir+'fort.770').to_many_body_state()
    print("INITIAL KET\n", ket.coefficients)
    asig = np.loadtxt(data_dir+'fort.7701').reshape((3,2,3,2), order='F')
    asigtau = np.loadtxt(data_dir+'fort.7702').reshape((3,2,3,2), order='F')
    atau = np.loadtxt(data_dir+'fort.7703').reshape((2,2), order='F')
    vcoul = np.loadtxt(data_dir+'fort.7704').reshape((2,2), order='F')
    bls = nt.make_bls()

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
        # ls_type = 'linear'
        ls_type = 'full'
        if ls_type is None:
            pass
        elif ls_type == 'linear':
            ket = g_linear_ls(bls, i, j) * ket
        elif ls_type == 'full':
            ket = g_ls_onebody(bls, i, j) * ket
            for a in range(3):
                for b in range(3):
                    asigls = - bls[a]* bls[b]
                    # norm = np.exp(- 0.5 * (np.abs(asigls) + bls[a]**2))
                    # ket = (1/norm) * g_rbm_sample(1, asigls, h, sig[i][a], sig[j][b]) * ket
                    ket = g_rbm_sample(1, asigls, h, sig[i][a], sig[j][b]) * ket

    print("FINAL KET\n", ket.coefficients)
    print('DONE')
