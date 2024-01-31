import nuctest as nt
from quap import *

ident = ManyBodyBasisSpinIsospinOperator(nt.n_particles)
# list constructors make generating operators more streamlined
sig = [[ManyBodyBasisSpinIsospinOperator(nt.n_particles).sigma(i,a) for a in [0, 1, 2]] for i in range(nt.n_particles)]
tau = [[ManyBodyBasisSpinIsospinOperator(nt.n_particles).tau(i,a) for a in [0, 1, 2]] for i in range(nt.n_particles)]
# sig[particle][xyz]


def g_pade_sig(dt, asig, i, j):
    out = ManyBodyBasisSpinIsospinOperator(nt.n_particles).zeros()
    for a in range(3):
        for b in range(3):
            out += asig[a, b] * sig[i][a] * sig[j][b]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_sigtau(dt, asigtau, i, j):
    out = ManyBodyBasisSpinIsospinOperator(nt.n_particles).zeros()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                out += asigtau[a, b] * sig[i][a] * sig[j][b] * tau[i][c] * tau[j][c]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_tau(dt, atau, i, j):
    out = ManyBodyBasisSpinIsospinOperator(nt.n_particles).zeros()
    for c in range(3):
        out += atau * tau[i][c] * tau[j][c]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_coul(dt, v, i, j):
    out = ident + tau[i][2] + tau[j][2] + tau[i][2] * tau[j][2]
    out = -0.125 * v * dt * out
    return out.exponentiate()


def g_coulomb_onebody(dt, v, i):
    """just the one-body part of the expanded coulomb propagator
    for use along with auxiliary field propagators"""
    out = - 0.125 * v * dt * tau[i][2]
    return out.exponentiate()

# def g_coul_onebody(dt, v, i, j):
#     """just the one-body part of the expanded coulomb propagator
#     for use along with auxiliary field propagators"""
#     out = - 0.125 * v * dt * (ident + tau[i][2] + tau[j][2])
#     return out.exponentiate()


def g_ls_linear(gls_ai, i, a):
    # linear approx to LS
    out = ident - 1.j * gls_ai * sig[i][a] 
    return out

def g_ls_onebody(gls_ai, i, a):
    # one-body part of the LS propagator factorization
    out = - 1.j * gls_ai * sig[i][a]
    return out.exponentiate()

def g_ls_twobody(gls_ai, gls_bj, i, j, a, b):
    # two-body part of the LS propagator factorization
    out = 0.5 * gls_ai * gls_bj * sig[i][a] * sig[j][b]
    return out.exponentiate()


def g_gauss_sample(dt, a, x, opi, opj):
    k = csqrt(-0.5 * dt * a)
    norm = cexp(0.5 * dt * a)
    gi = ccosh(k * x) * ident + csinh(k * x) * opi
    gj = ccosh(k * x) * ident + csinh(k * x) * opj
    return norm * gi * gj


def g_rbm_sample(dt, a, h, opi, opj):
    norm = cexp(-0.5 * dt * np.abs(a))
    W = carctanh(csqrt(ctanh(0.5 * dt * np.abs(a))))
    arg = W * (2 * h - 1)
    qi = ccosh(arg) * ident + csinh(arg) * opi
    qj = ccosh(arg) * ident - np.sign(a) * csinh(arg) * opj
    return norm * qi * qj


def load_h2():
    data_dir = './data/h2/'
    c = read_coeffs(data_dir+'fort.770')
    ket = OneBodyBasisSpinIsospinState(nt.n_particles, 'ket', c.reshape(nt.n_particles, 4, 1)).to_many_body_state()    
    asig = np.loadtxt(data_dir+'fort.7701').reshape((3,2,3,2), order='F')
    asigtau = np.loadtxt(data_dir+'fort.7702').reshape((3,2,3,2), order='F')
    atau = np.loadtxt(data_dir+'fort.7703').reshape((2,2), order='F')
    vcoul = np.loadtxt(data_dir+'fort.7704').reshape((2,2), order='F')
    bls = nt.make_bls()
    return ket, asig, asigtau, atau, vcoul, bls


def prop_gauss_fixed(bra, ket, pots, x):
    print('GAUSS')
    asig = pots['asig'] 
    asigtau = pots['asigtau']
    atau = pots['atau']
    vcoul = pots['vcoul']
    bls = pots['bls']
    gls = np.sum(pots['bls'], axis = 2)


    # SIGMA
    for i,j in nt.pairs_ij:
        for a in range(3):
            for b in range(3):
                # norm = cexp(-nt.dt * 0.5 * np.abs(asig[a, i, b, j]))
                # ket = (1/norm) * g_rbm_sample(nt.dt, asig[a, i, b, j], h, sig[i][a], sig[j][b]) * ket
                ket = g_gauss_sample(nt.dt, asig[a, i, b, j], x, sig[i][a], sig[j][b]) * ket
    # SIGMA TAU
    for i,j in nt.pairs_ij:
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    # norm = cexp(-nt.dt * 0.5 * np.abs(asigtau[a, i, b, j]))
                    # ket = (1/norm) * g_rbm_sample(nt.dt, asigtau[a, i, b, j], h, sig[i][a] * tau[i][c], sig[j][b] * tau[j][c]) * ket
                    ket = g_gauss_sample(nt.dt, asigtau[a, i, b, j], x, sig[i][a] * tau[i][c], sig[j][b] * tau[j][c]) * ket
    # TAU
    for i,j in nt.pairs_ij:
        for c in range(3):
            # norm = cexp(-nt.dt * 0.5 * np.abs(atau[i, j]))
            # ket = (1/norm) * g_rbm_sample(nt.dt, atau[i, j], h, tau[i][c], tau[j][c]) * ket
            ket = g_gauss_sample(nt.dt, atau[i, j], x, tau[i][c], tau[j][c]) * ket
    # COULOMB
    for i,j in nt.pairs_ij:
        # norm_1b = cexp(-nt.dt * 0.125 * vcoul[i, j])
        # norm_rbm = cexp(-nt.dt * 0.125 * np.abs(vcoul[i, j]))
        # ket = g_coulomb_onebody(nt.dt, vcoul[i, j], i) * g_coulomb_onebody(nt.dt, vcoul[i, j], j) * ket
        # ket = (1/norm) * g_rbm_sample(nt.dt, 0.25 * vcoul[i, j], h, tau[i][2], tau[j][2]) * ket
        ket = g_coulomb_onebody(nt.dt, vcoul[i, j], i) * g_coulomb_onebody(nt.dt, vcoul[i, j], j) * ket
        ket = g_gauss_sample(nt.dt, 0.25 * vcoul[i, j], x, tau[i][2], tau[j][2]) * ket
    # LS
    do_ls = True
    if do_ls:
        for i in range(nt.n_particles):
            for a in range(3):
                ket = g_ls_onebody(gls[a, i], i, a) * ket
        for i,j in nt.pairs_ij:
            for a in range(3):
                for b in range(3):
                    asigls = - gls[a, i]* gls[b, j]
                    ket = g_gauss_sample(1, asigls, x, sig[i][a], sig[j][b]) * ket
        trace_factor = cexp( 0.5 * np.sum(gls**2))
        ket = trace_factor * ket

    print("FINAL KET\n", ket.coefficients)
    print('norm = ', ket.dagger() * ket)
    print('MBB bracket = ', bra * ket)
    print('GAUSS DONE')



def prop_rbm_fixed(bra, ket, pots, h):
    print('RBM')
    asig = pots['asig'] 
    asigtau = pots['asigtau']
    atau = pots['atau']
    vcoul = pots['vcoul']
    bls = pots['bls']
    gls = np.sum(pots['bls'], axis = 2)

    # FIXED AUX FIELD CALCULATION
    
    # SIGMA
    for i,j in nt.pairs_ij:
        for a in range(3):
            for b in range(3):
                # norm = cexp(-nt.dt * 0.5 * np.abs(asig[a, i, b, j]))
                # ket = (1/norm) * g_rbm_sample(nt.dt, asig[a, i, b, j], h, sig[i][a], sig[j][b]) * ket
                ket = g_rbm_sample(nt.dt, asig[a, i, b, j], h, sig[i][a], sig[j][b]) * ket
    # SIGMA TAU
    for i,j in nt.pairs_ij:
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    # norm = cexp(-nt.dt * 0.5 * np.abs(asigtau[a, i, b, j]))
                    # ket = (1/norm) * g_rbm_sample(nt.dt, asigtau[a, i, b, j], h, sig[i][a] * tau[i][c], sig[j][b] * tau[j][c]) * ket
                    ket = g_rbm_sample(nt.dt, asigtau[a, i, b, j], h, sig[i][a] * tau[i][c], sig[j][b] * tau[j][c]) * ket
    # TAU
    for i,j in nt.pairs_ij:
        for c in range(3):
            # norm = cexp(-nt.dt * 0.5 * np.abs(atau[i, j]))
            # ket = (1/norm) * g_rbm_sample(nt.dt, atau[i, j], h, tau[i][c], tau[j][c]) * ket
            ket = g_rbm_sample(nt.dt, atau[i, j], h, tau[i][c], tau[j][c]) * ket
    # COULOMB
    for i,j in nt.pairs_ij:
        # norm_1b = cexp(-nt.dt * 0.125 * vcoul[i, j])
        # norm_rbm = cexp(-nt.dt * 0.125 * np.abs(vcoul[i, j]))
        # ket = g_coulomb_onebody(nt.dt, vcoul[i, j], i) * g_coulomb_onebody(nt.dt, vcoul[i, j], j) * ket
        # ket = (1/norm) * g_rbm_sample(nt.dt, 0.25 * vcoul[i, j], h, tau[i][2], tau[j][2]) * ket
        ket = g_coulomb_onebody(nt.dt, vcoul[i, j], i) * g_coulomb_onebody(nt.dt, vcoul[i, j], j) * ket
        ket = g_rbm_sample(nt.dt, 0.25 * vcoul[i, j], h, tau[i][2], tau[j][2]) * ket
    # LS
    do_ls = True
    if do_ls:
        for i in range(nt.n_particles):
            for a in range(3):
                ket = g_ls_onebody(gls[a, i], i, a) * ket
        for i,j in nt.pairs_ij:
            for a in range(3):
                for b in range(3):
                    asigls = - gls[a, i]* gls[b, j]
                    ket = g_rbm_sample(1, asigls, h, sig[i][a], sig[j][b]) * ket
        trace_factor = cexp( 0.5 * np.sum(gls**2))
        ket = trace_factor * ket

    print("FINAL KET\n", ket.coefficients)
    print('norm = ', ket.dagger() * ket)
    print('MBB bracket = ', bra * ket)
    print('RBM DONE')

if __name__ == "__main__":
    # ket, asig, asigtau, atau, vcoul, bls = load_h2()
    bra, ket = nt.make_test_states(manybody=True)
    pots = nt.make_all_potentials(scale = 1.0)

    prop_gauss_fixed(bra, ket, pots, x=1.0)
    prop_rbm_fixed(bra, ket, pots, h=1.0)   
    
