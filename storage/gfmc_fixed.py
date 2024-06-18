from spinbox import *

n_particles = 2
dt = 0.001
pairs_ij = interaction_indices(n_particles)

ident = GFMCSpinIsospinOperator(n_particles)
# list constructors make generating operators more streamlined
sig = [[GFMCSpinIsospinOperator(n_particles).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
tau = [[GFMCSpinIsospinOperator(n_particles).apply_tau(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
# sig[particle][xyz]



# this script is to check HS vs RBM for a single fixed value of ALL auxiliary fields
# mainly just for testing the new classes


def g_coulomb_pair(dt, v, i, j):
    out = - 0.125 * v * dt * (ident + tau[i][2] + tau[j][2]) 
    return out.exp()

def g_coulomb_onebody(dt, v, i):
    out = - 0.125 * v * dt * tau[i][2]
    return out.exp()

def g_ls_linear(gls_ai, i, a):
    # linear approx to LS
    out = ident - 1.j * gls_ai * sig[i][a] 
    return out

def g_ls_onebody(gls_ai, i, a):
    # one-body part of the LS propagator factorization
    out = - 1.j * gls_ai * sig[i][a]
    return out.exp()

def g_ls_twobody(gls_ai, gls_bj, i, j, a, b):
    # two-body part of the LS propagator factorization
    out = 0.5 * gls_ai * gls_bj * sig[i][a] * sig[j][b]
    return out.exp()


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


def prop_gauss_fixed(ket, pots, x):
    print('GAUSS')
    asig = pots['asig'] 
    asigtau = pots['asigtau']
    atau = pots['atau']
    vcoul = pots['vcoul']
    gls = pots['gls']

    # SIGMA
    for i,j in pairs_ij:
        for a in range(3):
            for b in range(3):
                ket = g_gauss_sample(dt, asig[a, i, b, j], x, sig[i][a], sig[j][b]) * ket
    # SIGMA TAU
    for i,j in pairs_ij:
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    ket = g_gauss_sample(dt, asigtau[a, i, b, j], x, sig[i][a] * tau[i][c], sig[j][b] * tau[j][c]) * ket
    # TAU
    for i,j in pairs_ij:
        for c in range(3):
            ket = g_gauss_sample(dt, atau[i, j], x, tau[i][c], tau[j][c]) * ket
    # COULOMB
    for i,j in pairs_ij:
        # ket = g_coulomb_onebody(dt, vcoul[i, j], i) * g_coulomb_onebody(dt, vcoul[i, j], j) * ket
        ket = g_coulomb_pair(dt, vcoul[i,j], i, j) * ket
        ket = g_gauss_sample(dt, 0.25 * vcoul[i, j], x, tau[i][2], tau[j][2]) * ket
    # LS
    for i in range(n_particles):
        for a in range(3):
            ket = g_ls_onebody(gls[a, i], i, a) * ket
    for i,j in pairs_ij:
        for a in range(3):
            for b in range(3):
                asigls = - gls[a, i]* gls[b, j]
                ket = g_gauss_sample(1, asigls, x, sig[i][a], sig[j][b]) * ket
    trace_factor = cexp( 0.5 * np.sum(gls**2))
    ket = trace_factor * ket

    return ket


def prop_rbm_fixed(ket, pots, h):
    print('RBM')
    asig = pots['asig'] 
    asigtau = pots['asigtau']
    atau = pots['atau']
    vcoul = pots['vcoul']
    gls = pots['gls']

    # FIXED AUX FIELD CALCULATION
    
    # SIGMA
    for i,j in pairs_ij:
        for a in range(3):
            for b in range(3):
                ket = g_rbm_sample(dt, asig[a, i, b, j], h, sig[i][a], sig[j][b]) * ket
    # SIGMA TAU
    for i,j in pairs_ij:
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    ket = g_rbm_sample(dt, asigtau[a, i, b, j], h, sig[i][a] * tau[i][c], sig[j][b] * tau[j][c]) * ket
    # TAU
    for i,j in pairs_ij:
        for c in range(3):
            ket = g_rbm_sample(dt, atau[i, j], h, tau[i][c], tau[j][c]) * ket
    # COULOMB
    for i,j in pairs_ij:
        ket = ket = g_coulomb_pair(dt, vcoul[i,j], i, j) * ket
        ket = g_rbm_sample(dt, 0.25 * vcoul[i, j], h, tau[i][2], tau[j][2]) * ket
    # LS
    for i in range(n_particles):
        for a in range(3):
            ket = g_ls_onebody(gls[a, i], i, a) * ket
    for i,j in pairs_ij:
        for a in range(3):
            for b in range(3):
                asigls = gls[a, i]* gls[b, j]
                ket = g_rbm_sample(1, - asigls, h, sig[i][a], sig[j][b]) * ket
    trace_factor = cexp( 0.5 * np.sum(gls**2))
    ket = trace_factor * ket

    return ket


def prop_rbm_fixed_unnorm(ket, pots, h):
    print('RBM')
    asig = pots['asig'] 
    asigtau = pots['asigtau']
    atau = pots['atau']
    vcoul = pots['vcoul']
    gls = pots['gls']

    # FIXED AUX FIELD CALCULATION
    
    # SIGMA
    for i,j in pairs_ij:
        for a in range(3):
            for b in range(3):
                norm = cexp(-dt * 0.5 * np.abs(asig[a, i, b, j]))
                ket = (1/norm) * g_rbm_sample(dt, asig[a, i, b, j], h, sig[i][a], sig[j][b]) * ket
    # SIGMA TAU
    for i,j in pairs_ij:
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    norm = cexp(-dt * 0.5 * np.abs(asigtau[a, i, b, j]))
                    ket = (1/norm) * g_rbm_sample(dt, asigtau[a, i, b, j], h, sig[i][a] * tau[i][c], sig[j][b] * tau[j][c]) * ket
    # TAU
    for i,j in pairs_ij:
        for c in range(3):
            norm = cexp(-dt * 0.5 * np.abs(atau[i, j]))
            ket = (1/norm) * g_rbm_sample(dt, atau[i, j], h, tau[i][c], tau[j][c]) * ket
    # COULOMB
    for i,j in pairs_ij:
        norm_1b = cexp(-dt * 0.125 * vcoul[i, j])
        norm_rbm = cexp(-dt * 0.125 * np.abs(vcoul[i, j]))
        ket = (1/norm_1b) * g_coulomb_pair(dt, vcoul[i, j], i, j) * ket
        ket = (1/norm_rbm) * g_rbm_sample(dt, 0.25 * vcoul[i, j], h, tau[i][2], tau[j][2]) * ket
    # LS
    for i in range(n_particles):
        for a in range(3):
            ket = g_ls_onebody(gls[a, i], i, a) * ket
    for i,j in pairs_ij:
        for a in range(3):
            for b in range(3):
                asigls = gls[a, i]* gls[b, j]
                norm = cexp(0.5 * np.abs(asigls))
                ket = (1/norm) * g_rbm_sample(1, - asigls, h, sig[i][a], sig[j][b]) * ket

    return ket

def load_h2(manybody=False, data_dir = './data/h2/'):
    # data_dir = './data/h2/'
    c_i = read_from_file(data_dir+'fort.770', complex=True, shape=(2,4,1))
    c_f = read_from_file(data_dir+'fort.775', complex=True, shape=(2,4,1))
    ket = AFDMCSpinIsospinState(2, c_i) 
    ket_f = AFDMCSpinIsospinState(2, c_f) 
    if manybody:
        ket = ket.to_manybody_basis()
        ket_f = ket_f.to_manybody_basis()
    asig = read_from_file(data_dir+'fort.7701', shape=(3,2,3,2))
    asigtau = read_from_file(data_dir+'fort.7702', shape=(3,2,3,2))
    atau = read_from_file(data_dir+'fort.7703', shape=(2,2))
    vcoul = read_from_file(data_dir+'fort.7704', shape=(2,2))
    gls = read_from_file(data_dir+'fort.7705', shape=(3,2))
    asigls = read_from_file(data_dir+'fort.7706', shape=(3,2,3,2))

    pot_dict={}
    pot_dict['asig'] = asig
    pot_dict['asigtau'] = asigtau
    pot_dict['atau'] = atau
    pot_dict['vcoul'] = vcoul
    pot_dict['gls'] = gls
    pot_dict['asigls'] = asigls
    # return ket, asig, asigtau, atau, vcoul, gls, asigls
    return ket, pot_dict, ket_f

def main(method):
    ket, pots, _ = load_h2(manybody=True, data_dir = './data/h2/')
    pots['asig'] = 1*pots['asig']
    pots['asigtau'] = 1*pots['asigtau']
    pots['atau'] = 1*pots['atau']
    pots['vcoul'] = 1*pots['vcoul']
    pots['gls'] = 1*pots['gls']
    bra = ket.copy().dagger()

    # print(pots["asig"].flatten())
    
    if method=='hs':
        ket_prop = prop_gauss_fixed(ket.copy(), pots, x=1.0)
    elif method=='rbm':
        ket_prop = prop_rbm_fixed(ket, pots, h=1.0)
    # ket_prop = prop_rbm_fixed_unnorm(ket, pots, h=1.0)
    return bra * ket_prop
    
def main_new(method):
    ket = AFDMCSpinIsospinState(n_particles, read_from_file("./data/h2/fort.770",complex=True, shape=(2,4,1))).to_manybody_basis()
    bra = ket.copy().dagger()

    pot = ArgonnePotential(n_particles)
    pot.read_sigma("./data/h2/fort.7701")
    pot.read_sigmatau("./data/h2/fort.7702")
    pot.read_tau("./data/h2/fort.7703")
    pot.read_coulomb("./data/h2/fort.7704")
    pot.read_spinorbit("./data/h2/fort.7705")
    
    # print(pot.sigma.coefficients.flatten())

    if method=='hs':
        hsprop = GFMCPropagatorHS(n_particles, dt, include_prefactors=True, mix=False)
    elif method=='rbm':
        hsprop = GFMCPropagatorRBM(n_particles, dt, include_prefactors=True, mix=False)

    ket_prop = ket.copy()
    ket_prop = hsprop.apply_sigma(ket_prop,pot,[1.0]*9)
    ket_prop = hsprop.apply_sigmatau(ket_prop,pot,[1.0]*27)
    ket_prop = hsprop.apply_tau(ket_prop,pot,[1.0]*3)
    ket_prop = hsprop.apply_coulomb(ket_prop,pot,[1.0])
    ket_prop = hsprop.apply_spinorbit(ket_prop,pot,[1.0]*9)
    # print(ket_prop)
    # print("bracket = ", bra * ket_prop)
    return bra * ket_prop

if __name__ == "__main__":
    method = 'rbm'
    b_old = main(method)
    b_new = main_new(method)
    print('ratio =', b_new/b_old)

