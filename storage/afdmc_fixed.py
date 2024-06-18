from spinbox import *

n_particles = 2
dt = 0.001
pairs_ij = interaction_indices(n_particles)

ident = np.identity(4)
sig = [repeated_kronecker_product([np.identity(2), pauli(a)]) for a in [0, 1, 2]]
tau = [repeated_kronecker_product([pauli(a), np.identity(2)]) for a in [0, 1, 2]]

def g_onebody(k, i, opi):
    """exp (- k opi)"""
    out = AFDMCSpinIsospinOperator(n_particles)
    out.op_stack[i] = ccosh(k) * ident - csinh(k) * opi
    return out

def g_gauss_sample(k, x, i: int, j: int, opi, opj, normalize=True):
    """N * exp(x sqrt(-k) opi) exp(x sqrt(-k) opj)
    if normalize then N = exp(k), otherwise N = 1
    """
    if normalize:
        prefactor = cexp(k)
    else:
        prefactor = 1.0
    out = AFDMCSpinIsospinOperator(n_particles)
    arg = csqrt(-k)*x
    out.op_stack[i] = ccosh(arg) * ident + csinh(arg) * opi
    out.op_stack[j] = ccosh(arg) * ident + csinh(arg) * opj
    out.op_stack[i] *= csqrt(prefactor)
    out.op_stack[j] *= csqrt(prefactor)
    return out
  

def g_rbm_sample(k, h, i: int, j: int, opi, opj, normalize=True):
    """ exp( - k opi opj) =  exp(-|k|) exp(W (2h-1) opi) exp(- sgn(k) W (2h-1) opj) """
    if normalize:
        prefactor = cexp(-abs(k))
    else:
        prefactor = 1.0
    out = AFDMCSpinIsospinOperator(n_particles)
    W = carctanh(csqrt(ctanh(np.abs(k))))
    arg = W * (2 * h - 1)
    out.op_stack[i] = ccosh(arg) * ident + csinh(arg) * opi
    out.op_stack[j] = ccosh(arg) * ident - np.sign(k) * csinh(arg) * opj
    out.op_stack[i] *= csqrt(prefactor)
    out.op_stack[j] *= csqrt(prefactor)
    return out

def prop_gauss_fixed(ket, pots, x, normalize=True):
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
                ket = g_gauss_sample(0.5 * dt * asig[a, i, b, j], x, i, j, sig[a], sig[b], normalize=normalize) * ket
    # SIGMA TAU
    for i,j in pairs_ij:    
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    opi = sig[a] @ tau[c]
                    opj = sig[b] @ tau[c]
                    ket = g_gauss_sample(0.5 * dt * asigtau[a, i, b, j], x, i, j, opi, opj, normalize=normalize) * ket
    # TAU
    for i,j in pairs_ij:
        for c in range(3):
            ket = g_gauss_sample(0.5 * dt * atau[i, j], x, i, j, tau[c], tau[c], normalize=normalize) * ket
    # COULOMB
    for i,j in pairs_ij:
        k = 0.125 * dt * vcoul[i, j]
        if normalize:
            ket = ket.spread_scalar_mult(cexp(-k))
        ket = g_onebody(k, i, tau[2]) * ket
        ket = g_onebody(k, j, tau[2]) * ket
        ket = g_gauss_sample(k, x, i, j, tau[2], tau[2], normalize=normalize) * ket
    # LS
    for i in range(n_particles):
        for a in range(3):
            ket = g_onebody( 1.j * gls[a, i], i, sig[a]) * ket
    for i,j in pairs_ij:
        for a in range(3):
            for b in range(3):
                asigls = gls[a, i] * gls[b, j]
                ket = g_gauss_sample(- 0.5 * asigls, x, i, j, sig[a], sig[b], normalize=normalize) * ket
    if normalize:
        trace = cexp(0.5 * np.sum(gls**2))
        ket = ket.spread_scalar_mult(trace)

    return ket
    

def prop_rbm_fixed(ket, pots, h, normalize=True):
    print('RBM')
    asig = pots['asig'] 
    asigtau = pots['asigtau']
    atau = pots['atau']
    vcoul = pots['vcoul']
    # bls = pots['bls']
    # gls = np.sum(pots['bls'], axis = 2)
    gls = pots['gls']

    # SIGMA
    for i,j in pairs_ij:
        for a in range(3):
            for b in range(3):
                ket = g_rbm_sample(0.5 * dt * asig[a, i, b, j], h, i, j, sig[a], sig[b], normalize=normalize) * ket
    # SIGMA TAU
    for i,j in pairs_ij:    
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    opi = sig[a] @ tau[c]
                    opj = sig[b] @ tau[c]
                    ket = g_rbm_sample(0.5 * dt * asigtau[a, i, b, j], h, i, j, opi, opj, normalize=normalize) * ket
    # TAU
    for i,j in pairs_ij:
        for c in range(3):
            ket = g_rbm_sample(0.5 * dt * atau[i, j], h, i, j, tau[c], tau[c], normalize=normalize) * ket
    # COULOMB
    for i,j in pairs_ij:
        k = dt * 0.125 * vcoul[i,j]
        if normalize:
            ket = ket.spread_scalar_mult(cexp(-k))
        ket = g_onebody(k, i, tau[2]) * ket
        ket = g_onebody(k, j, tau[2]) * ket
        ket = g_rbm_sample(k, h, i, j, tau[2], tau[2], normalize=normalize) * ket
    # LS
    for i in range(n_particles):
        for a in range(3):
            ket = g_onebody( 1.j * gls[a, i], i, sig[a]) * ket
    for i,j in pairs_ij:
        for a in range(3):
            for b in range(3):  
                asigls = gls[a, i]* gls[b, j]
                ket = g_rbm_sample(-0.5 * asigls, h, i, j, sig[a], sig[b], normalize=normalize) * ket
    if normalize:
        trace = cexp(0.5 * np.sum(gls**2))
        ket = ket.spread_scalar_mult(trace)
    
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
    ket, pots, _ = load_h2(manybody=False, data_dir = './data/h2/')
    pots['asig'] = 1*pots['asig']
    pots['asigtau'] = 1*pots['asigtau']
    pots['atau'] = 1*pots['atau']
    pots['vcoul'] = 1*pots['vcoul']
    pots['gls'] = 1*pots['gls']
    bra = ket.copy().dagger()

    if method=='hs':
        ket_prop = prop_gauss_fixed(ket.copy(), pots, x=1.0)
    elif method=='rbm':
        ket_prop = prop_rbm_fixed(ket, pots, h=1.0)
        
    return bra * ket_prop

def main_new(method):
    ket = AFDMCSpinIsospinState(n_particles, read_from_file("./data/h2/fort.770",complex=True, shape=(2,4,1)))
    bra = ket.copy().dagger()

    pot = ArgonnePotential(n_particles)
    pot.read_sigma("./data/h2/fort.7701")
    pot.read_sigmatau("./data/h2/fort.7702")
    pot.read_tau("./data/h2/fort.7703")
    pot.read_coulomb("./data/h2/fort.7704")
    pot.read_spinorbit("./data/h2/fort.7705")

    if method=='hs':
        hsprop = AFDMCPropagatorHS(n_particles, dt, include_prefactors=True, mix=False)
    elif method=='rbm':
        hsprop = AFDMCPropagatorRBM(n_particles, dt, include_prefactors=True, mix=False)

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
