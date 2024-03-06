import nuctest as nt
from quap import *

ident = np.identity(4)
sig = [repeated_kronecker_product([np.identity(2), pauli(a)]) for a in [0, 1, 2]]
tau = [repeated_kronecker_product([pauli(a), np.identity(2)]) for a in [0, 1, 2]]

def load_ket(filename):
    c = read_coeffs(filename)
    sp = OneBodyBasisSpinIsospinState(nt.n_particles, 'ket', c.reshape(-1, 1))
    return sp

def g_onebody(k, i, opi):
    """exp (- k opi)"""
    out = OneBodyBasisSpinIsospinOperator(nt.n_particles)
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
    out = OneBodyBasisSpinIsospinOperator(nt.n_particles)
    arg = csqrt(-k)*x
    print('arg = ', arg)
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
    out = OneBodyBasisSpinIsospinOperator(nt.n_particles)
    W = carctanh(csqrt(ctanh(np.abs(k))))
    arg = W * (2 * h - 1)
    out.op_stack[i] = ccosh(arg) * ident + csinh(arg) * opi
    out.op_stack[j] = ccosh(arg) * ident - np.sign(k) * csinh(arg) * opj
    out.op_stack[i] *= csqrt(prefactor)
    out.op_stack[j] *= csqrt(prefactor)
    return out

def prop_gauss_fixed(ket, pots, x):
    print('GAUSS')
    asig = pots['asig'] 
    asigtau = pots['asigtau']
    atau = pots['atau']
    vcoul = pots['vcoul']
    gls = pots['gls']

    # SIGMA
    for i,j in nt.pairs_ij:
        for a in range(3):
            for b in range(3):
                ket = g_gauss_sample(0.5 * nt.dt * asig[a, i, b, j], x, i, j, sig[a], sig[b]) * ket
    # SIGMA TAU
    for i,j in nt.pairs_ij:    
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    opi = sig[a] @ tau[c]
                    opj = sig[b] @ tau[c]
                    ket = g_gauss_sample(0.5 * nt.dt * asigtau[a, i, b, j], x, i, j, opi, opj) * ket
    # TAU
    for i,j in nt.pairs_ij:
        for c in range(3):
            ket = g_gauss_sample(0.5 * nt.dt * atau[i, j], x, i, j, tau[c], tau[c]) * ket
    # COULOMB
    for i,j in nt.pairs_ij:
        ket = g_onebody(0.5 * nt.dt * vcoul[i, j], i, tau[2]) * ket
        ket = g_onebody(0.5 * nt.dt * vcoul[i, j], j, tau[2]) * ket
        ket = g_gauss_sample(0.125 * nt.dt * vcoul[i, j], x, i, j, tau[2], tau[2]) * ket
    # LS
    for i in range(nt.n_particles):
        for a in range(3):
            ket = g_onebody(0.5 * nt.dt * gls[a, i], i, sig[a]) * ket
    for i,j in nt.pairs_ij:
        for a in range(3):
            for b in range(3):
                asigls = gls[a, i]* gls[b, j]
                ket = g_gauss_sample(- 0.5 * asigls, x, i, j, sig[a], sig[b]) * ket
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
    for i,j in nt.pairs_ij:
        for a in range(3):
            for b in range(3):
                ket = g_rbm_sample(0.5 * nt.dt * asig[a, i, b, j], h, i, j, sig[a], sig[b], normalize=normalize) * ket
    # SIGMA TAU
    for i,j in nt.pairs_ij:    
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    opi = sig[a] @ tau[c]
                    opj = sig[b] @ tau[c]
                    ket = g_rbm_sample(0.5 * nt.dt * asigtau[a, i, b, j], h, i, j, opi, opj, normalize=normalize) * ket
    # TAU
    for i,j in nt.pairs_ij:
        for c in range(3):
            ket = g_rbm_sample(0.5 * nt.dt * atau[i, j], h, i, j, tau[c], tau[c], normalize=normalize) * ket
    # COULOMB
    for i,j in nt.pairs_ij:
        if normalize:
            ket = ket.spread_scalar_mult(cexp(-nt.dt * 0.125 * vcoul[i,j]))
        ket = g_onebody(0.125 * nt.dt * vcoul[i, j], i, tau[2]) * ket
        ket = g_onebody(0.125 * nt.dt * vcoul[i, j], j, tau[2]) * ket
        ket = g_rbm_sample(0.125 * nt.dt * vcoul[i, j], h, i, j, tau[2], tau[2], normalize=normalize) * ket
    # LS
    for i in range(nt.n_particles):
        for a in range(3):
            ket = g_onebody(1.j * gls[a, i], i, sig[a]) * ket
    for i,j in nt.pairs_ij:
        for a in range(3):
            for b in range(3):
                asigls = gls[a, i]* gls[b, j]
                ket = g_rbm_sample(-0.5 * asigls, h, i, j, sig[a], sig[b], normalize=normalize) * ket
    if normalize:
        trace = cexp(0.5 * np.sum(gls**2))
        ket = ket.spread_scalar_mult(trace)
    
    return ket


def main():
    # bra, ket = nt.make_test_states(manybody=False)
    # pots = nt.make_all_potentials(mode='test')

    ket, pots, ket_ref = nt.load_h2(data_dir = './data/h2/')
    pots['asig'] = 1*pots['asig']
    pots['asigtau'] = 1*pots['asigtau']
    pots['atau'] = 1*pots['atau']
    pots['vcoul'] = 0*pots['vcoul']
    pots['gls'] = 1*pots['gls']
    bra = ket.dagger()
    
    # print("INITIAL KET\n", ket)
    ket = prop_rbm_fixed(ket, pots, h=1.0, normalize=False)
    print("FINAL KET\n", ket)
    # print("bracket = ", bra * ket)
    # print("REFERENCE\n", ket_ref)



if __name__ == "__main__":
    print('OBB')
    main()
