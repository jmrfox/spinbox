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

def g_gauss_sample(k, x, i: int, j: int, opi, opj, prefactor=None):
    """exp(k) exp(x sqrt(-k) opi) exp(x sqrt(-k) opj)"""
    if prefactor is None:
        prefactor = cexp(k)
        print('Standard norm')
    out = OneBodyBasisSpinIsospinOperator(nt.n_particles)
    arg = 1.j*csqrt(k)*x
    out.op_stack[i] = ccosh(arg) * ident + csinh(arg) * opi
    out.op_stack[j] = ccosh(arg) * ident + csinh(arg) * opj
    out.op_stack[i] *= csqrt(prefactor)
    out.op_stack[j] *= csqrt(prefactor)
    # out = out.spread_scalar_mult(prefactor)
    return out


def g_rbm_sample(k, h, i: int, j: int, opi, opj, prefactor=None):
    """ exp(-|k|) exp(W (2h-1) opi) exp(- sgn(k) W (2h-1) opj) """
    if prefactor is None:
        prefactor = cexp(-abs(k))
    out = OneBodyBasisSpinIsospinOperator(nt.n_particles)
    W = carctanh(csqrt(ctanh(np.abs(k))))
    arg = W * (2 * h - 1)
    out.op_stack[i] = ccosh(arg) * ident + csinh(arg) * opi
    out.op_stack[j] = ccosh(arg) * ident - np.sign(k) * csinh(arg) * opj
    out.op_stack[i] *= csqrt(prefactor)
    out.op_stack[j] *= csqrt(prefactor)
    # out = out.spread_scalar_mult(norm)
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
    

def prop_rbm_fixed(ket, pots, h):
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
                ket = g_rbm_sample(nt.dt, asig[a, i, b, j], h, i, j, sig[a], sig[b]) * ket
    # SIGMA TAU
    for i,j in nt.pairs_ij:    
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    opi = sig[a] @ tau[c]
                    opj = sig[b] @ tau[c]
                    ket = g_rbm_sample(nt.dt, asigtau[a, i, b, j], h, i, j, opi, opj) * ket
    # TAU
    for i,j in nt.pairs_ij:
        for c in range(3):
            ket = g_rbm_sample(nt.dt, atau[i, j], h, i, j, tau[c], tau[c]) * ket
    # COULOMB
    for i,j in nt.pairs_ij:
        ket = g_coulomb_onebody(nt.dt, vcoul[i, j], i) * g_coulomb_onebody(nt.dt, vcoul[i, j], j) * ket
        ket = g_rbm_sample(nt.dt, 0.25 * vcoul[i, j], h, i, j, tau[2], tau[2]) * ket
    # LS
    for i in range(nt.n_particles):
        for a in range(3):
            ket = g_ls_onebody(gls[a, i], i, a) * ket
    for i,j in nt.pairs_ij:
        for a in range(3):
            for b in range(3):
                asigls = gls[a, i]* gls[b, j]
                ket = g_rbm_sample(1, - asigls, h, i, j, sig[a], sig[b]) * ket
    trace = cexp( 0.5 * np.sum(gls**2))
    ket = trace * ket
    
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
    for i,j in nt.pairs_ij:
        for a in range(3):
            for b in range(3):
                norm = cexp(-nt.dt * 0.5 * np.abs(asig[a, i, b, j]))
                ket = g_rbm_sample(nt.dt, asig[a, i, b, j], h, i, j, sig[a], sig[b]).spread_scalar_mult(1/norm) * ket
    # SIGMA TAU
    for i,j in nt.pairs_ij:
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    norm = cexp(-nt.dt * 0.5 * np.abs(asigtau[a, i, b, j]))
                    opi = sig[a] @ tau[c]
                    opj = sig[b] @ tau[c]
                    ket = g_rbm_sample(nt.dt, asigtau[a, i, b, j], h, i, j, opi, opj).spread_scalar_mult(1/norm) * ket
    # TAU
    for i,j in nt.pairs_ij:
        for c in range(3):
            norm = cexp(-nt.dt * 0.5 * np.abs(atau[i, j]))
            ket = g_rbm_sample(nt.dt, atau[i, j], h, i, j, tau[c], tau[c]).spread_scalar_mult(1/norm) * ket
    # COULOMB
    for i,j in nt.pairs_ij:
        norm_1b = cexp(-nt.dt * 0.125 * vcoul[i, j])
        norm_rbm = cexp(-nt.dt * 0.125 * np.abs(vcoul[i, j]))
        # ket = g_coulomb_onebody(nt.dt, vcoul[i, j], i).spread_scalar_mult(1/norm_1b) * g_coulomb_onebody(nt.dt, vcoul[i, j], j).spread_scalar_mult(1/norm_1b) * ket
        ket = g_coulomb_pair(nt.dt, vcoul[i, j], i, j).spread_scalar_mult(1/norm_1b) * ket
        ket = g_rbm_sample(nt.dt, 0.25 * vcoul[i, j], h, i, j, tau[2], tau[2]).spread_scalar_mult(1/norm_rbm) * ket
    # LS
    for i in range(nt.n_particles):
        for a in range(3):
            ket = g_ls_onebody(gls[a, i], i, a) * ket
    for i,j in nt.pairs_ij:
        for a in range(3):
            for b in range(3):
                asigls = gls[a, i]* gls[b, j]
                norm = cexp(0.5 * np.abs(asigls))
                ket = g_rbm_sample(1, -asigls, h, i, j, sig[a], sig[b]).spread_scalar_mult(1/norm) * ket

    return ket

def main():
    # bra, ket = nt.make_test_states(manybody=False)
    # pots = nt.make_all_potentials(mode='test')

    ket, pots, ket_ref = nt.load_h2(data_dir = './data/h2/')
    pots['asig'] = 1*pots['asig']
    pots['asigtau'] = 0*pots['asigtau']
    pots['atau'] = 0*pots['atau']
    pots['vcoul'] = 0*pots['vcoul']
    pots['gls'] = 0*pots['gls']
    bra = ket.dagger()
    
    print("INITIAL KET\n", ket)
    ket = prop_gauss_fixed(ket, pots, x=1.0)
    # ket = prop_rbm_fixed_unnorm(ket, pots, h=1.0)
    print("FINAL KET\n", ket)
    print("bracket = ", bra * ket)
    # print("REFERENCE\n", ket_ref)

def debug():
    ket, pots, ket_ref = nt.load_h2(data_dir = './data/h2/')
    pots['asig'] = 1*pots['asig']
    pots['asigtau'] = 0*pots['asigtau']
    pots['atau'] = 0*pots['atau']
    pots['vcoul'] = 0*pots['vcoul']
    pots['gls'] = 0*pots['gls']
    bra = ket.dagger()
    
    print("INITIAL KET\n", ket)

    ket = g_gauss_sample(1,1,0,1,sig[0],sig[0]) * ket

    print("FINAL KET\n", ket)
    print("bracket = ", bra * ket)


if __name__ == "__main__":
    debug()