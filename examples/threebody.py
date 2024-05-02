from quap import *    

log = lambda x: np.log(x, dtype=complex)


        

def a3b_factors(a3):
    if a3>0:
        x = csqrt(cexp(8*a3) - 1)
        x = csqrt( 2*cexp(4*a3)*( cexp(4*a3)*x + cexp(8*a3) - 1  ) - x)
        x = x + cexp(6*a3) + cexp(2*a3)*csqrt(cexp(8*a3) - 1)
        x = x*2*cexp(2*a3) - 1
        c = 0.5*log(x)
        a2 = 0.125*( 2*c - log(cexp(4*c) + 1) + log(2) )
        a1 = 0.125*( 6*c - log(cexp(4*c) + 1) + log(2) )
        top = cexp( 5 * c / 4)
        bottom = 2**(3/8) * csqrt(cexp(2*c) + 1) * (cexp(4*c) + 1)**0.125
        n = top/bottom
    else:
        x = csqrt(1 - cexp(8*a3))
        x = csqrt( 2*(x + 1) - cexp(8*a3) * ( x + 2) )
        x = x + 1 + csqrt(1 - cexp(8*a3))
        c = 0.5 * log(2*cexp(-8*a3)*x - 1)
        a2 = 0.125*( 2*c - log(cexp(4*c) + 1) + log(2) )
        a1 = 0.125*( log(0.5*(cexp(4*c) + 1)) - 6*c )
        top = cexp( c / 4)
        bottom = 2**(3/8) * csqrt(cexp(-2*c) + 1) * (cexp(4*c) + 1)**0.125
        n = top/bottom
    return n, c, a1, a2


def gfmc_3bprop(n_particles, dt, seed):
    seeder = itertools.count(seed, 1)
    sig = [[GFMCSpinIsospinOperator(n_particles).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    tau = [[GFMCSpinIsospinOperator(n_particles).apply_tau(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    ket = AFDMCSpinIsospinState(n_particles, np.zeros(shape=(n_particles, 4, 1)), ketwise=True).randomize(seed=next(seeder)).to_manybody_basis()
    bra = ket.copy().dagger()
    
    asig3b = ThreeBodyCoupling(n_particles).generate(scale=1, seed=next(seeder))
    g_exact = ExactGFMC(n_particles)
    idx_3b = interaction_indices(n_particles, 3)
    
    ket_prop = ket.copy()
    for i,j,k in idx_3b:
        ket_prop = g_exact.g_pade_sig_3b(dt, asig3b, i, j, k) * ket_prop

    b_exact = bra * ket_prop 

    # nnn rbm
    ket_prop = ket.copy()
    for i,j,k in idx_3b:
        # first, apply the one-body parts
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    n, c, a1, a2 = a3b_factors(0.5 * dt * asig3b[a,i,b,j,c,k])
                    ket_prop = cexp(c*h)
                    
    

    return b_exact

if __name__=="__main__":
    n_particles = 3
    dt = 0.001
    seed = 0
    b = gfmc_3bprop(n_particles, dt, seed)
    print("result = ", b)