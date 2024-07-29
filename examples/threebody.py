from spinbox import *    

from cProfile import Profile
from pstats import SortKey, Stats

log = lambda x: np.log(x, dtype=complex)

isospin = False

def a3b_factors(a3):
    if a3>0:
        x = csqrt(cexp(8*a3) - 1)
        x = csqrt( 2*cexp(4*a3)*( cexp(4*a3)*x + cexp(8*a3) - 1  ) - x)
        x = x + cexp(6*a3) + cexp(2*a3)*csqrt(cexp(8*a3) - 1)
        x = x*2*cexp(2*a3) - 1
        c = 0.5*log(x)
        w = c
        a1 = 0.125*( 6*c - log(cexp(4*c) + 1) + log(2) )
        a2 = 0.125*( 2*c - log(cexp(4*c) + 1) + log(2) )
        top = cexp( 5 * c / 4)
        bottom = 2**(3/8) * csqrt(cexp(2*c) + 1) * (cexp(4*c) + 1)**0.125
        n = top/bottom
    else:
        x = csqrt(1 - cexp(8*a3))
        x = csqrt( 2*(x + 1) - cexp(8*a3) * ( x + 2) )
        x = x + 1 + csqrt(1 - cexp(8*a3))
        c = 0.5 * log(2*cexp(-8*a3)*x - 1)
        w = -c
        a1 = 0.125*( log(0.5*(cexp(4*c) + 1)) - 6*c )
        a2 = 0.125*( 2*c - log(cexp(4*c) + 1) + log(2) )
        top = cexp( c / 4)
        bottom = 2**(3/8) * csqrt(cexp(-2*c) + 1) * (cexp(4*c) + 1)**0.125
        n = top/bottom
    return n, c, w, a1, a2
    

def a2b_factors(k):
    n = cexp(-abs(k))/2.
    w = carctanh(csqrt(ctanh(abs(k))))
    s = k/abs(k)
    return n, w, s



def gfmc_3b_1d(n_particles, dt, a3, mode=1):
    # exp( - dt/2 sig_1x sig_2x sig_3x)
    ident = HilbertOperator(n_particles, isospin=isospin)
    sig = [[HilbertOperator(n_particles, isospin=isospin).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    ket = ProductState(n_particles, isospin=isospin, ketwise=True).randomize(0)
    bra = ProductState(n_particles, isospin=isospin, ketwise=False).randomize(1)
    ket = ket.to_manybody_basis()
    bra = bra.to_manybody_basis()

    # exact
    ket_prop = ket.copy()
    force = sig[0][0].multiply_operator(sig[1][0]).multiply_operator(sig[2][0]).scale(a3)
    ket_prop = force.scale(- 0.5 * dt).exp().multiply_state(ket_prop)
    b_exact = bra.inner(ket_prop)

    
    if mode==1:    # rbm take 1: use 3b rbm and exact 2b
        ket_prop = ket.copy() # outside loops
        # 3b
        N, C, W, A1, A2 = a3b_factors(0.5 * dt * a3)
        ket_temp = ket_prop.copy().zero()  #right before h loop
        for h in [0.,1.]:
            ket_temp += (ident.scale(-h*C) + (sig[0][0] + sig[1][0] + sig[2][0]).scale(A1 - h*W)).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.scale(N)
        # 2b
        ket_prop = (sig[0][0].multiply_operator(sig[1][0]) + sig[0][0].multiply_operator(sig[2][0]) + sig[1][0].multiply_operator(sig[2][0])).scale(A2).exp().multiply_state(ket_prop)
        b_rbm = bra.inner(ket_prop)

    elif mode==2: # rbm take 2: use 3b rbm and 2b rbm (x3)
        ket_prop = ket.copy() # outside loops
        ## 3b
        ket_temp = ket_prop.copy().zero()  #right before h loop
        N, C, W, A1, A2 = a3b_factors(0.5 * dt * a3)
        for h in [0.,1.]:
            ket_temp += (ident.scale(-h*C) + (sig[0][0] + sig[1][0] + sig[2][0]).scale((h*W - A1))).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.copy().scale(N)
        ## 2b
        N, W, S = a2b_factors(-A2)
        ## i,j
        ket_temp = ket_prop.copy().zero()
        for h in [0.,1.]:
            ket_temp += (sig[0][0] - sig[1][0].scale(S)).scale(W*(2*h-1)).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.copy().scale(N)
        ## i,k
        ket_temp = ket_prop.copy().zero()
        for h in [0.,1.]:
            ket_temp += (sig[0][0] - sig[2][0].scale(S)).scale(W*(2*h-1)).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.copy().scale(N)
        ## j,k
        ket_temp = ket_prop.copy().zero()
        for h in [0.,1.]:
            ket_temp += (sig[1][0] - sig[2][0].scale(S)).scale(W*(2*h-1)).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.copy().scale(N)
        b_rbm = bra.inner(ket_prop)
    
    elif mode==3: # sum using propagator class
        ket_prop = ket.copy() # outside loops
        prop_3b = HilbertPropagatorRBM3(n_particles, dt, isospin)
        h_list = itertools.product([0,1], repeat=4)
        for h in h_list:
            print(h)
            ket_prop = prop_3b.threebody_sample_2b(0.5 * dt * a3, h, sig[0][0], sig[1][0], sig[2][0]) * ket_prop
        b_rbm = bra * ket_prop

    return b_exact, b_rbm


def afdmc_3b_1d(n_particles, dt, a3):
    # exp( - dt/2 sig_1x sig_2x sig_3x)
    ident = np.identity(2)
    sig = pauli('list')
    if isospin:
        ident = np.identity(4)
        sig = [repeated_kronecker_product([np.identity(2), pauli(a)]) for a in [0, 1, 2]]
        tau = [repeated_kronecker_product([pauli(a), np.identity(2)]) for a in [0, 1, 2]]
    ket = ProductState(n_particles, isospin=isospin, ketwise=True).randomize(0)
    bra = ProductState(n_particles, isospin=isospin, ketwise=False).randomize(1)
    
    prop_0 = ProductOperator(n_particles, isospin=isospin)
    prop_1 = ProductOperator(n_particles, isospin=isospin)
    
    # 3b
    N, C, W, A1, A2 = a3b_factors(0.5 * dt * a3)
    prop_0.coefficients[0] = (ccosh(A1)*ident + csinh(A1)*sig[0]) @ prop_0.coefficients[0]
    prop_0.coefficients[1] = (ccosh(A1)*ident + csinh(A1)*sig[0]) @ prop_0.coefficients[1]
    prop_0.coefficients[2] = (ccosh(A1)*ident + csinh(A1)*sig[0]) @ prop_0.coefficients[2]
    prop_1.coefficients[0] = cexp(-C/3)*(ccosh(A1-W)*ident + csinh(A1-W)*sig[0]) @ prop_1.coefficients[0]
    prop_1.coefficients[1] = cexp(-C/3)*(ccosh(A1-W)*ident + csinh(A1-W)*sig[0]) @ prop_1.coefficients[1]
    prop_1.coefficients[2] = cexp(-C/3)*(ccosh(A1-W)*ident + csinh(A1-W)*sig[0]) @ prop_1.coefficients[2]
    
    ket_0 = prop_0.multiply_state(ket)
    ket_1 = prop_1.multiply_state(ket)
    
    N, W, S = a2b_factors(-A2)
    ## i,j
    ket_0_temp = ket_0.copy().zero()
    ket_1_temp = ket_1.copy().zero()
    for h in [0.,1.]:
        arg = W*(2*h-1)
        ket_0_temp.coefficients[0] = ccosh(arg) * ket_0_temp.coefficients[0] + csinh(arg) * sig[0] @ ket_0_temp.coefficients[0]
        ket_0_temp.coefficients[1] = ccosh(arg) * ket_0_temp.coefficients[1] - S * csinh(arg) * sig[0] @ ket_0_temp.coefficients[1]
        ket_1_temp.coefficients[0] = ccosh(arg) * ket_1_temp.coefficients[0] + csinh(arg) * sig[0] @ ket_1_temp.coefficients[0]
        ket_1_temp.coefficients[1] = ccosh(arg) * ket_0_temp.coefficients[1] - S * csinh(arg) * sig[0] @ ket_0_temp.coefficients[1]
    ket_prop = N * ket_temp.copy()
    ## i,k
    ket_temp = ket_prop.copy().zero()
    for h in [0.,1.]:
        ket_temp += (W*(2*h-1)*(sig[0][0] - S*sig[2][0])).exp() * ket_prop
    ket_prop = N * ket_temp.copy()
    ## j,k
    ket_temp = ket_prop.copy().zero()
    for h in [0.,1.]:
        ket_temp += (W*(2*h-1)*(sig[1][0] - S*sig[2][0])).exp() * ket_prop
    ket_prop = N * ket_temp.copy()
    b_rbm = bra * ket_prop

    return b_exact, b_rbm


def gfmc_3bprop(n_particles, dt, seed):
    seeder = itertools.count(seed, 1)
    ident = HilbertOperator(n_particles, isospin=isospin)
    sig = [[HilbertOperator(n_particles, isospin=isospin).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    if isospin:
        tau = [[HilbertOperator(n_particles, isospin=isospin).apply_tau(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    ket = ProductState(n_particles, isospin=isospin, ketwise=True).randomize(seed=next(seeder)).to_manybody_basis()
    bra = ket.copy().dagger()
    
    asig3b = ThreeBodyCoupling(n_particles).generate(scale=100, seed=next(seeder))
    g_exact = ExactGFMC(n_particles)
    idx_3b = interaction_indices(n_particles, 3)
    
    ket_prop = ket.copy()
    for i,j,k in idx_3b:
        ket_prop = g_exact.g_pade_sigma_3b(dt, asig3b, i, j, k) * ket_prop

    b_exact = bra * ket_prop 

    ket_prop = ket.copy()
    for i,j,k in idx_3b:
        print(f"--- particles: {i} {j} {k}")
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    print(f"dimension: {a} {b} {c}")
                    ket_temp = ket_prop.copy().zero()
                    for h in [0.,1.]:
                        N, C, W, A1, A2 = a3b_factors(0.5 * dt * asig3b[a,i,b,j,c,k])
                        ket_temp += (-h*C*ident + (A1 - h*W)*(sig[i][a] + sig[j][b] + sig[k][c])).exp() * ket_prop
                    ket_prop = N * ket_temp.copy()
                    # ket_prop = (A2*(sig[i][a]*sig[j][b] + sig[i][a]*sig[k][c] + sig[j][b]*sig[k][c])).exp() * ket_prop
                    N, W, S = a2b_factors(-A2)
                    ## iajb
                    ket_temp = ket_prop.copy().zero()
                    for h in [0.,1.]:
                        ket_temp += (W*(2*h-1)*(sig[i][a] - S*sig[j][b])).exp() * ket_prop
                    ket_prop = N * ket_temp.copy()
                    ## iakc
                    ket_temp = ket_prop.copy().zero()
                    for h in [0.,1.]:
                        ket_temp += (W*(2*h-1)*(sig[i][a] - S*sig[k][c])).exp() * ket_prop
                    ket_prop = N * ket_temp.copy()
                    ## jbkc
                    ket_temp = ket_prop.copy().zero()
                    for h in [0.,1.]:
                        ket_temp += (W*(2*h-1)*(sig[j][b] - S*sig[k][c])).exp() * ket_prop
                    ket_prop = N * ket_temp.copy()

    b_rbm = bra * ket_prop
    return b_exact, b_rbm    


def three_body_comms():
    n_particles = 3
    sig = [[HilbertOperator(n_particles, isospin=isospin).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    # tau = [[GFMCSpinIsospinOperator(n_particles).apply_tau(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    o_0 = sig[0][0] * sig[1][0] * sig[2][0]
    o_1 = sig[0][1] * sig[1][1] * sig[2][1]
    print(np.linalg.norm((o_0*o_1 - o_1*o_0).matrix) )


def compare():
    n_particles = 3
    dt = 0.001
    a3 = 1.0
    seed = 1

    b_exact, b_rbm = gfmc_3b_1d(n_particles, dt, a3, mode=3)
    # b_exact, b_rbm = gfmc_3bprop(n_particles, dt, seed)
    print("rbm = ", b_rbm)
    print("exact = ", b_exact)
    print("difference = ", b_rbm - b_exact)
    print("error = ", (b_rbm - b_exact)/b_exact)  
    print("error by ratio = ", abs(1 - abs(b_rbm)/abs(b_exact)))
    print("--------------")
    # b_rbm = afdmc_3b_1d(n_particles, dt, a3)
    # print("rbm = ", b_rbm)
    # print("exact = ", b_exact)
    # print("difference = ", b_rbm - b_exact)
    # print("error = ", (b_rbm - b_exact)/b_exact)  
    # print("error by ratio = ", abs(1 - abs(b_rbm)/abs(b_exact)))
    


if __name__=="__main__":
    # with Profile() as profile:
    #     compare()
    #     Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats()
    compare()