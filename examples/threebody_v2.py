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

    op_1 = sig[0][0]; op_2 = sig[1][0]; op_3 = sig[2][0]

    # exact
    ket_prop = ket.copy()
    force = (op_1 * op_2 * op_3).scale(a3)
    ket_prop = force.scale(- 0.5 * dt).exp().multiply_state(ket_prop)
    b_exact = bra.inner(ket_prop)

    
    if mode==1:    # rbm take 1: use 3b rbm and exact 2b
        ket_prop = ket.copy() # outside loops
        # 3b
        N, C, W, A1, A2 = a3b_factors(0.5 * dt * a3)
        ket_temp = ket_prop.copy().zero()  #right before h loop
        for h in [0.,1.]:
            ket_temp += (ident.scale(-h*C) + (op_1 + op_2 + op_3).scale(A1 - h*W)).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.scale(N)
        # 2b
        ket_prop = (op_1 * op_2 + op_1 * op_3 + op_2 * op_3).scale(A2).exp().multiply_state(ket_prop)
        b_rbm = bra.inner(ket_prop)

    elif mode==2: # rbm take 2: use 3b rbm and 2b rbm (x3)
        ket_prop = ket.copy() # outside loops
        ## 3b
        ket_temp = ket_prop.copy().zero()  #right before h loop
        N3, C3, W3, A1, A2 = a3b_factors(0.5 * dt * a3)
        for h in [0.,1.]:
            ket_temp += (ident.scale(-h*C3) + (op_1 + op_2 + op_3).scale((A1 - h*W3))).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.copy().scale(N3)
        ## 2b
        N2, W2, S2 = a2b_factors(-A2)
        ## i,j
        ket_temp = ket_prop.copy().zero()
        for h in [0.,1.]:
            ket_temp += (op_1 - op_2.scale(S2)).scale(W2*(2*h-1)).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.copy().scale(N2)
        ## i,k
        ket_temp = ket_prop.copy().zero()
        for h in [0.,1.]:
            ket_temp += (op_1 - op_3.scale(S2)).scale(W2*(2*h-1)).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.copy().scale(N2)
        # j,k
        ket_temp = ket_prop.copy().zero()
        for h in [0.,1.]:
            ket_temp += (op_2 - op_3.scale(S2)).scale(W2*(2*h-1)).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.copy().scale(N2)
        b_rbm = bra.inner(ket_prop)
    
    elif mode==3:
        def boltz_f(h, a1, a2, w1, w2, c):
            s2 = a2/abs(a2)
            g1 = op_1.scale(w2*(2*h[1]-1) + w2*(2*h[2]-1) + a1 - h[0]*w1).exp()
            g2 = op_2.scale(w2*s2*(2*h[1]-1) + w2*(2*h[3]-1) + a1 - h[0]*w1).exp()
            g3 = op_3.scale(w2*s2*(2*h[2]-1) + w2*s2*(2*h[3]-1) + a1 - h[0]*w1).exp()
            return (g1 * g2 * g3).scale(cexp(-h[0]*c))
        
        N3, C3, W3, A1, A2 = a3b_factors(0.5 * dt * a3)
        W2 = carctanh(csqrt(ctanh(abs(A2))))
        h_list = itertools.product([0,1], repeat=4)
        ket_prop = ket.copy()
        ket_temp = ket_prop.copy().zero()
        for h_vec in h_list:
            ket_temp += boltz_f(h_vec, A1, A2, W3, W2, C3) * ket_prop.copy()
        ket_prop = ket_temp.scale(N3 * cexp(-3*abs(A2)) * 0.125)
        b_rbm = bra * ket_prop
        
        
    elif mode==4: # sum using propagator class
        prop_3b = HilbertPropagatorRBM(n_particles, dt, isospin)
        h_list = itertools.product([0,1], repeat=4)
        b_array = []
        for h in h_list:
            print(h)
            ket_temp = prop_3b.threebody_sample_full(0.5 * dt * a3, h, op_1, op_2, op_3).multiply_state(ket.copy()).scale(1/8)
            b_array.append(bra * ket_temp)
        b_rbm = np.sum(b_array)
    return b_exact, b_rbm



def three_body_comms():
    n_particles = 3
    sig = [[HilbertOperator(n_particles, isospin=isospin).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    o_0 = sig[0][0] * sig[1][0] * sig[2][0]
    o_1 = sig[0][1] * sig[1][1] * sig[2][1]
    print(np.linalg.norm((o_0*o_1 - o_1*o_0).matrix) )


def compare():
    n_particles = 3
    dt = 0.01
    a3 = 1.0
    seed = 1

    b_exact, b_rbm = gfmc_3b_1d(n_particles, dt, a3, mode=3)
    print("rbm = ", b_rbm)
    print("exact = ", b_exact)
    print("difference = ", b_exact - b_rbm)
    print("error = ", (b_exact - b_rbm)/b_exact)
    
    # print("--------------")
    # b_rbm = afdmc_3b_1d(n_particles, dt, a3)
    # print("rbm = ", b_rbm)
    # print("exact = ", b_exact)
    # print("difference = ", b_exact - b_rbm)
    # print("error = ", (b_exact - b_rbm)/b_exact)  
    


if __name__=="__main__":
    # with Profile() as profile:
    #     compare()
    #     Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats()
    compare()