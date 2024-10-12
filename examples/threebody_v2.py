from spinbox import *    

from cProfile import Profile
from pstats import SortKey, Stats

log = lambda x: np.log(x, dtype=complex)
cabs = lambda x: np.abs(np.real(x)) + 1j * np.abs(np.imag(x))

isospin = True

i,j,k = 0,1,2
a,b,c = 0,0,0

seed = 1

dt = 0.002

def a3b_factors(coupling):
    # a3 must be fully imaginary
    # e.g. - delta tau A / 2 = - i dt A / 2
    a3 = dt * 0.5 * coupling
    if coupling>0:
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
    

def a2b_factors(a2):
    # a2 must be fully imaginary
    # e.g. - delta tau A / 2 = - i dt A / 2
    n = cexp(-cabs(a2))/2.
    w = carctanh(csqrt(ctanh(cabs(a2))))
    s = a2/cabs(a2)
    return n, w, s



def gfmc_3b_1d(n_particles, dt, coupling, mode=1):
    # exp( - dt/2 sig_1x sig_2x sig_3x)
    ident = HilbertOperator(n_particles, isospin=isospin)
    sig = [[HilbertOperator(n_particles, isospin=isospin).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    tau = [[HilbertOperator(n_particles, isospin=isospin).apply_tau(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    ket = ProductState(n_particles, isospin=isospin, ketwise=True).random(0)
    bra = ProductState(n_particles, isospin=isospin, ketwise=False).random(1)
    ket = ket.to_full_basis()
    bra = bra.to_full_basis()

    op_i = sig[i][a] * tau[i][a]
    op_j = sig[j][b] * tau[j][b]
    op_k = sig[k][c] * tau[k][c]

    # exact
    ket_prop = ket.copy()
    force = (op_i * op_j * op_k).scale(coupling)
    ket_prop = force.scale(- dt * 0.5).exp().multiply_state(ket_prop)
    b_exact = bra.inner(ket_prop)

    
    if mode==1:    # rbm take 1: use 3b rbm and exact 2b
        ket_prop = ket.copy() # outside loops
        # 3b
        N, C, W, A1, A2 = a3b_factors(coupling)
        ket_temp = ket_prop.copy().zero()  #right before h loop
        for h in [0.,1.]:
            ket_temp += (ident.scale(-h*C) + (op_i + op_j + op_k).scale(A1 - h*W)).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.scale(N)
        # 2b
        ket_prop = (op_i * op_j + op_i * op_k + op_j * op_k).scale(A2).exp().multiply_state(ket_prop)
        b_rbm = bra.inner(ket_prop)

    elif mode==2: # rbm take 2: use 3b rbm and 2b rbm (x3)
        ket_prop = ket.copy() # outside loops
        ## 3b
        ket_temp = ket_prop.copy().zero()  #right before h loop
        N3, C3, W3, A1, A2 = a3b_factors(coupling)
        print(A2)
        for h in [0.,1.]:
            ket_temp += (ident.scale(-h*C3) + (op_i + op_j + op_k).scale((A1 - h*W3))).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.copy().scale(N3)
        ## 2b
        N2, W2, S2 = a2b_factors(-A2)
        ## i,j
        ket_temp = ket_prop.copy().zero()
        for h in [0.,1.]:
            ket_temp += (op_i - op_j.scale(S2)).scale(W2*(2*h-1)).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.copy().scale(N2)
        ## i,k
        ket_temp = ket_prop.copy().zero()
        for h in [0.,1.]:
            ket_temp += (op_i - op_k.scale(S2)).scale(W2*(2*h-1)).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.copy().scale(N2)
        # j,k
        ket_temp = ket_prop.copy().zero()
        for h in [0.,1.]:
            ket_temp += (op_j - op_k.scale(S2)).scale(W2*(2*h-1)).exp().multiply_state(ket_prop)
        ket_prop = ket_temp.copy().scale(N2)
        b_rbm = bra.inner(ket_prop)
    
    elif mode==3:
        def boltz_f(h, a1, a2, w1, w2, c):
            s2 = a2/cabs(a2)            
            g1 = op_i.scale(w2*(2*h[1]-1) + w2*(2*h[2]-1) + a1 - h[0]*w1).exp()
            g2 = op_j.scale(w2*s2*(2*h[1]-1) + w2*(2*h[3]-1) + a1 - h[0]*w1).exp()
            g3 = op_k.scale(w2*s2*(2*h[2]-1) + w2*s2*(2*h[3]-1) + a1 - h[0]*w1).exp()
            out = g1 * g2 * g3
            out = out.scale(cexp(-h[0]*c))
            return out
        
        N3, C3, W3, A1, A2 = a3b_factors(coupling)
        W2 = carctanh(csqrt(ctanh(cabs(A2))))
        h_list = itertools.product([0,1], repeat=4)
        ket_prop = ket.copy().zero()
        for h_vec in h_list:
            op = boltz_f(h_vec, A1, A2, W3, W2, C3)
            ket_prop += op * ket.copy()
        ket_prop = ket_prop.scale(N3 * cexp(-3*cabs(A2)) * 0.125)
        b_rbm = bra * ket_prop
        
        
    elif mode==4: # sum using propagator class
        prop = HilbertPropagatorRBM(n_particles, dt, isospin)
        h_list = itertools.product([0,1], repeat=4)
        b_array = []
        for h in h_list:
            b_array.append(bra * prop.threebody_sample(coupling, h, op_i, op_j, op_k) * ket)
        b_rbm = np.sum(b_array) / 16   # correct for sampling normalization 1/2**4
    
    elif mode==5: # "sampling" but the aux field cycles through all 16 values 
        prop = HilbertPropagatorRBM(n_particles, dt, isospin)
        
        h_list = itertools.product([0,1], repeat=4)
        h_list = 100*list(h_list)
        
        # n_samples = 10000
        # h_list = np.random.randint(0,2,size=(n_samples, 4))
        
        b_array = []
        for h in tqdm(h_list):
            b_array.append(bra * prop.threebody_sample(coupling, h, op_i, op_j, op_k) * ket)
        b_rbm = np.mean(b_array) 
    
    elif mode==6: # sample, no balance
        n_samples = 10000
        prop = HilbertPropagatorRBM(n_particles, dt, isospin)
        rng = np.random.default_rng(seed=0)
        h_list = rng.integers(0,2,size=(n_samples, 4))
        b_array = np.zeros(n_samples, dtype=complex)
        for ih,h in enumerate(tqdm(h_list)):
            b_array[ih:ih+1] = bra * prop.threebody_sample(coupling, h, op_i, op_j, op_k) * ket
        b_rbm = np.mean(b_array)
    
    elif mode==7: # sample, balance h and 1-h
        n_samples = 20000
        prop = HilbertPropagatorRBM(n_particles, dt, isospin)
        rng = np.random.default_rng(seed=0)
        h_list = rng.integers(0,2,size=(n_samples, 4))
        b_array = np.zeros(2*n_samples, dtype=complex)
        for ih,h in enumerate(tqdm(h_list)):
            b_array[2*ih:2*ih+1] = bra * prop.threebody_sample(coupling, h, op_i, op_j, op_k) * ket
            b_array[2*ih+1:2*ih+2] = bra * prop.threebody_sample(coupling, 1-h, op_i, op_j, op_k) * ket
        b_rbm = np.mean(b_array)
    return b_exact, b_rbm


def afdmc_3b_1d(n_particles, dt, coupling, mode=1):
    # exp( - dt/2 sig_1x sig_2x sig_3x)
    sig = [kronecker_product([np.identity(2), pauli(a)]) for a in [0, 1, 2]]
    tau = [kronecker_product([pauli(a), np.identity(2)]) for a in [0, 1, 2]]
    ket = ProductState(n_particles, isospin=isospin, ketwise=True).random(0)
    bra = ProductState(n_particles, isospin=isospin, ketwise=False).random(1)
    
    op_i = sig[a] @ tau[a]
    op_j = sig[b] @ tau[b]
    op_k = sig[c] @ tau[c]

    
    if mode==3:
        def boltz_f(h, a1, a2, w1, w2, c):
            s2 = a2/abs(a2)
            arg_i = (w2*(2*h[1]-1) + w2*(2*h[2]-1) + a1 - h[0]*w1)
            arg_j = (w2*s2*(2*h[1]-1) + w2*(2*h[3]-1) + a1 - h[0]*w1)
            arg_k = (w2*s2*(2*h[2]-1) + w2*s2*(2*h[3]-1) + a1 - h[0]*w1)
            out = ProductOperator(n_particles, isospin=isospin)
            out.coefficients[i] = ccosh(arg_i) * out.coefficients[i] + csinh(arg_i) * op_i @ out.coefficients[i]
            out.coefficients[j] = ccosh(arg_j) * out.coefficients[j] + csinh(arg_j) * op_j @ out.coefficients[j]
            out.coefficients[k] = ccosh(arg_k) * out.coefficients[k] + csinh(arg_k) * op_k @ out.coefficients[k]
            out = out.scale_all(cexp(-h[0]*c))
            return out
        
        N3, C3, W3, A1, A2 = a3b_factors(coupling)
        W2 = carctanh(csqrt(ctanh(abs(A2))))
        h_list = itertools.product([0,1], repeat=4)
        b_rbm = 0.
        for h_vec in h_list:
            op = boltz_f(h_vec, A1, A2, W3, W2, C3)
            b_rbm += bra * (op * ket.copy())
        b_rbm *= N3 * cexp(-3*abs(A2)) * 0.125
        
    elif mode==4: # sum using propagator class
        prop = ProductPropagatorRBM(n_particles, dt, isospin)
        h_list = itertools.product([0,1], repeat=4)
        b_array = []
        for h in h_list:
            ket_temp = prop.threebody_sample(dt * 0.5 * coupling, h, i, j, k, op_i, op_j, op_k).multiply_state(ket.copy())
            b_array.append(bra * ket_temp)
        b_rbm = np.sum(b_array) / 16   # correct for sampling normalization 1/2**4
    
    elif mode==5: # sample using propagator class
        n_samples = 1000
        prop = ProductPropagatorRBM(n_particles, dt, isospin)
        rng = np.random.default_rng(seed=0)
        h_list = rng.integers(0,2,size=(n_samples, 4))
        b_array = []
        for h in h_list:
            ket_temp = prop.threebody_sample(dt * 0.5 * coupling, h, i, j, k, op_i, op_j, op_k).multiply_state(ket.copy())
            b_array.append(bra * ket_temp)
            h_flip = 1-h
            ket_temp = prop.threebody_sample(dt * 0.5 * coupling, h_flip, i, j, k, op_i, op_j, op_k).multiply_state(ket.copy())
            b_array.append(bra * ket_temp)
        b_rbm = np.mean(b_array)

    return b_rbm


def three_body_comms():
    n_particles = 3
    sig = [[HilbertOperator(n_particles, isospin=isospin).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    o_0 = sig[0][0] * sig[1][0] * sig[2][0]
    o_1 = sig[0][1] * sig[1][1] * sig[2][1]
    print(np.linalg.norm((o_0*o_1 - o_1*o_0).matrix) )


def compare():
    n_particles = 3
    coupling = -3.14

    b_exact, b_rbm = gfmc_3b_1d(n_particles, dt, coupling, mode=5)
    print("rbm = ", b_rbm)
    print("exact = ", b_exact)
    print("difference = ", b_exact - b_rbm)
    print("error = ", abs((b_exact - b_rbm)/b_exact) )
    
    print("--------------")
    b_rbm = afdmc_3b_1d(n_particles, dt, coupling, mode=5)
    print("rbm = ", b_rbm)
    print("exact = ", b_exact)
    print("difference = ", b_exact - b_rbm)
    print("error = ", abs((b_exact - b_rbm)/b_exact) )  
    


if __name__=="__main__":
    n_particles = 3
    coupling = -3.14
    b_exact, b_rbm = gfmc_3b_1d(n_particles, dt, coupling, mode=6)
    print("rbm = ", b_rbm)
    print("exact = ", b_exact)
    print("difference = ", b_exact - b_rbm)
    print("error = ", abs((b_exact - b_rbm)/b_exact) )
    
    # compare()