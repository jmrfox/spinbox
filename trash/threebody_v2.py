from spinbox import *    

from cProfile import Profile
from pstats import SortKey, Stats

log = lambda x: np.log(x, dtype=complex)

isospin = False


def gfmc_3b_1d(n_particles, dt, a3):
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

    ket_prop = ket.copy()
    prop = HilbertPropagatorRBM3(n_particles, dt, isospin)
    rbm = prop.threebody_sample(- 0.5 * dt * a3, )
    b_rbm = bra.inner(prop.multiply_state(ket_prop))

    return b_exact, b_rbm


def afdmc_3b_1d(n_particles, dt, a3):
    # exp( - dt/2 sig_1x sig_2x sig_3x)
    ident = np.identity(2)
    sig = pauli('list')
    if isospin:
        ident = np.identity(4)
        sig = [kronecker_product([np.identity(2), pauli(a)]) for a in [0, 1, 2]]
        tau = [kronecker_product([pauli(a), np.identity(2)]) for a in [0, 1, 2]]
    ket = ProductState(n_particles, isospin=isospin, ketwise=True).randomize(0)
    bra = ProductState(n_particles, isospin=isospin, ketwise=False).randomize(1)
    
    ##### rbm take 1: use 3b rbm and exact 2b
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
    
    return N * (bra.inner(prop_0.multiply_state(ket)) + bra.inner(prop_1.multiply_state(ket)))
    
    
    # ##### rbm take 2: use 3b rbm and 2b rbm (x3)
    # ket_prop = ket.copy() # outside loops
    # ## 3b
    # ket_temp = ket_prop.copy().zero()  #right before h loop
    # N, C, W, A1, A2 = a3b_factors(0.5 * dt * a3)
    # for h in [0.,1.]:
    #     ket_temp += (-h*C*ident + (A1 - h*W)*(sig[0][0] + sig[1][0] + sig[2][0])).exp() * ket_prop
    # ket_prop = N * ket_temp.copy()
    # ## 2b
    # N, W, S = a2b_factors(-A2)
    # ## i,j
    # ket_temp = ket_prop.copy().zero()
    # for h in [0.,1.]:
    #     ket_temp += (W*(2*h-1)*(sig[0][0] - S*sig[1][0])).exp() * ket_prop
    # ket_prop = N * ket_temp.copy()
    # ## i,k
    # ket_temp = ket_prop.copy().zero()
    # for h in [0.,1.]:
    #     ket_temp += (W*(2*h-1)*(sig[0][0] - S*sig[2][0])).exp() * ket_prop
    # ket_prop = N * ket_temp.copy()
    # ## j,k
    # ket_temp = ket_prop.copy().zero()
    # for h in [0.,1.]:
    #     ket_temp += (W*(2*h-1)*(sig[1][0] - S*sig[2][0])).exp() * ket_prop
    # ket_prop = N * ket_temp.copy()
    # b_rbm = bra * ket_prop

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
    g_exact = ExactPropagator(n_particles)
    idx_3b = interaction_indices(n_particles, 3)
    
    ket_prop = ket.copy()
    for i,j,k in idx_3b:
        ket_prop = g_exact.propagator_sigma_3b(dt, asig3b, i, j, k) * ket_prop

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

    b_exact, b_rbm = gfmc_3b_1d(n_particles, dt, a3)
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