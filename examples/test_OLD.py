from spinbox import *

# need to actually make this an independent test script

error_threshold = 10**-12

def test_overlaps(isospin):

    def exam(s0, s1):
        out = []
        out.append( s0.dagger().inner(s0)  )
        out.append( s1.dagger().inner(s1)  )
        out.append( s0.dagger().inner(s1)  )
        return out


    s0 = ProductState(n_particles, isospin=isospin).randomize(100).to_manybody_basis()
    s1 = ProductState(n_particles, isospin=isospin).randomize(101).to_manybody_basis()
    hilbert_list = exam(s0, s1)
    s0 = ProductState(n_particles, isospin=isospin).randomize(100)
    s1 = ProductState(n_particles, isospin=isospin).randomize(101)
    product_list = exam(s0, s1)
    err = 0.
    for x0, x1 in zip(hilbert_list, product_list):
        err += np.sum(abs(x0 - x1))
    return err < error_threshold

def test_overlaps_unsafe(isospin):

    def exam(s0, s1):
        out = []
        out.append( s0.dagger() * s0  )
        out.append( s1.dagger() * s1  )
        out.append( s0.dagger() * s1  )
        return out

    s0 = ProductState(n_particles, isospin=isospin, safe=False).randomize(100).to_manybody_basis()
    s1 = ProductState(n_particles, isospin=isospin, safe=False).randomize(101).to_manybody_basis()
    hilbert_list = exam(s0, s1)
    s0 = ProductState(n_particles, isospin=isospin, safe=False).randomize(100)
    s1 = ProductState(n_particles, isospin=isospin, safe=False).randomize(101)
    product_list = exam(s0, s1)
    err = 0.
    for x0, x1 in zip(hilbert_list, product_list):
        err += np.sum(abs(x0 - x1))
    return err < error_threshold


def test_hilbert_basics(n_particles, isospin):
    print('HILBERT RANDOM STATES')
    # s0 = ProductState(n_particles, isospin=isospin).randomize(100).to_manybody_basis()
    # s1 = ProductState(n_particles, isospin=isospin).randomize(101).to_manybody_basis()
    s0 = HilbertState(n_particles, isospin=isospin).randomize(100)
    s1 = HilbertState(n_particles, isospin=isospin).randomize(101)
    print("|0> = \n", s0)
    print("|1> = \n", s1)
    print('HILBERT INNER PRODUCTS')
    print("<0|0> = \n", s0.dagger().inner(s0)  )
    print("<1|1> = \n", s1.dagger().inner(s1)  )
    print("<0|1> = \n", s0.dagger().inner(s1) )
    print('HILBERT OUTER PRODUCTS')
    print("|0><0| = \n", s0.outer(s0.dagger()) )
    print("|1><1| = \n", s1.outer(s1.dagger()) )
    print("|0><1| = \n", s0.outer(s1.dagger()) )
    print('DONE TESTING HILBERT STATES')

def test_hilbert_operators(n_particles, isospin):
    s0 = HilbertState(n_particles, isospin=isospin).randomize(100)
    s1 = HilbertState(n_particles, isospin=isospin).randomize(101)
    sig = [[HilbertOperator(n_particles, isospin=isospin).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    for i in range(n_particles):
        for a in range(3):
            print('sigma', i, a)
            print(sig[i][a])
            
    print("<0|sig(0,0)|0> = \n", s0.dagger().inner(sig[0][0].multiply_state(s0)))
    print("<1|sig(0,0)|1> = \n", s1.dagger().inner(sig[0][0].multiply_state(s1)))
    print("<0|sig(0,0)|1> = \n", s0.dagger().inner(sig[0][0].multiply_state(s1)))
    
    print("<0|sig(0,0)sig(1,0)|0> = \n", s0.dagger().inner(sig[0][0].multiply_operator(sig[1][0]).multiply_state(s0)))
    
    print('DONE TESTING HILBERT OPERATORS')


def test_product_basics(n_particles, isospin):
    print('PRODUCT RANDOM STATES')
    s0 = ProductState(n_particles, isospin=isospin).randomize(100)
    s1 = ProductState(n_particles, isospin=isospin).randomize(101)
    print("|0> = \n", s0)
    print("|1> = \n", s1)
    print('PRODUCT INNER PRODUCTS')
    print("<0|0> = \n", s0.dagger().inner(s0)  )
    print("<1|1> = \n", s1.dagger().inner(s1)  )
    print("<0|1> = \n", s0.dagger().inner(s1) )
    print('PRODUCT OUTER PRODUCTS')
    print("|0><0| = \n", s0.outer(s0.dagger()) )
    print("|1><1| = \n", s1.outer(s1.dagger()) )
    print("|0><1| = \n", s0.outer(s1.dagger()) )
    print('PRODUCT TO MBB')
    s0 = s0.to_manybody_basis()
    s1 = s1.to_manybody_basis()
    print("<MBB|0> = \n", s0)
    print("<MBB|1> = \n", s1)
    print('INNER PRODUCTS')
    print("<0|0> = \n", s0.dagger().inner(s0)  )
    print("<1|1> = \n", s1.dagger().inner(s1)  )
    print("<0|1> = \n", s0.dagger().inner(s1) )
    print('OUTER PRODUCTS')
    print("|0><0| = \n", s0.outer(s0.dagger()) )
    print("|1><1| = \n", s1.outer(s1.dagger()) )
    print("|0><1| = \n", s0.outer(s1.dagger()) )
    print('DONE TESTING PRODUCT STATES')


def test_product_operators(n_particles, isospin):
    s0 = ProductState(n_particles, isospin=isospin).randomize(100)
    s1 = ProductState(n_particles, isospin=isospin).randomize(101)
    sig = [[ProductOperator(n_particles, isospin=isospin).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    for i in range(n_particles):
        for a in range(3):
            print('sigma', i, a)
            print(sig[i][a])
    
    print("<0|sig(0,0)|0> = \n", s0.dagger().inner(sig[0][0].multiply_state(s0)))
    print("<1|sig(0,0)|1> = \n", s1.dagger().inner(sig[0][0].multiply_state(s1)))
    print("<0|sig(0,0)|1> = \n", s0.dagger().inner(sig[0][0].multiply_state(s1)))
    
    print("<0|sig(0,0)sig(1,0)|0> = \n", s0.dagger().inner(sig[0][0].multiply_operator(sig[1][0]).multiply_state(s0)))
    
    
    
    print('DONE TESTING PRODUCT OPERATORS')


def test_hilbert_prop(n_particles, dt, isospin):
    seeder = itertools.count(0, 1)
    
    ket = ProductState(n_particles, isospin=isospin).randomize(seed=next(seeder)).to_manybody_basis()
    bra = ket.copy().dagger()
    
    pot = NuclearPotential(n_particles)
    pot.sigma.generate(1.0, seed=next(seeder))
    pot.sigmatau.generate(1.0, seed=next(seeder))
    pot.tau.generate(1.0, seed=next(seeder))
    pot.coulomb.generate(0.1, seed=next(seeder))
    pot.spinorbit.generate(dt, seed=next(seeder))
    
    # prop = HilbertPropagatorHS(n_particles, dt, isospin=isospin)
    prop = HilbertPropagatorRBM(n_particles, dt, isospin=isospin)
    aux = np.ones(prop.n_aux_sigma).flatten()
    factors = prop.factors_sigma(pot, aux)
    ket_prop = ket.copy()
    for f in factors:
        ket_prop = f.multiply_state(ket_prop)
        print(ket_prop)
    out = bra.inner(ket_prop)
    return out
    
    
def test_product_prop(n_particles, dt, isospin):
    seeder = itertools.count(0, 1)
    
    ket = ProductState(n_particles, isospin=isospin).randomize(seed=next(seeder))
    bra = ket.copy().dagger()
    
    pot = NuclearPotential(n_particles)
    pot.sigma.generate(1.0, seed=next(seeder))
    pot.sigmatau.generate(1.0, seed=next(seeder))
    pot.tau.generate(1.0, seed=next(seeder))
    pot.coulomb.generate(0.1, seed=next(seeder))
    pot.spinorbit.generate(dt, seed=next(seeder))
    
    # prop = ProductPropagatorHS(n_particles, dt, isospin=isospin)
    prop = ProductPropagatorRBM(n_particles, dt, isospin=isospin)
    aux = np.ones(prop.n_aux_sigma).flatten()
    factors = prop.factors_sigma(pot, aux)
    ket_prop = ket.copy()
    for f in factors:
        ket_prop = f.multiply_state(ket_prop)
    out = bra.inner(ket_prop)
    return out

    
    
    
if __name__=="__main__":
    n_particles = 2
    dt = 0.01
    isospin = True

    # test_hilbert_basics(n_particles, isospin)
    # test_product_basics(n_particles, isospin)
    
    # test_hilbert_operators(n_particles, isospin)
    # test_product_operators(n_particles, isospin)

    # b_hilbert = test_hilbert_prop(n_particles, dt, isospin)
    # b_product = test_product_prop(n_particles, dt, isospin)
    # print(b_hilbert, b_product)
    # print(b_hilbert - b_product)
    
    # result = test_overlaps(isospin=True)
    result = test_overlaps_unsafe(isospin=isospin)
    print(result)
        
    