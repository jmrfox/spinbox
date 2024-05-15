from quap import *


def test_overlaps_spin():

    def exam(s0, s1):
        out = []
        out.append(s0.coefficients)
        out.append(s1.coefficients)
        out.append( s0.dagger().inner(s0)  )
        out.append( s1.dagger().inner(s1)  )
        out.append( s0.dagger().inner(s1)  )
        out.append( s0.outer(s0.dagger()) )
        out.append( s1.outer(s1.dagger()) )
        out.append( s0.outer(s1.dagger()) )
        return out


    s0 = ProductState(n_particles, isospin=False).randomize(100).to_manybody_basis()
    s1 = ProductState(n_particles, isospin=False).randomize(101).to_manybody_basis()
    hilbert_list = exam(s0, s1)
    s0 = ProductState(n_particles, isospin=False).randomize(100)
    s1 = ProductState(n_particles, isospin=False).randomize(101)
    
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



def test_hilbert_spin_basics(n_particles):
    print('HILBERT RANDOM STATES')
    s0 = ProductState(n_particles, isospin=False).randomize(100).to_manybody_basis()
    s1 = ProductState(n_particles, isospin=False).randomize(101).to_manybody_basis()
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

def test_hilbert_spin_operators(n_particles):
    s0 = ProductState(n_particles, isospin=False).randomize(100).to_manybody_basis()
    s1 = ProductState(n_particles, isospin=False).randomize(101).to_manybody_basis()
    sig = [[HilbertOperator(n_particles, isospin=False).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    for i in range(n_particles):
        for a in range(3):
            print('sigma', i, a)
            print(sig[i][a])
            
    print("<0|sig(0,0)|0> = \n", s0.dagger().inner(sig[0][0]*s0))
    print("<1|sig(0,0)|1> = \n", s1.dagger().inner(sig[0][0]*s1))
    print("<0|sig(0,0)|1> = \n", s0.dagger().inner(sig[0][0]*s1))
    
    print('DONE TESTING HILBERT OPERATORS')


def test_product_spin_basics(n_particles):
    print('PRODUCT RANDOM STATES')
    s0 = ProductState(n_particles, isospin=False).randomize(100)
    s1 = ProductState(n_particles, isospin=False).randomize(101)
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


def test_product_spin_operators(n_particles):
    s0 = ProductState(n_particles, isospin=False).randomize(100)
    s1 = ProductState(n_particles, isospin=False).randomize(101)
    sig = [[ProductOperator(n_particles, isospin=False).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    for i in range(n_particles):
        for a in range(3):
            print('sigma', i, a)
            print(sig[i][a])
    
    print("<0|sig(0,0)|0> = \n", s0.dagger().inner(sig[0][0]*s0))
    print("<1|sig(0,0)|1> = \n", s1.dagger().inner(sig[0][0]*s1))
    print("<0|sig(0,0)|1> = \n", s0.dagger().inner(sig[0][0]*s1))

    print('DONE TESTING PRODUCT OPERATORS')



def test_hilbert_isospin(n_particles):
    print('HILBERT RANDOM STATES')
    s0 = HilbertState(n_particles).randomize(100)
    s1 = HilbertState(n_particles).randomize(101)
    print("|0> = \n", s0)
    print("|1> = \n", s1)
    print('INNER PRODUCTS')
    print("<0|0> = \n", s0.dagger().inner(s0)  )
    print("<1|1> = \n", s1.dagger().inner(s1)  )
    print("<0|1> = \n", s0.dagger().inner(s1) )
    print('OUTER PRODUCTS')
    print("|0><0| = \n", s0.outer(s0.dagger()) )
    print("|1><1| = \n", s1.outer(s1.dagger()) )
    print("|0><1| = \n", s0.outer(s1.dagger()) )
    print('DONE TESTING HILBERT STATES')

    print('TESTING HILBERT OPERATORS')
    sig = [[HilbertOperator(n_particles).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    for i in range(n_particles):
        for a in range(3):
            print('sigma', i, a)
            print(sig[i][a])
    
    print('HILBERT TEST COMPLETE')


def test_product_isospin(n_particles):
    print('PRODUCT RANDOM STATES')
    s0 = ProductState(n_particles).randomize(100)
    s1 = ProductState(n_particles).randomize(101)
    print("|0> = \n", s0)
    print("|1> = \n", s1)
    print('INNER PRODUCTS')
    print("<0|0> = \n", s0.dagger().inner(s0)  )
    print("<1|1> = \n", s1.dagger().inner(s1)  )
    print("<0|1> = \n", s0.dagger().inner(s1) )
    print('OUTER PRODUCTS')
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

    print('TESTING PRODUCT OPERATORS')
    sig = [[ProductOperator(n_particles).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    for i in range(n_particles):
        for a in range(3):
            print('sigma', i, a)
            print(sig[i][a])
    
    print('PRODUCT TEST COMPLETE')


def test_hilbert_prop(n_particles, dt):
    seeder = itertools.count(0, 1)
    
    ket = ProductState(n_particles).randomize(seed=next(seeder)).to_manybody_basis()
    bra = ket.copy().dagger()
    
    pot = ArgonnePotential(n_particles)
    pot.sigma.generate(1.0, seed=next(seeder))
    pot.sigmatau.generate(1.0, seed=next(seeder))
    pot.tau.generate(1.0, seed=next(seeder))
    pot.coulomb.generate(0.1, seed=next(seeder))
    pot.spinorbit.generate(dt, seed=next(seeder))
    
    prop = HilbertPropagatorRBM(n_particles, dt)
    aux = np.ones(prop.n_aux_sigma).flatten()
    factors = prop.factors_sigma(pot, aux)
    ket_prop = ket.copy()
    for f in factors:
        ket_prop = f * ket_prop
    out = bra * ket_prop
    return out
    
    
def test_product_prop(n_particles, dt):
    seeder = itertools.count(0, 1)
    
    ket = ProductState(n_particles).randomize(seed=next(seeder))
    bra = ket.copy().dagger()
    
    pot = ArgonnePotential(n_particles)
    pot.sigma.generate(1.0, seed=next(seeder))
    pot.sigmatau.generate(1.0, seed=next(seeder))
    pot.tau.generate(1.0, seed=next(seeder))
    pot.coulomb.generate(0.1, seed=next(seeder))
    pot.spinorbit.generate(dt, seed=next(seeder))
    
    prop = ProductPropagatorRBM(n_particles, dt)
    aux = np.ones(prop.n_aux_sigma).flatten()
    factors = prop.factors_sigma(pot, aux)
    ket_prop = ket.copy()
    for f in factors:
        ket_prop = f * ket_prop
    out = bra * ket_prop
    return out

    
    

if __name__=="__main__":
    n_particles = 2
    dt = 0.01
    
    # test_hilbert_spin_basics(n_particles)
    # test_product_spin_basics(n_particles)
    
    test_hilbert_spin_operators(n_particles)
    test_product_spin_operators(n_particles)

    # test_hilbert_isospin(n_particles)
    # test_product_isospin(n_particles)

    # test_hilbert_prop(n_particles, dt)
    # test_product_prop(n_particles, dt)
