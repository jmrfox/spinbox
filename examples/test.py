from quap import *


def test_basic_afdmc(n_particles):
    print('AFDMC RANDOM STATES')
    s0 = AFDMCSpinState(n_particles, np.zeros(shape=(n_particles, 2, 1))).randomize(100)
    s1 = AFDMCSpinState(n_particles, np.zeros(shape=(n_particles, 2, 1))).randomize(101)
    print("|0> = \n", s0)
    print("|1> = \n", s1)
    print('AFDMC INNER PRODUCTS')
    print("<0|0> = \n", s0.dagger() * s0)
    print("<1|1> = \n", s1.dagger() * s1)
    print("<0|1> = \n", s1.dagger() * s0)
    print('AFDMC OUTER PRODUCTS')
    print("|0><0| = \n", s0 * s0.dagger())
    print("|1><1| = \n", s1 * s1.dagger())
    print("|0><1| = \n", s0 * s1.dagger())
    print('AFDMC TO MBB')
    s0 = s0.to_manybody_basis()
    s1 = s1.to_manybody_basis()
    print("<MBB|0> = \n", s0)
    print("<MBB|1> = \n", s1)
    print('INNER PRODUCTS')
    print("<0|0> = \n", s0.dagger() * s0)
    print("<1|1> = \n", s1.dagger() * s1)
    print("<0|1> = \n", s0.dagger() * s1)
    print('OUTER PRODUCTS')
    print("|0><0| = \n", s0 * s0.dagger())
    print("|1><1| = \n", s1 * s1.dagger())
    print("|0><1| = \n", s0 * s1.dagger())
    print('DONE TESTING AFDMC STATES')

    print('TESTING AFDMC OPERATORS')
    sig = [[AFDMCSpinOperator(n_particles).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    
    for i in range(n_particles):
        for a in range(3):
            print('sigma', i, a)
            print(sig[i][a])
    
    print('AFDMC TEST COMPLETE')

def test_basic_gfmc(n_particles):
    print('GFMC RANDOM STATES')
    s0 = GFMCSpinState(n_particles, np.zeros(shape=(2**n_particles, 1))).randomize(100)
    s1 = GFMCSpinState(n_particles, np.zeros(shape=(2**n_particles, 1))).randomize(101)
    print("|0> = \n", s0)
    print("|1> = \n", s1)
    print('GFMC INNER PRODUCTS')
    print("<0|0> = \n", s0.dagger() * s0)
    print("<1|1> = \n", s1.dagger() * s1)
    print("<0|1> = \n", s1.dagger() * s0)
    print('GFMC OUTER PRODUCTS')
    print("|0><0| = \n", s0 * s0.dagger())
    print("|1><1| = \n", s1 * s1.dagger())
    print("|0><1| = \n", s0 * s1.dagger())
    print('DONE TESTING GFMC STATES')

    print('TESTING GFMC OPERATORS')
    sig = [[GFMCSpinOperator(n_particles).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    
    for i in range(n_particles):
        for a in range(3):
            print('sigma', i, a)
            print(sig[i][a])
    
    print('GFMC TEST COMPLETE')


def test_afdmc(n_particles):
    print('AFDMC RANDOM STATES')
    s0 = AFDMCSpinIsospinState(n_particles, np.zeros(shape=(n_particles, 4, 1))).randomize(100)
    s1 = AFDMCSpinIsospinState(n_particles, np.zeros(shape=(n_particles, 4, 1))).randomize(101)
    print("|0> = \n", s0)
    print("|1> = \n", s1)
    print('AFDMC INNER PRODUCTS')
    print("<0|0> = \n", s0.dagger() * s0)
    print("<1|1> = \n", s1.dagger() * s1)
    print("<0|1> = \n", s1.dagger() * s0)
    print('AFDMC OUTER PRODUCTS')
    print("|0><0| = \n", s0 * s0.dagger())
    print("|1><1| = \n", s1 * s1.dagger())
    print("|0><1| = \n", s0 * s1.dagger())
    print('AFDMC TO MBB')
    s0 = s0.to_manybody_basis()
    s1 = s1.to_manybody_basis()
    print("<MBB|0> = \n", s0)
    print("<MBB|1> = \n", s1)
    print('INNER PRODUCTS')
    print("<0|0> = \n", s0.dagger() * s0)
    print("<1|1> = \n", s1.dagger() * s1)
    print("<0|1> = \n", s0.dagger() * s1)
    print('OUTER PRODUCTS')
    print("|0><0| = \n", s0 * s0.dagger())
    print("|1><1| = \n", s1 * s1.dagger())
    print("|0><1| = \n", s0 * s1.dagger())
    print('DONE TESTING AFDMC STATES')

    print('TESTING AFDMC OPERATORS')
    sig = [[AFDMCSpinIsospinOperator(n_particles).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    
    for i in range(n_particles):
        for a in range(3):
            print('sigma', i, a)
            print(sig[i][a])
    
    print('AFDMC TEST COMPLETE')

def test_gfmc(n_particles):
    print('GFMC RANDOM STATES')
    s0 = GFMCSpinIsospinState(n_particles, np.zeros(shape=(4**n_particles, 1))).randomize(100)
    s1 = GFMCSpinIsospinState(n_particles, np.zeros(shape=(4**n_particles, 1))).randomize(101)
    print("|0> = \n", s0)
    print("|1> = \n", s1)
    print('GFMC INNER PRODUCTS')
    print("<0|0> = \n", s0.dagger() * s0)
    print("<1|1> = \n", s1.dagger() * s1)
    print("<0|1> = \n", s1.dagger() * s0)
    print('GFMC OUTER PRODUCTS')
    print("|0><0| = \n", s0 * s0.dagger())
    print("|1><1| = \n", s1 * s1.dagger())
    print("|0><1| = \n", s0 * s1.dagger())
    print('DONE TESTING GFMC STATES')

    print('TESTING GFMC OPERATORS')
    sig = [[GFMCSpinIsospinOperator(n_particles).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    
    for i in range(n_particles):
        for a in range(3):
            print('sigma', i, a)
            print(sig[i][a])
    
    print('GFMC TEST COMPLETE')


def test_gfmc_prop(n_particles, dt):
    seeder = itertools.count(0, 1)
    
    ket = AFDMCSpinIsospinState(n_particles, np.zeros(shape=(n_particles, 4, 1)), ketwise=True).randomize(seed=next(seeder)).to_manybody_basis()
    bra = ket.copy().dagger()
    
    pot = ArgonnePotential(n_particles)
    pot.sigma.generate(1.0, seed=next(seeder))
    pot.sigmatau.generate(1.0, seed=next(seeder))
    pot.tau.generate(1.0, seed=next(seeder))
    pot.coulomb.generate(0.1, seed=next(seeder))
    pot.spinorbit.generate(dt, seed=next(seeder))
    
    prop = GFMCPropagatorRBM(n_particles, dt)
    aux = np.ones(prop.n_aux_sigma).flatten()
    factors = prop.factors_sigma(pot, aux)
    print( bra * np.prod(factors) * ket)
    
    
def test_afdmc_prop(n_particles, dt):
    seeder = itertools.count(0, 1)
    
    ket = AFDMCSpinIsospinState(n_particles, np.zeros(shape=(n_particles, 4, 1)), ketwise=True).randomize(seed=next(seeder))
    bra = ket.copy().dagger()
    
    pot = ArgonnePotential(n_particles)
    pot.sigma.generate(1.0, seed=next(seeder))
    pot.sigmatau.generate(1.0, seed=next(seeder))
    pot.tau.generate(1.0, seed=next(seeder))
    pot.coulomb.generate(0.1, seed=next(seeder))
    pot.spinorbit.generate(dt, seed=next(seeder))
    
    prop = AFDMCPropagatorRBM(n_particles, dt)
    aux = np.ones(prop.n_aux_sigma).flatten()
    factors = prop.factors_sigma(pot, aux)
    print( bra * np.prod(factors) * ket)
    
if __name__=="__main__":
    n_particles = 2
    dt = 0.01
    # test_gfmc(n_particles)
    # test_afdmc(n_particles)
    test_gfmc_prop(n_particles, dt)
    test_afdmc_prop(n_particles, dt)

    