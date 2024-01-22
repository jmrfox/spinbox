from quap import *


def test_obb(n_particles):
    print('OBB RANDOM STATES')
    s0 = OneBodyBasisSpinState(n_particles, 'ket', np.zeros(shape=(n_particles, 2, 1))).randomize(100)
    s1 = OneBodyBasisSpinState(n_particles, 'ket', np.zeros(shape=(n_particles, 2, 1))).randomize(101)
    print("|0> = \n", s0)
    print("|1> = \n", s1)
    print('OBB INNER PRODUCTS')
    print("<0|0> = \n", s0.dagger() * s0)
    print("<1|1> = \n", s1.dagger() * s1)
    print("<0|1> = \n", s1.dagger() * s0)
    print('OBB OUTER PRODUCTS')
    print("|0><0| = \n", s0 * s0.dagger())
    print("|1><1| = \n", s1 * s1.dagger())
    print("|0><1| = \n", s0 * s1.dagger())
    print('OBB TO MBB')
    s0 = s0.to_many_body_state()
    s1 = s1.to_many_body_state()
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
    print('DONE TESTING OBB STATES')

    print('TESTING OPERATORS')
    print('OBB OPERATORS')
    sig = [[OneBodyBasisSpinOperator(n_particles).sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    
    for i in range(n_particles):
        for a in range(3):
            print('sigma', i, a)
            print(sig[i][a])
    
    print('OBB TEST COMPLETE')

def test_mbb(n_particles):
    print('MBB RANDOM STATES')
    s0 = ManyBodyBasisSpinState(n_particles, 'ket', np.zeros(shape=(2**n_particles, 1))).randomize(100)
    s1 = ManyBodyBasisSpinState(n_particles, 'ket', np.zeros(shape=(2**n_particles, 1))).randomize(101)
    print("|0> = \n", s0)
    print("|1> = \n", s1)
    print('MBB INNER PRODUCTS')
    print("<0|0> = \n", s0.dagger() * s0)
    print("<1|1> = \n", s1.dagger() * s1)
    print("<0|1> = \n", s1.dagger() * s0)
    print('MBB OUTER PRODUCTS')
    print("|0><0| = \n", s0 * s0.dagger())
    print("|1><1| = \n", s1 * s1.dagger())
    print("|0><1| = \n", s0 * s1.dagger())
    print('DONE TESTING MBB STATES')

    print('TESTING OPERATORS')
    print('MBB OPERATORS')
    sig = [[ManyBodyBasisSpinOperator(n_particles).sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    
    for i in range(n_particles):
        for a in range(3):
            print('sigma', i, a)
            print(sig[i][a])
    
    print('MBB TEST COMPLETE')


if __name__=="__main__":
    z=2
    test_obb(z)
    test_mbb(z)