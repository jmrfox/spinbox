from spinbox import *

n_particles = 2
dt = 0.001

seed = itertools.count(0,1)
pot = NuclearPotential(n_particles)
pot.sigma.random(1.0, seed=next(seed))
pot.sigmatau.random(1.0, seed=next(seed))
pot.tau.random(1.0, seed=next(seed))
pot.coulomb.random(0.1, seed=next(seed))
pot.spinorbit.random(dt, seed=next(seed))


def hilbert_bracket(method, controls):
    seed = itertools.count(0,1)
    ket = ProductState(n_particles).randomize(seed=next(seed)).to_manybody_basis()
    bra = ket.copy().dagger()
    
    if method=='hs':
        prop = HilbertPropagatorHS(n_particles, dt)
    elif method=='rbm':
        prop = HilbertPropagatorRBM(n_particles, dt)
    else:
        raise ValueError

    integ = Integrator(pot, prop) 
    integ.setup(n_samples=1, **controls, parallel=False, seed=next(seed)) # use the integrator class to produce one sample of the aux field
    print(integ.aux_fields_samples.shape)
    b = integ.bracket(bra, ket, integ.aux_fields_samples[0])
    return b


def product_bracket(method, controls):
    seed = itertools.count(0,1)
    ket = ProductState(n_particles).randomize(seed=next(seed))
    bra = ket.copy().dagger()
    
    if method=='hs':
        prop = ProductPropagatorHS(n_particles, dt)
    elif method=='rbm':
        prop = ProductPropagatorRBM(n_particles, dt)
    else:
        raise ValueError

    integ = Integrator(pot, prop)
    integ.setup(n_samples=1, **controls, parallel=False, seed=next(seed))
    print(integ.aux_fields_samples.shape)
    b = integ.bracket(bra, ket, integ.aux_fields_samples[0])
    return b


if __name__ == "__main__":
    method = 'rbm'
    
    controls = {"sigma": True,
                "sigmatau": True,
                "tau": True,
                "coulomb": True,
                "spinorbit": True,
                }
    b_hilbert = hilbert_bracket(method, controls)
    print('hilbert = ', b_hilbert)
    b_product = product_bracket(method, controls)
    print('product = ', b_product)
    print('ratio =', np.abs(b_hilbert/b_product) )
    print('abs error =', np.abs(1-np.abs(b_hilbert/b_product)) )
    