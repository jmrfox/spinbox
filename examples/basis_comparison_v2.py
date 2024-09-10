from spinbox import *

n_particles = 2
dt = 0.001

global_seed = 1
pot = NuclearPotential(n_particles)
pot.random(seed=1729)

def hilbert_bracket(method, controls):
    seeder = itertools.count(global_seed,1)
    ket = ProductState(n_particles).random(seed=next(seeder)).to_full_basis()
    bra = ket.copy().dagger()
    
    if method=='hs':
        prop = HilbertPropagatorHS(n_particles, dt)
    elif method=='rbm':
        prop = HilbertPropagatorRBM(n_particles, dt)
    else:
        raise ValueError

    integ = Integrator(pot, prop) 
    integ.setup(n_samples=1, **controls, parallel=False, seed=next(seeder)) # use the integrator class to produce one sample of the aux field
    print(integ.aux_fields_samples.shape)
    b = integ.bracket(bra, ket, integ.aux_fields_samples[0])
    return b


def product_bracket(method, controls):
    seeder = itertools.count(global_seed,1)
    ket = ProductState(n_particles).random(seed=next(seeder))
    bra = ket.copy().dagger()
    
    if method=='hs':
        prop = ProductPropagatorHS(n_particles, dt)
    elif method=='rbm':
        prop = ProductPropagatorRBM(n_particles, dt)
    else:
        raise ValueError

    integ = Integrator(pot, prop)
    integ.setup(n_samples=1, **controls, parallel=False, seed=next(seeder))
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
                "sigma_3b": True,
                }
    
    if method=="hs" and controls["sigma_3b"]:
        raise ValueError("There is no HS transform for the 3-body force")

    b_hilbert = hilbert_bracket(method, controls)
    print('hilbert = ', b_hilbert)
    b_product = product_bracket(method, controls)
    print('product = ', b_product)
    print('ratio =', np.abs(b_hilbert/b_product) )
    print('abs error =', np.abs(1-np.abs(b_hilbert/b_product)) )
    