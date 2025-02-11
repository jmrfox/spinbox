from spinbox import *

n_particles = 3
dt = 0.01

global_seed = 5
pot = NuclearPotential(n_particles)
pot.random(seed=1729)

def hilbert_bracket(controls):
    seeder = itertools.count(global_seed,1)
    ket = ProductState(n_particles).random(seed=next(seeder)).to_full_basis()
    bra = ket.copy().dagger()
    
    prop = HilbertPropagatorRBM(n_particles, dt)

    integ = Integrator(pot, prop)
    integ.setup(n_samples=1, **controls, parallel=False, seed=next(seeder)) # use the integrator class to produce one sample of the aux field

    b = integ.bracket(bra, ket, integ.aux_fields_samples[0])
    return b


def product_bracket(controls):
    seeder = itertools.count(global_seed,1)
    ket = ProductState(n_particles).random(seed=next(seeder))
    bra = ket.copy().dagger()

    prop = ProductPropagatorRBM(n_particles, dt)

    integ = Integrator(pot, prop)
    integ.setup(n_samples=1, **controls, parallel=False, seed=next(seeder))
    
    b = integ.bracket(bra, ket, integ.aux_fields_samples[0])
    return b


if __name__ == "__main__":
    controls = {"sigma": True,
                "sigmatau": False,
                "tau": False,
                "coulomb": False,
                "spinorbit": False,
                "sigma_3b": False,
                }

    b_hilbert = hilbert_bracket(controls)
    print('hilbert = ', b_hilbert)
    b_product = product_bracket(controls)
    print('product = ', b_product)
    print('ratio =', np.abs(b_hilbert/b_product) )
    print('abs error =', np.abs(1-np.abs(b_hilbert/b_product)) )
    