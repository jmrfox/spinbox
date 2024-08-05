from spinbox import *

n_particles = 3
dt = 0.001

seed = itertools.count(0,1)
pot = NuclearPotential(n_particles)
# pot.sigma.random(1.0, seed=next(seed))
# pot.sigmatau.random(1.0, seed=next(seed))
# pot.tau.random(1.0, seed=next(seed))
# pot.coulomb.random(0.1, seed=next(seed))
# pot.spinorbit.random(dt, seed=next(seed))
# pot.sigma_3b.random(1.0, seed=next(seed))
pot.sigma_3b[0,0,0,1,0,2] = 10.0

def hilbert_bracket(controls):
    seed = itertools.count(0,1)
    ket = ProductState(n_particles).randomize(seed=next(seed)).to_manybody_basis()
    bra = ket.copy().dagger()
    
    prop = HilbertPropagatorRBM(n_particles, dt)

    integ = Integrator(pot, prop)
    integ.setup(n_samples=1, **controls, parallel=False, seed=next(seed)) # use the integrator class to produce one sample of the aux field
    print(integ.aux_fields_samples)
    b = integ.bracket(bra, ket, integ.aux_fields_samples[0])
    return b


def product_bracket(controls):
    seed = itertools.count(0,1)
    ket = ProductState(n_particles).randomize(seed=next(seed))
    bra = ket.copy().dagger()

    prop = ProductPropagatorRBM(n_particles, dt)

    integ = Integrator(pot, prop)
    integ.setup(n_samples=1, **controls, parallel=False, seed=next(seed))
    print(integ.aux_fields_samples)
    b = integ.bracket(bra, ket, integ.aux_fields_samples[0])
    return b


if __name__ == "__main__":
    controls = {"sigma": False,
                "sigmatau": False,
                "tau": False,
                "coulomb": False,
                "spinorbit": False,
                "sigma_3b": True,
                }

    b_hilbert = hilbert_bracket(controls)
    print('hilbert = ', b_hilbert)
    b_product = product_bracket(controls)
    print('product = ', b_product)
    print('ratio =', np.abs(b_hilbert/b_product) )
    print('abs error =', np.abs(1-np.abs(b_hilbert/b_product)) )
    