from spinbox.core_backup import *
from spinbox.core import ArgonnePotential

n_particles = 2
dt = 0.001

seed = itertools.count(0,1)
pot = ArgonnePotential(n_particles)
pot.sigma.generate(1.0, seed=next(seed))
pot.sigmatau.generate(1.0, seed=next(seed))
pot.tau.generate(1.0, seed=next(seed))
pot.coulomb.generate(0.1, seed=next(seed))
pot.spinorbit.generate(dt, seed=next(seed))


def hilbert_bracket(method, controls):
    seed = itertools.count(0,1)
    ket = AFDMCSpinIsospinState(n_particles, np.zeros(shape=(n_particles, 4, 1))).randomize(seed=next(seed)).to_manybody_basis()
    bra = ket.copy().dagger()
    
    if method=='hs':
        prop = GFMCPropagatorHS(n_particles, dt)
    elif method=='rbm':
        prop = GFMCPropagatorRBM(n_particles, dt)
    else:
        raise ValueError

    prop_list = []
    if controls["sigma"]:
        aux = np.ones(prop.n_aux_sigma).flatten()
        prop_list.extend( prop.factors_sigma(pot, aux) )
    if controls["sigmatau"]:
        aux = np.ones(prop.n_aux_sigmatau).flatten()
        prop_list.extend( prop.factors_sigmatau(pot, aux) )
    if controls["tau"]:
        aux = np.ones(prop.n_aux_tau).flatten()
        prop_list.extend( prop.factors_tau(pot, aux) )
    if controls["coulomb"]:
        aux = np.ones(prop.n_aux_coulomb).flatten()
        prop_list.extend( prop.factors_coulomb(pot, aux) )
    if controls["spinorbit"]: 
        aux = np.ones(prop.n_aux_spinorbit).flatten()  # use sigma shape for spinorbit
        prop_list.extend( prop.factors_spinorbit(pot, aux) )
    
    rng = np.random.default_rng(seed=next(seed))
    rng.shuffle(prop_list)
    ket_prop = ket.copy()
    for p in prop_list:
        ket_prop = p * ket_prop
    return bra * ket_prop


def product_bracket(method, controls):
    seed = itertools.count(0,1)
    ket = AFDMCSpinIsospinState(n_particles, np.zeros(shape=(n_particles, 4, 1))).randomize(seed=next(seed))
    bra = ket.copy().dagger()
    
    if method=='hs':
        prop = AFDMCPropagatorHS(n_particles, dt)
    elif method=='rbm':
        prop = AFDMCPropagatorRBM(n_particles, dt)
    else:
        raise ValueError

    prop_list = []
    if controls["sigma"]:
        aux = np.ones(prop.n_aux_sigma).flatten()
        prop_list.extend( prop.factors_sigma(pot, aux) )
    if controls["sigmatau"]:
        aux = np.ones(prop.n_aux_sigmatau).flatten()
        prop_list.extend( prop.factors_sigmatau(pot, aux) )
    if controls["tau"]:
        aux = np.ones(prop.n_aux_tau).flatten()
        prop_list.extend( prop.factors_tau(pot, aux) )
    if controls["coulomb"]:
        aux = np.ones(prop.n_aux_coulomb).flatten()
        prop_list.extend( prop.factors_coulomb(pot, aux) )
    if controls["spinorbit"]: 
        aux = np.ones(prop.n_aux_spinorbit).flatten()  # use sigma shape for spinorbit
        prop_list.extend( prop.factors_spinorbit(pot, aux) )
    
    rng = np.random.default_rng(seed=next(seed))
    rng.shuffle(prop_list)
    ket_prop = ket.copy()
    for p in prop_list:
        ket_prop = p * ket_prop
    return bra * ket_prop


if __name__ == "__main__":
    method = 'hs'
    
    pot.sigma = 0.*pot.sigma
    pot.sigma[0,0,0,1] = 1.
    
    controls = {"sigma": True,
                "sigmatau": False,
                "tau": False,
                "coulomb": False,
                "spinorbit": False,
                }
    b_hilbert = hilbert_bracket(method, controls)
    print('hilbert = ', b_hilbert)
    b_product = product_bracket(method, controls)
    print('product = ', b_product)
    print('ratio =', np.abs(b_hilbert/b_product) )
    print('abs error =', np.abs(1-np.abs(b_hilbert/b_product)) )
    