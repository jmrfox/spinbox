from spinbox import *

n_particles = 3
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
    ket = ProductState(n_particles).randomize(seed=next(seed)).to_full_basis()
    bra = ket.copy().dagger()
    
    if method=='hs':
        prop = HilbertPropagatorHS(n_particles, dt)
    elif method=='rbm':
        prop = HilbertPropagatorRBM(n_particles, dt)
    else:
        raise ValueError

    prop_list = []
    if controls["sigma"]:
        aux = np.ones(prop.n_aux_sigma).flatten()
        prop_list.extend( prop.factors_sigma(pot.sigma, aux) )
    if controls["sigmatau"]:
        aux = np.ones(prop.n_aux_sigmatau).flatten()
        prop_list.extend( prop.factors_sigmatau(pot.sigmatau, aux) )
    if controls["tau"]:
        aux = np.ones(prop.n_aux_tau).flatten()
        prop_list.extend( prop.factors_tau(pot.tau, aux) )
    if controls["coulomb"]:
        aux = np.ones(prop.n_aux_coulomb).flatten()
        prop_list.extend( prop.factors_coulomb(pot.coulomb, aux) )
    if controls["spinorbit"]: 
        aux = np.ones(prop.n_aux_spinorbit).flatten()  # use sigma shape for spinorbit
        prop_list.extend( prop.factors_spinorbit(pot.spinorbit, aux) )
    
    rng = np.random.default_rng(seed=next(seed))
    rng.shuffle(prop_list)
    ket_prop = ket.copy()
    for p in prop_list:
        ket_prop = p.multiply_state(ket_prop)
    return bra.inner(ket_prop)


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


    prop_list = []   
    if controls["sigma"]:
        aux = np.ones(prop.n_aux_sigma).flatten() # evaluate aux field at 1 for all variables
        prop_list.extend( prop.factors_sigma(pot.sigma, aux) )
    if controls["sigmatau"]:
        aux = np.ones(prop.n_aux_sigmatau).flatten()
        prop_list.extend( prop.factors_sigmatau(pot.sigmatau, aux) )
    if controls["tau"]:
        aux = np.ones(prop.n_aux_tau).flatten()
        prop_list.extend( prop.factors_tau(pot.tau, aux) )
    if controls["coulomb"]:
        aux = np.ones(prop.n_aux_coulomb).flatten()
        prop_list.extend( prop.factors_coulomb(pot.coulomb, aux) )
    if controls["spinorbit"]: 
        aux = np.ones(prop.n_aux_spinorbit).flatten()  # use sigma shape for spinorbit
        prop_list.extend( prop.factors_spinorbit(pot.spinorbit, aux) )
        
            
    rng = np.random.default_rng(seed=next(seed))
    rng.shuffle(prop_list)
    ket_prop = ket.copy()
    for p in prop_list:
        ket_prop = p.multiply_state(ket_prop)
    return bra.inner(ket_prop)


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
    