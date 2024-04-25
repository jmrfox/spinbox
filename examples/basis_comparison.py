from quap import *

n_particles = 4
dt = 0.001

def gfmc_bracket(method, controls):
    seed = itertools.count(0,1)
    ket = AFDMCSpinIsospinState(n_particles, np.zeros(shape=(n_particles, 4, 1)), ketwise=True).randomize(seed=next(seed)).to_manybody_basis()
    bra = ket.copy().dagger()

    pot = ArgonnePotential(n_particles)
    pot.sigma.generate(1.0, seed=next(seed))
    pot.sigmatau.generate(1.0, seed=next(seed))
    pot.tau.generate(1.0, seed=next(seed))
    pot.coulomb.generate(0.1, seed=next(seed))
    pot.spinorbit.generate(dt, seed=next(seed))
    
    if method=='hs':
        prop = GFMCPropagatorHS(n_particles, dt, mix=False)
    elif method=='rbm':
        prop = GFMCPropagatorRBM(n_particles, dt, mix=False)
    else:
        raise ValueError

    ket_prop = ket.copy()
    if controls["sigma"]:
        aux = np.ones(prop.n_aux_sigma).flatten()
        ket_prop = prop.apply_sigma(ket_prop, pot, aux)
    if controls["sigmatau"]:
        aux = np.ones(prop.n_aux_sigmatau).flatten()
        ket_prop = prop.apply_sigmatau(ket_prop, pot, aux)
    if controls["tau"]:
        aux = np.ones(prop.n_aux_tau).flatten()
        ket_prop = prop.apply_tau(ket_prop, pot, aux)
    if controls["coulomb"]:
        aux = np.ones(prop.n_aux_coulomb).flatten()
        ket_prop = prop.apply_coulomb(ket_prop, pot, aux)
    if controls["spinorbit"]: 
        aux = np.ones(prop.n_aux_spinorbit).flatten()  # use sigma shape for spinorbit
        ket_prop = prop.apply_spinorbit(ket_prop, pot, aux)
    
    return bra * ket_prop


def afdmc_bracket(method, controls):
    seed = itertools.count(0,1)
    ket = AFDMCSpinIsospinState(n_particles, np.zeros(shape=(n_particles, 4, 1)), ketwise=True).randomize(seed=next(seed))
    bra = ket.copy().dagger()

    pot = ArgonnePotential(n_particles)
    pot.sigma.generate(1.0, seed=next(seed))
    pot.sigmatau.generate(1.0, seed=next(seed))
    pot.tau.generate(1.0, seed=next(seed))
    pot.coulomb.generate(0.1, seed=next(seed))
    pot.spinorbit.generate(dt, seed=next(seed))
    
    if method=='hs':
        prop = AFDMCPropagatorHS(n_particles, dt, mix=False)
    elif method=='rbm':
        prop = AFDMCPropagatorRBM(n_particles, dt, mix=False)
    else:
        raise ValueError

    ket_prop = ket.copy()
    if controls["sigma"]:
        aux = np.ones(prop.n_aux_sigma).flatten()
        ket_prop = prop.apply_sigma(ket_prop, pot, aux)
    if controls["sigmatau"]:
        aux = np.ones(prop.n_aux_sigmatau).flatten()
        ket_prop = prop.apply_sigmatau(ket_prop, pot, aux)
    if controls["tau"]:
        aux = np.ones(prop.n_aux_tau).flatten()
        ket_prop = prop.apply_tau(ket_prop, pot, aux)
    if controls["coulomb"]:
        aux = np.ones(prop.n_aux_coulomb).flatten()
        ket_prop = prop.apply_coulomb(ket_prop, pot, aux)
    if controls["spinorbit"]: 
        aux = np.ones(prop.n_aux_spinorbit).flatten()  # use sigma shape for spinorbit
        ket_prop = prop.apply_spinorbit(ket_prop, pot, aux)
    
    return bra * ket_prop


if __name__ == "__main__":
    method = 'hs'
    controls = {"sigma": True,
                "sigmatau": True,
                "tau": True,
                "coulomb": True,
                "spinorbit": True,
                }
    b_gfmc = gfmc_bracket(method, controls)
    b_afdmc = afdmc_bracket(method, controls)
    print('ratio =', np.abs(b_gfmc/b_afdmc) )
    print('abs error =', np.abs(1-np.abs(b_gfmc/b_afdmc)) )
    