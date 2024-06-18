from spinbox import *

n_particles = 2
dt = 0.001

def gfmc_bracket(method, controls):
    ket = AFDMCSpinIsospinState(n_particles, read_from_file("./data/h2/fort.770",complex=True, shape=(2,4,1)), ketwise=True).to_manybody_basis()
    bra = ket.copy().dagger()

    pot = ArgonnePotential(n_particles)
    pot.read_sigma("./data/h2/fort.7701")
    pot.read_sigmatau("./data/h2/fort.7702")
    pot.read_tau("./data/h2/fort.7703")
    pot.read_coulomb("./data/h2/fort.7704")
    pot.read_spinorbit("./data/h2/fort.7705")
    
    if method=='hs':
        hsprop = GFMCPropagatorHS(n_particles, dt, mix=False)
    elif method=='rbm':
        hsprop = GFMCPropagatorRBM(n_particles, dt, mix=False)
    else:
        raise ValueError

    ket_prop = ket.copy()
    if controls["sigma"]:
        ket_prop = hsprop.apply_sigma(ket_prop, pot, np.ones(shape=(9)))
    if controls["sigmatau"]:
        ket_prop = hsprop.apply_sigmatau(ket_prop, pot, np.ones(shape=(27)))
    if controls["tau"]:
        ket_prop = hsprop.apply_tau(ket_prop, pot, np.ones(shape=(3)))
    if controls["coulomb"]:
        ket_prop = hsprop.apply_coulomb(ket_prop, pot, np.ones(shape=(1)))
    if controls["spinorbit"]:
        ket_prop = hsprop.apply_spinorbit(ket_prop, pot, np.ones(shape=(9)))
    
    return bra * ket_prop


def afdmc_bracket(method, controls):
    ket = AFDMCSpinIsospinState(n_particles, read_from_file("./data/h2/fort.770",complex=True, shape=(2,4,1)), ketwise=True)
    bra = ket.copy().dagger()

    pot = ArgonnePotential(n_particles)
    pot.read_sigma("./data/h2/fort.7701")
    pot.read_sigmatau("./data/h2/fort.7702")
    pot.read_tau("./data/h2/fort.7703")
    pot.read_coulomb("./data/h2/fort.7704")
    pot.read_spinorbit("./data/h2/fort.7705")
    
    if method=='hs':
        hsprop = AFDMCPropagatorHS(n_particles, dt, mix=False)
    elif method=='rbm':
        hsprop = AFDMCPropagatorRBM(n_particles, dt, mix=False)
    else:
        raise ValueError

    ket_prop = ket.copy()
    if controls["sigma"]:
        ket_prop = hsprop.apply_sigma(ket_prop, pot, np.ones(shape=(9)))
    if controls["sigmatau"]:
        ket_prop = hsprop.apply_sigmatau(ket_prop, pot, np.ones(shape=(27)))
    if controls["tau"]:
        ket_prop = hsprop.apply_tau(ket_prop, pot, np.ones(shape=(3)))
    if controls["coulomb"]:
        ket_prop = hsprop.apply_coulomb(ket_prop, pot, np.ones(shape=(1)))
    if controls["spinorbit"]:
        ket_prop = hsprop.apply_spinorbit(ket_prop, pot, np.ones(shape=(9)))
    
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
    