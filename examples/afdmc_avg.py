from quap import *
from quap.exact import ExactGFMC

n_particles = 2
dt = 0.001
n_samples = 100000
seed = 2


def main():
    ket = AFDMCSpinIsospinState(n_particles,'ket', read_from_file("./data/h2/fort.770",complex=True, shape=(2,4,1)))
    bra = ket.copy().dagger()

    pot = ArgonnePotential(n_particles)
    pot.read_sigma("./data/h2/fort.7701")
    pot.read_sigmatau("./data/h2/fort.7702")
    pot.read_tau("./data/h2/fort.7703")
    pot.read_coulomb("./data/h2/fort.7704")
    pot.read_spinorbit("./data/h2/fort.7705")

    # prop = AFDMCPropagatorHS(n_particles, dt, include_prefactor=True, mix=True, seed=seed)
    prop = AFDMCPropagatorRBM(n_particles, dt, include_prefactor=True, mix=True, seed=seed)
    
    integ = Integrator(pot, prop)
    integ.controls["sigma"] = True
    integ.controls["sigmatau"] = False
    integ.controls["tau"] = False
    integ.controls["coulomb"] = False
    integ.controls["spinorbit"] = False
    integ.controls["balanced"] = True 
    integ.setup(n_samples=n_samples, seed=seed)
    
    b_array = integ.run(bra, ket, parallel=True)
    b_m = np.mean(b_array)
    b_s = np.std(b_array)/np.sqrt(n_samples)
    print(f'bracket = {b_m} +/- {b_s}')
    # chistogram(b_array, filename='hs_test.pdf', title='HS test')
    
    ex = ExactGFMC(n_particles)
    g_exact = ex.make_g_exact(n_particles, dt, pot, integ.controls)
    b_exact = bra * g_exact * ket
    print('exact = ',b_exact)

    print("ratio = ", np.abs(b_m)/np.abs(b_exact) )
    print("abs error = ", abs(1-np.abs(b_m)/np.abs(b_exact)) )

if __name__ == "__main__":
    main()
    