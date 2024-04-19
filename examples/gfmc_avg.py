from quap import *

n_particles = 2
dt = 0.001
n_samples = 10000
seed = 11

def main():

    ket = AFDMCSpinIsospinState(n_particles,'ket', read_from_file("./data/h2/fort.770",complex=True, shape=(2,4,1))).to_manybody_basis()
    bra = ket.copy().dagger()

    pot = ArgonnePotential(n_particles)
    pot.read_sigma("./data/h2/fort.7701")
    pot.read_sigmatau("./data/h2/fort.7702")
    pot.read_tau("./data/h2/fort.7703")
    pot.read_coulomb("./data/h2/fort.7704")
    pot.read_spinorbit("./data/h2/fort.7705")

    # prop = GFMCPropagatorHS(n_particles, dt, include_prefactor=True, mix=True, seed=seed)
    prop = GFMCPropagatorRBM(n_particles, dt, include_prefactor=True, mix=True)
    
    integ = Integrator(pot, prop)
    integ.setup(n_samples=n_samples, 
                balance=True,
                mix=True,
                seed=seed,
                sigma=True,
                sigmatau=True,
                tau=True,
                coulomb=False,
                spinorbit=False)
    
    b_array = integ.run(bra, ket, parallel=True)
    b_m = np.mean(b_array)
    b_s = np.std(b_array)/np.sqrt(n_samples)
    b_exact = integ.exact(bra, ket)

    print("<bra|ket> = ", bra * ket)
    print(f'<bra|G|ket> = {b_m} +/- {b_s}')
    print('exact = ',b_exact)
    print("ratio = ", np.abs(b_m)/np.abs(b_exact) )
    print("abs error = ", abs(1-np.abs(b_m)/np.abs(b_exact)) )

   # chistogram(b_array, filename='hs_test.pdf', title='HS test')

if __name__ == "__main__":
    main()
    