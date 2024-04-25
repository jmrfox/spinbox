from quap import *
from quap.extras import chistogram

def gfmc_avg(n_particles, dt, n_samples, method, controls, balance=True, plot=False):
    seeder = itertools.count(controls["seed"], 1)

    ket = AFDMCSpinIsospinState(n_particles, np.zeros(shape=(n_particles, 4, 1)), ketwise=True).randomize(seed=next(seeder)).to_manybody_basis()
    bra = ket.copy().dagger()
    pot = ArgonnePotential(n_particles)
    pot.sigma.generate(1.0, seed=next(seeder))
    pot.sigmatau.generate(1.0, seed=next(seeder))
    pot.tau.generate(1.0, seed=next(seeder))
    pot.coulomb.generate(0.1, seed=next(seeder))
    pot.spinorbit.generate(dt, seed=next(seeder))

    if method=='hs':
        prop = GFMCPropagatorHS(n_particles, dt, mix=True, seed=next(seeder))
    elif method=='rbm':
        prop = GFMCPropagatorRBM(n_particles, dt, mix=True, seed=next(seeder))
    
    integ = Integrator(pot, prop)

    integ.setup(n_samples=n_samples, **controls)
    b_plus = integ.run(bra, ket, parallel=True)
    if balance:
        integ.setup(n_samples=n_samples, **controls, flip_aux=True)
        b_minus = integ.run(bra, ket, parallel=True)
        b_array = np.concatenate([b_plus, b_minus])
    else:
        b_array = b_plus

    b_m = np.mean(b_array)
    b_s = np.std(b_array)/np.sqrt(n_samples)
    b_exact = integ.exact(bra, ket)

    print("<bra|ket> = ", bra * ket)
    print(f'<bra|G|ket> = {b_m} +/- {b_s}')
    print('exact = ',b_exact)
    print("ratio = ", np.abs(b_m)/np.abs(b_exact) )
    print("abs error = ", abs(1-np.abs(b_m)/np.abs(b_exact)) )
    print("dt^2 = ", dt**2)
    print("1/sqrt(N) = ", 1/np.sqrt(n_samples) )

    if plot:
        chistogram(b_array, filename='./outputs/gfmc_avg_test.pdf', title='GFMC')



def afdmc_avg(n_particles, dt, n_samples, method, controls, balance=True, plot=False):
    seeder = itertools.count(controls["seed"], 1)

    ket = AFDMCSpinIsospinState(n_particles, np.zeros(shape=(n_particles, 4, 1)), ketwise=True).randomize(seed=next(seeder))
    bra = ket.copy().dagger()
    pot = ArgonnePotential(n_particles)
    pot.sigma.generate(1.0, seed=next(seeder))
    pot.sigmatau.generate(1.0, seed=next(seeder))
    pot.tau.generate(1.0, seed=next(seeder))
    pot.coulomb.generate(0.1, seed=next(seeder))
    pot.spinorbit.generate(dt, seed=next(seeder))

    if method=='hs':
        prop = AFDMCPropagatorHS(n_particles, dt, mix=True, seed=next(seeder))
    elif method=='rbm':
        prop = AFDMCPropagatorRBM(n_particles, dt, mix=True, seed=next(seeder))
    
    integ = Integrator(pot, prop)

    integ.setup(n_samples=n_samples, **controls)
    b_plus = integ.run(bra, ket, parallel=True)
    if balance:
        integ.setup(n_samples=n_samples, **controls, flip_aux=True)
        b_minus = integ.run(bra, ket, parallel=True)
        b_array = np.concatenate([b_plus, b_minus])
    else:
        b_array = b_plus

    b_m = np.mean(b_array)
    b_s = np.std(b_array)/np.sqrt(n_samples)

    b_exact = integ.exact(bra.to_manybody_basis(), ket.to_manybody_basis())

    print("<bra|ket> = ", bra * ket)
    print(f'<bra|G|ket> = {b_m} +/- {b_s}')
    print('exact = ',b_exact)
    print("ratio = ", np.abs(b_m)/np.abs(b_exact) )
    print("abs error = ", abs(1-np.abs(b_m)/np.abs(b_exact)) )
    print("dt^2 = ", dt**2)
    print("1/sqrt(N) = ", 1/np.sqrt(n_samples) )

    if plot:
        chistogram(b_array, filename='./outputs/afdmc_avg_test.pdf', title='AFDMC')



if __name__ == "__main__":

    n_particles = 2
    dt = 0.001
    n_samples = 10000

    method = 'rbm'
    controls = {"mix": True,
                "seed": 0,
                "sigma": True,
                "sigmatau": True,
                "tau": True,
                "coulomb": True,
                "spinorbit": True,
                }
    
    # gfmc_avg(n_particles, dt, n_samples, method, controls)
    afdmc_avg(n_particles, dt, n_samples, method, controls)
    