from spinbox import *
from spinbox.extras import chistogram

def gfmc_avg(n_particles, dt, n_samples, method, controls: dict, balance=True, plot=False):
    seeder = itertools.count(controls["seed"], 1)

    ket = AFDMCSpinIsospinState(n_particles, np.zeros(shape=(n_particles, 4, 1)), ketwise=True).randomize(seed=next(seeder)).to_manybody_basis()
    bra = ket.copy().dagger()
    pot = NuclearPotential(n_particles)
    pot.sigma.generate(1.0, seed=next(seeder))
    pot.sigmatau.generate(1.0, seed=next(seeder))
    pot.tau.generate(1.0, seed=next(seeder))
    pot.coulomb.generate(0.1, seed=next(seeder))
    pot.spinorbit.generate(dt, seed=next(seeder))

    if method=='hs':
        prop = GFMCPropagatorHS(n_particles, dt)
    elif method=='rbm':
        prop = GFMCPropagatorRBM(n_particles, dt)
    
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
        filename = f"examples/outputs/gfmc_A{n_particles}_N{n_samples}_{method}.pdf"
        chistogram(b_array, filename=filename, title='GFMC')



def afdmc_avg(n_particles, dt, n_samples, method, controls: dict, balance=True, plot=False):
    seeder = itertools.count(controls["seed"], 1)

    ket = AFDMCSpinIsospinState(n_particles, np.zeros(shape=(n_particles, 4, 1)), ketwise=True).randomize(seed=next(seeder))
    bra = ket.copy().dagger()
    pot = NuclearPotential(n_particles)
    pot.sigma.generate(1.0, seed=next(seeder))
    pot.sigmatau.generate(1.0, seed=next(seeder))
    pot.tau.generate(1.0, seed=next(seeder))
    pot.coulomb.generate(0.1, seed=next(seeder))
    pot.spinorbit.generate(dt, seed=next(seeder))

    if method=='hs':
        prop = AFDMCPropagatorHS(n_particles, dt)
    elif method=='rbm':
        prop = AFDMCPropagatorRBM(n_particles, dt)
    
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
        filename = f"examples/outputs/afdmc_A{n_particles}_N{n_samples}_{method}.pdf"
        chistogram(b_array, filename=filename, title='AFDMC')

    out = {"n_particles": n_particles,
           "dt": dt,
           "n_samples": n_samples,
           "method": method,
           "balance": balance,
           "b_m": b_m,
           "b_s": b_s,
           "b_exact": b_exact
           }
    out.update(controls)
    return out



def main():
    n_particles = 2
    dt = 0.001
    n_samples = 100000

    method = 'rbm'
    controls = {"mix": True,
                "seed": 1,
                "sigma": True,
                "sigmatau": True,
                "tau": True,
                "coulomb": False,
                "spinorbit": False,
                }
    plot = True

    # gfmc_avg(n_particles, dt, n_samples, method, controls, plot=plot)
    afdmc_avg(n_particles, dt, n_samples, method, controls, plot=plot)
    


def list_from_dict(input_dict):
    """ produces a list of dicts from a dict of lists """
    return [dict(zip(input_dict,t)) for t in zip(*input_dict.values())]


def experiment():
    n_particles_list = [2]
    n_samples_list = [1000]
    # dt_list = [0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]
    dt_list = [0.001]
    method_list = ["rbm"]
    seed_list = [0]
    balance_list = [True, False]
    mix_list = [True, False]
    sigma_list = [True, False]
    sigmatau_list = [True, False]
    tau_list = [True, False]
    coulomb_list = [True, False]
    spinorbit_list = [True, False]
    
    input_dict = {
    "n_particles": n_particles_list,
    "n_samples": n_samples_list,
    "dt": dt_list,
    "seed": seed_list,
    "method": method_list,
    "balance": balance_list,
    "mix": mix_list,
    "sigma": sigma_list,
    "sigmatau": sigmatau_list,
    "tau": tau_list,
    "coulomb": coulomb_list,
    "spinorbit": spinorbit_list,
    }

    print(input_dict)
    # arg_grid = list_from_dict(input_dict)
    # print(arg_grid)


def test():
    DL = {'a': [0, 1], 'b': [2, 3]}
    v = [dict(zip(DL,t)) for t in zip(*DL.values())]
    print(v)

if __name__ == "__main__":
    main()