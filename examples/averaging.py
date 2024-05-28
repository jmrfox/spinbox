from quap import *
from quap.extras import chistogram
import pandas as pd

parallel = True

def print_dict(d):
    for key in d.keys():
        print(f"{key} : {d[key]}")

def hilbert_average(n_particles, dt, n_samples, method, controls: dict, balance=True, plot=False, isospin=True):
    seeder = itertools.count(controls["seed"], 1)

    ket = ProductState(n_particles, ketwise=True, isospin=isospin).randomize(seed=next(seeder)).to_manybody_basis()
    bra = ket.copy().dagger()
    
    pot = ArgonnePotential(n_particles)
    pot.sigma.generate(1.0, seed=next(seeder))
    pot.sigmatau.generate(1.0, seed=next(seeder))
    pot.tau.generate(1.0, seed=next(seeder))
    pot.coulomb.generate(0.1, seed=next(seeder))
    pot.spinorbit.generate(dt, seed=next(seeder))

    if method=='hs':
        prop = HilbertPropagatorHS(n_particles, dt, isospin=isospin)
    elif method=='rbm':
        prop = HilbertPropagatorRBM(n_particles, dt, isospin=isospin)
    
    integ = Integrator(pot, prop, isospin=isospin)

    integ.setup(n_samples=n_samples, **controls)
    b_plus = integ.run(bra, ket, parallel=parallel)
    if balance:
        integ.setup(n_samples=n_samples, **controls, flip_aux=True)
        b_minus = integ.run(bra, ket, parallel=parallel)
        b_array = np.concatenate([b_plus, b_minus])
    else:
        b_array = b_plus

    b_m = np.mean(b_array)
    b_s = np.std(b_array)/np.sqrt(n_samples)
    b_exact = integ.exact(bra, ket)
    ratio = np.abs(b_m)/np.abs(b_exact) 
    abs_error = abs(1-np.abs(b_m)/np.abs(b_exact))
    
    print("<bra|ket> = ", bra.inner(ket) )
    print(f'<bra|G|ket> = {b_m} +/- {b_s}')
    print('exact = ',b_exact)
    print("ratio = ", ratio )
    print("abs error = ", abs_error )
    print("dt^2 = ", dt**2)
    print("1/sqrt(N) = ", 1/np.sqrt(n_samples) )

    if plot:
        filename = f"examples/outputs/hilbert_A{n_particles}_N{n_samples}_{method}.pdf"
        chistogram(b_array, filename=filename, title='Hilbert space simulation')

    out = {"n_particles": n_particles,
           "dt": dt,
           "n_samples": n_samples,
           "method": method,
           "balance": balance,
           "b_m": b_m,
           "b_s": b_s,
           "b_exact": b_exact,
           "ratio": ratio,
           "abs_error": abs_error
           }
    out.update(controls)
    return out


def product_average(n_particles, dt, n_samples, method, controls: dict, balance=True, plot=False, isospin=True):
    seeder = itertools.count(controls["seed"], 1)

    ket = ProductState(n_particles, ketwise=True, isospin=isospin).randomize(seed=next(seeder))
    bra = ket.copy().dagger()
    pot = ArgonnePotential(n_particles)
    pot.sigma.generate(1.0, seed=next(seeder))
    pot.sigmatau.generate(1.0, seed=next(seeder))
    pot.tau.generate(1.0, seed=next(seeder))
    pot.coulomb.generate(0.1, seed=next(seeder))
    pot.spinorbit.generate(dt, seed=next(seeder))

    if method=='hs':
        prop = ProductPropagatorHS(n_particles, dt, isospin=isospin)
    elif method=='rbm':
        prop = ProductPropagatorRBM(n_particles, dt, isospin=isospin)
    
    integ = Integrator(pot, prop, isospin=isospin)

    # integ.setup(n_samples=n_samples, **controls)
    integ.setup(**controls)
    b_plus = integ.run(bra, ket, parallel=parallel)
    if balance:
        integ.setup(n_samples=n_samples, **controls, flip_aux=True)
        b_minus = integ.run(bra, ket, parallel=parallel)
        b_array = np.concatenate([b_plus, b_minus])
    else:
        b_array = b_plus

    b_m = np.mean(b_array)
    b_s = np.std(b_array)/np.sqrt(n_samples)
    b_exact = integ.exact(bra.to_manybody_basis(), ket.to_manybody_basis())
    ratio = np.abs(b_m)/np.abs(b_exact) 
    abs_error = abs(1-np.abs(b_m)/np.abs(b_exact))

    print("<bra|ket> = ", bra.inner(ket))
    print(f'<bra|G|ket> = {b_m} +/- {b_s}')
    print('exact = ',b_exact)
    print("ratio = ", ratio )
    print("abs error = ", abs_error )
    print("dt^2 = ", dt**2)
    print("1/sqrt(N) = ", 1/np.sqrt(n_samples) )

    if plot:
        filename = f"examples/outputs/product_A{n_particles}_N{n_samples}_{method}.pdf"
        chistogram(b_array, filename=filename, title='Product space simulation')

    out = {"n_particles": n_particles,
           "dt": dt,
           "n_samples": n_samples,
           "method": method,
           "balance": balance,
           "b_array": b_array,
           "b_m": b_m,
           "b_s": b_s,
           "b_exact": b_exact,
            "ratio": ratio,
           "abs_error": abs_error
           }
    out.update(controls)
    return out



def main():
    n_particles = 2
    dt = 0.001
    n_samples = 100000
    method = 'rbm'
    isospin = True
    balance = True
    plot = False

    controls = {"mix": True,
                "seed": 0,
                "sigma": True,
                "sigmatau": False,
                "tau": False,
                "coulomb": False,
                "spinorbit": False,
                }
    
    hilbert_out = hilbert_average(n_particles, dt, n_samples, method, controls, balance, plot, isospin)
    product_out = product_average(n_particles, dt, n_samples, method, controls, balance, plot, isospin)
    print_dict(hilbert_out)
    print_dict(product_out)
    
def list_from_dict(input_dict):
    """ produces a list of dicts from a dict of lists """
    keys, values = zip(*input_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def experiment():
    n_particles_list = [2]
    n_samples_list = [1000]
    # dt_list = [0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]
    dt_list = [0.001]
    method_list = ["hs","rbm"]
    seed_list = [0]
    balance_list = [True, False]
    mix_list = [True, False]
    sigma_list = [True]
    sigmatau_list = [False]
    tau_list = [False]
    coulomb_list = [False]
    spinorbit_list = [False]
    
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

    # print(input_dict)
    arg_grid = list_from_dict(input_dict)
    # print(arg_grid)
    out = []
    for arg in arg_grid:
        out.append(product_average(arg["n_particles"], arg["dt"], arg["n_samples"], arg["method"], controls=arg, balance=arg["balance"], plot=False, isospin=True))
        
    df = pd.DataFrame.from_dict(out)
    df.to_csv("test.csv")


if __name__ == "__main__":
    # main()
    experiment()
    