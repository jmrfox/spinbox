from spinbox import *
from spinbox.extras import chistogram
import time
import pickle as pkl

parallel = True
n_processes = None

def print_dict(d):
    for key in d.keys():
        print(f"{key} : {d[key]}")

def average_1d(n_particles, 
            dt = 0.001, 
            full_basis = False,
            isospin=True,
            n_samples = 100, 
            balance=True, 
            plot=False, 
            seed=0):
    
    seeder = itertools.count(seed, 1)

    ket = ProductState(n_particles, ketwise=True, isospin=isospin).random(seed=next(seeder))
    if full_basis:
        ket = ket.to_full_basis()
        
    bra = ket.copy().dagger()

    pot = NuclearPotential(n_particles)
    pot.sigma_3b[0,0,0,1,0,2] = 10.0  # coupling is all zero except the i=0 j=1 k=2 a=b=c=x element
        
    if full_basis:
        prop = HilbertPropagatorRBM(n_particles, dt, isospin=isospin)
    else:
        prop = ProductPropagatorRBM(n_particles, dt, isospin=isospin)
        
    integ = Integrator(pot, prop, isospin=isospin)

    t0 = time.time()
    integ.setup(n_samples=n_samples,
                seed=seed, 
                mix=False,
                flip_aux=False,
                sigma_3b=True, 
                parallel=parallel,
                n_processes=n_processes)
    b_plus = integ.run(bra, ket)
    if balance:
        integ.setup(n_samples=n_samples,
                    seed=seed, 
                    mix=False,
                    flip_aux=True,
                    sigma_3b=True, 
                    parallel=parallel,
                    n_processes=n_processes)
        b_minus = integ.run(bra, ket)
        b_array = np.concatenate([b_plus, b_minus])
    else:
        b_array = b_plus

    b_m = np.mean(b_array)
    b_s = np.std(b_array)/np.sqrt(n_samples)
    if full_basis:
        b_exact = integ.exact(bra, ket)
    else:
        b_exact = integ.exact(bra.to_full_basis(), ket.to_full_basis())
    ratio = np.abs(b_m)/np.abs(b_exact) 
    err = abs(1-np.abs(b_m)/np.abs(b_exact))
    
    print("<bra|ket> = ", bra.inner(ket) )
    print(f'<bra|G|ket> = {b_m} +/- {b_s}')
    print('exact = ',b_exact)
    print("ratio = ", ratio )
    print("error = ", err )
    print("dt^2 = ", dt**2)
    print("1/sqrt(N) = ", 1/np.sqrt(n_samples) )

    if plot:
        if full_basis:
            filename = f"examples/outputs/avg3b_hilbert_1d_A{n_particles}_N{n_samples}.pdf"
            chistogram(b_array, filename=filename, title='Hilbert space simulation')
        else:
            filename = f"examples/outputs/avg3b_product_1d_A{n_particles}_N{n_samples}.pdf"
            chistogram(b_array, filename=filename, title='Product space simulation')
    
    out = {"n_particles": n_particles,
        "dt": dt,
        "full_basis": full_basis,
        "n_samples": n_samples,
        "balance": balance,
        "b_array": b_array,
        "b_m": b_m,
        "b_s": b_s,
        "b_exact": b_exact,
        "ratio": ratio,
        "error": err,
        "parallel": parallel,
        "n_processes": n_processes,
        "time": time.time()-t0,
        }
    return out


def average_nuclear(n_particles, 
            dt = 0.001, 
            full_basis = False,
            isospin=True,
            n_samples = 100, 
            method = 'hs',
            sigma=False, 
            sigmatau=False, 
            tau=False, 
            coulomb=False, 
            spinorbit=False, 
            sigma_3b=False, 
            mix=True,
            balance=True, 
            plot=False, 
            seed=0):

    seeder = itertools.count(seed, 1)

    if full_basis:
        ket = ProductState(n_particles, ketwise=True, isospin=isospin).random(seed=next(seeder)).to_full_basis()
    else:
        ket = ProductState(n_particles, ketwise=True, isospin=isospin).random(seed=next(seeder))
        
    bra = ket.copy().dagger()

    pot = NuclearPotential(n_particles)
    # pot.random(seed=next(seeder))
    pot.sigma_3b[0,0,0,1,0,2] = -10.0
    pot.sigma_3b[1,0,1,1,1,2] = -10.0
    # pot.sigma_3b[2,0,2,1,2,2] = -10.0
    # pot.sigma_3b.random(-5,5, seed=next(seeder))
    print(pot.sigma_3b.coefficients.flatten())

    if full_basis:
        if method=='hs':
            prop = HilbertPropagatorHS(n_particles, dt, isospin=isospin)
        elif method=='rbm':
            prop = HilbertPropagatorRBM(n_particles, dt, isospin=isospin)
    else:
        if method=='hs':
            prop = ProductPropagatorHS(n_particles, dt, isospin=isospin)
        elif method=='rbm':
            prop = ProductPropagatorRBM(n_particles, dt, isospin=isospin)
        
    integ = Integrator(pot, prop, isospin=isospin)

    t0 = time.time()
    integ.setup(n_samples=n_samples,
                seed=seed, 
                mix=mix,
                flip_aux=False,
                sigma=sigma, 
                sigmatau=sigmatau, 
                tau=tau, 
                coulomb=coulomb, 
                spinorbit=spinorbit,
                sigma_3b=sigma_3b,
                parallel=parallel,
                n_processes=n_processes)
    b_plus = integ.run(bra, ket)
    if balance:
        integ.setup(n_samples=n_samples,
                seed=seed, 
                mix=mix,
                flip_aux=True,
                sigma=sigma, 
                sigmatau=sigmatau, 
                tau=tau, 
                coulomb=coulomb, 
                spinorbit=spinorbit,
                sigma_3b=sigma_3b,
                parallel=parallel,
                n_processes=n_processes)
        b_minus = integ.run(bra, ket)
        b_array = np.concatenate([b_plus, b_minus])
    else:
        b_array = b_plus

    b_m = np.mean(b_array)
    b_s = np.std(b_array)/np.sqrt(n_samples)
    if full_basis:
        b_exact = integ.exact(bra, ket)
    else:
        b_exact = integ.exact(bra.to_full_basis(), ket.to_full_basis())
    ratio = np.abs(b_m)/np.abs(b_exact) 
    err = abs(1-np.abs(b_m)/np.abs(b_exact))
    
    print("<bra|ket> = ", bra.inner(ket) )
    print(f'<bra|G|ket> = {b_m} +/- {b_s}')
    print('exact = ',b_exact)
    print("ratio = ", ratio )
    print("error = ", err )
    print("dt^2 = ", dt**2)
    print("1/sqrt(N) = ", 1/np.sqrt(n_samples) )

    if plot:
        if full_basis:
            filename = f"examples/outputs/hilbert_A{n_particles}_N{n_samples}_{method}.pdf"
            chistogram(b_array, filename=filename, title='Hilbert space simulation')
        else:
            filename = f"examples/outputs/product_A{n_particles}_N{n_samples}_{method}.pdf"
            chistogram(b_array, filename=filename, title='Product space simulation')
    
    out = {"n_particles": n_particles,
           "dt": dt,
           "full_basis": full_basis,
           "n_samples": n_samples,
           "method": method,
           "balance": balance,
           "sigma": sigma,
           "sigmatau": sigmatau,
           "tau": tau,
           "coulomb": coulomb,
           "spinorbit": spinorbit,
           "sigma_3b":sigma_3b,
           "mix": mix,
           "b_array": b_array,
           "b_m": b_m,
           "b_s": b_s,
           "b_exact": b_exact,
           "ratio": ratio,
           "error": err,
           "parallel": parallel,
           "n_processes": n_processes,
           "time": time.time()-t0,
           }
    return out


def main_1d():
    args = {
    "n_particles": 3,
    "n_samples": 10000,
    "dt": 0.01,
    "full_basis": True,
    "seed": 0,
    "balance": True,
    "plot":True
    }

    out = average_1d(**args)
    # print_dict(out)
    return out

def main():
    args = {
    "n_particles": 3,
    "n_samples": 100000,
    "dt": 0.01,
    "full_basis": False,
    "seed": 0,
    "method": "rbm",
    "balance": False,
    "mix": True,
    "sigma": False,
    "sigmatau": False,
    "tau": False,
    "coulomb": False,
    "spinorbit": False,
    "sigma_3b": True,
    "plot":False
    }

    out = average_nuclear(**args)
    # print_dict(out)
    
    tag = int(time.time())
    with open(f"examples/outputs/averaging_{tag}.pkl","wb") as f:
        pkl.dump(out, f)
    return out


def plot_from_pickle(filename_in, filename_out, title):
    with open(filename_in,"rb") as f:
        x = pkl.load(f)["b_array"]
        print(x)
        chistogram(x, filename_out, title, bins=50)


def list_from_dict(input_dict):
    """ produces a list of dicts from a dict of lists """
    keys, values = zip(*input_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def experiment():
    
    input_dict = {
    "n_particles": [2],
    "n_samples": [1000],
    "dt": np.logspace(-5, -2, 5), # [0.01, 0.001, 0.0001],
    "full_basis": [False],
    "seed": [0],
    "method": ["hs", "rbm"],
    "balance": [True, False],
    "mix": [True, False],
    "sigma": [True],
    "sigmatau": [True],
    "tau": [True],
    "coulomb": [True],
    "spinorbit": [False],
    "plot":[False]
    }

    args_list = list_from_dict(input_dict)
    out = []
    for args in args_list:
        out.append(average_nuclear(**args))
        
    tag = int(time.time())
    with open(f"examples/outputs/experiment_{tag}.pkl","wb") as f:
        pkl.dump(out, f)

if __name__ == "__main__":
    # main_1d()
    main()
    # experiment()
    
    # plot_from_pickle(".\\examples\\outputs\\averaging_1718939734.pkl", ".\\examples\\outputs\\test.pdf", "test")
     