import nuctest as nt
from quap import *
from tqdm import tqdm
from cProfile import Profile
from pstats import SortKey, Stats
from multiprocessing.pool import Pool

one = OneBodyBasisSpinIsospinOperator(2)
sigx0 = OneBodyBasisSpinIsospinOperator(2).sigma(0, 'x')
sigy0 = OneBodyBasisSpinIsospinOperator(2).sigma(0, 'y')
sigz0 = OneBodyBasisSpinIsospinOperator(2).sigma(0, 'z')
sigx1 = OneBodyBasisSpinIsospinOperator(2).sigma(1, 'x')
sigy1 = OneBodyBasisSpinIsospinOperator(2).sigma(1, 'y')
sigz1 = OneBodyBasisSpinIsospinOperator(2).sigma(1, 'z')
sig0vec = [sigx0, sigy0, sigz0]
sig1vec = [sigx1, sigy1, sigz1]
taux0 = OneBodyBasisSpinIsospinOperator(2).tau(0, 'x')
tauy0 = OneBodyBasisSpinIsospinOperator(2).tau(0, 'y')
tauz0 = OneBodyBasisSpinIsospinOperator(2).tau(0, 'z')
taux1 = OneBodyBasisSpinIsospinOperator(2).tau(1, 'x')
tauy1 = OneBodyBasisSpinIsospinOperator(2).tau(1, 'y')
tauz1 = OneBodyBasisSpinIsospinOperator(2).tau(1, 'z')
tau0vec = [taux0, tauy0, tauz0]
tau1vec = [taux1, tauy1, tauz1]

def Ggauss_sample(dt: float, A: float, x, i: int, j: int, opi: OneBodyBasisSpinIsospinOperator, opj: OneBodyBasisSpinIsospinOperator):
    k = np.sqrt(-0.5 * dt * A, dtype=complex)
    norm = np.exp(0.5 * dt * A)
    op = OneBodyBasisSpinIsospinOperator(2)
    out = op.copy().scalar_mult(i, np.cosh(k * x)).scalar_mult(j, np.cosh(k * x)) + opi.scalar_mult(i, np.sinh(k * x)) * opj.scalar_mult(j, np.sinh(k * x))
    return out.spread_scalar_mult(norm)

def gauss_task(x, bra, ket, Asig, Asigtau, Atau, Vcoul):
    ket_p = ket.copy()
    ket_m = ket.copy()
    n = 0
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            ket_p = Ggauss_sample(nt.dt, Asig[a, b], x[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_p
            ket_m = Ggauss_sample(nt.dt, Asig[a, b], -x[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_m
            n += 1
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            for c in [0, 1, 2]:
                ket_p = Ggauss_sample(nt.dt, Asigtau[a, b, c], x[n], 0, 1, sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_p
                ket_m = Ggauss_sample(nt.dt, Asigtau[a, b, c], -x[n], 0, 1, sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_m
                n += 1
    for c in [0, 1, 2]:
        ket_p = Ggauss_sample(nt.dt, Atau[c], x[n], 0, 1, tau0vec[c], tau1vec[c]) * ket_p
        ket_m = Ggauss_sample(nt.dt, Atau[c], -x[n], 0, 1, tau0vec[c], tau1vec[c]) * ket_m
        n += 1
    ket_p = Ggauss_sample(nt.dt, 0.25 * Vcoul, x[n], 0, 1, tauz0, tauz1) * ket_p
    ket_m = Ggauss_sample(nt.dt, 0.25 * Vcoul, -x[n], 0, 1, tauz0, tauz1) * ket_m
    return 0.5 * (bra * ket_p + bra * ket_m)


def gaussian_brackets_parallel(n_samples=100, mix=False, plot=False, disable_tqdm=False):
    print('HS brackets')
    Asig, Asigtau, Atau, Vcoul = nt.make_potentials(random=True)
    bra, ket = nt.make_test_states()

    n_aux = 9 + 28 + 3 + 1
    x_set = rng.standard_normal((n_samples, n_aux))  # different x for each x,y,z
    with Pool(processes=nt.n_procs) as pool:
        b_list = pool.starmap_async(gauss_task, tqdm([(x, bra, ket, Asig, Asigtau, Atau, Vcoul) for x in x_set], disable=disable_tqdm)).get()

    if plot:
        nt.plot_samples(b_list, range=(-2, 2), filename=f'hsprop_ob{nt.run_tag}.pdf', title='HS (OBB)')

    b = np.mean(b_list)
    s = np.std(b_list) / np.sqrt(n_samples)
    print(f'gauss =  {b}  +/- {s}')


def Grbm_sample(dt, A, h, i, j, opi, opj):
    norm = np.exp(-0.5 * dt * np.abs(A)) / 2
    W = np.arctanh(np.sqrt(np.tanh(0.5 * dt * np.abs(A))))
    arg = W * (2 * h - 1)
    out = one.scalar_mult(i, np.cosh(arg)).scalar_mult(j, np.cosh(arg)) + opi.scalar_mult(i, np.sinh(arg)) * opj.scalar_mult(j, -np.sign(A) * np.sinh(arg))
    return out.spread_scalar_mult(2 * norm)
    # return out.scalar_mult(0, 2*norm)


def rbm_task(h, bra, ket, Asig, Asigtau, Atau, Vcoul):
    ket_p = ket.copy()
    ket_m = ket.copy()
    n = 0
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            ket_p = Grbm_sample(nt.dt, Asig[a, b], h[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_p
            ket_m = Grbm_sample(nt.dt, Asig[a, b], 1 - h[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_m
            n += 1
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            for c in [0, 1, 2]:
                ket_p = Grbm_sample(nt.dt, Asigtau[a, b, c], h[n], 0, 1, sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_p
                ket_m = Grbm_sample(nt.dt, Asigtau[a, b, c], 1 - h[n], 0, 1, sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_m
                n += 1
    for c in [0, 1, 2]:
        ket_p = Grbm_sample(nt.dt, Atau[c], h[n], 0, 1, tau0vec[c], tau1vec[c]) * ket_p
        ket_m = Grbm_sample(nt.dt, Atau[c], 1 - h[n], 0, 1, tau0vec[c], tau1vec[c]) * ket_m
        n += 1

    ket_p = Grbm_sample(nt.dt, 0.25 * Vcoul, h[n], 0, 1, tauz0, tauz1) * ket_p
    ket_m = Grbm_sample(nt.dt, 0.25 * Vcoul, 1 - h[n], 0, 1, tauz0, tauz1) * ket_m
    n += 1
    return 0.5 * (bra * ket_p + bra * ket_m)


def rbm_brackets_parallel(n_samples=100, mix=False, plot=False, disable_tqdm=False):
    print('RBM brackets')
    Asig, Asigtau, Atau, Vcoul = nt.make_potentials(random=True)
    bra, ket = nt.make_test_states()

    # make population of identical wfns

    n_aux = 9 + 28 + 3 + 1
    h_set = rng.integers(0, 2, size=(n_samples, n_aux))

    with Pool(processes=nt.n_procs) as pool:
        b_list = pool.starmap_async(rbm_task, tqdm([(h, bra, ket, Asig, Asigtau, Atau, Vcoul) for h in h_set], disable=disable_tqdm)).get()

    if plot:
        nt.plot_samples(b_list, range=(-2, 2), filename=f'rbmprop_ob{nt.run_tag}.pdf', title='RBM (OBB)')

    b = np.mean(b_list)
    s = np.std(b_list) / np.sqrt(n_samples)
    print(f'rbm =  {b}  +/- {s}')


if __name__ == "__main__":
    # test_gaussian_sample()
    # test_rbm_sample()

    n_samples = nt.n_samples
    plot = True
    disable_tqdm = False
    with Profile() as profile:
        gaussian_brackets_parallel(n_samples=n_samples, plot=plot, disable_tqdm=disable_tqdm)
        rbm_brackets_parallel(n_samples=n_samples, plot=plot, disable_tqdm=disable_tqdm)
        # Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()

    print('DONE')
