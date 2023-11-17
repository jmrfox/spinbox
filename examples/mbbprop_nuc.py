import nuctest as nt
from quap import *
from tqdm import tqdm
from cProfile import Profile
from pstats import SortKey, Stats
from multiprocessing.pool import Pool

one = ManyBodyBasisSpinIsospinOperator(2)
sigx0 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'x')
sigy0 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'y')
sigz0 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'z')
sigx1 = ManyBodyBasisSpinIsospinOperator(2).sigma(1, 'x')
sigy1 = ManyBodyBasisSpinIsospinOperator(2).sigma(1, 'y')
sigz1 = ManyBodyBasisSpinIsospinOperator(2).sigma(1, 'z')
sig0vec = [sigx0, sigy0, sigz0]
sig1vec = [sigx1, sigy1, sigz1]
taux0 = ManyBodyBasisSpinIsospinOperator(2).tau(0, 'x')
tauy0 = ManyBodyBasisSpinIsospinOperator(2).tau(0, 'y')
tauz0 = ManyBodyBasisSpinIsospinOperator(2).tau(0, 'z')
taux1 = ManyBodyBasisSpinIsospinOperator(2).tau(1, 'x')
tauy1 = ManyBodyBasisSpinIsospinOperator(2).tau(1, 'y')
tauz1 = ManyBodyBasisSpinIsospinOperator(2).tau(1, 'z')
tau0vec = [taux0, tauy0, tauz0]
tau1vec = [taux1, tauy1, tauz1]

def Gpade_sigma(dt, Amat):
    out = ManyBodyBasisSpinIsospinOperator(2).zeros()
    for a in range(3):
        for b in range(3):
            out += Amat[a, b] * sig0vec[a] * sig1vec[b]
    out = -0.5 * dt * out
    return out.exponentiate()


def Gpade_sigmatau(dt, Amat):
    out = ManyBodyBasisSpinIsospinOperator(2).zeros()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                out += Amat[a, b, c] * sig0vec[a] * sig1vec[b] * tau0vec[c] * tau1vec[c]
    out = -0.5 * dt * out
    return out.exponentiate()


def Gpade_tau(dt, Amat):
    out = ManyBodyBasisSpinIsospinOperator(2).zeros()
    for c in range(3):
        out += Amat[c] * tau0vec[c] * tau1vec[c]
    out = -0.5 * dt * out
    return out.exponentiate()


def Gpade_coul(dt, v):
    out = 0.25 * v * tauz0 * tauz1
    out = -0.5 * dt * out
    return out.exponentiate()



def Ggauss_sample(dt, A, x, opi, opj):
    k = np.sqrt(-0.5 * dt * A, dtype=complex)
    norm = np.exp(0.5 * dt * A)
    gi = np.cosh(k * x) * one + np.sinh(k * x) * opi
    gj = np.cosh(k * x) * one + np.sinh(k * x) * opj
    return norm * gi * gj


def gauss_task(x, bra, ket, Asig, Asigtau, Atau, Vcoul):
    ket_p = ket.copy()
    ket_m = ket.copy()
    n = 0
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            ket_p = Ggauss_sample(nt.dt, Asig[a, b], x[n], sig0vec[a], sig1vec[b]) * ket_p
            ket_m = Ggauss_sample(nt.dt, Asig[a, b], -x[n], sig0vec[a], sig1vec[b]) * ket_m
            n += 1
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            for c in [0, 1, 2]:
                ket_p = Ggauss_sample(nt.dt, Asigtau[a, b, c], x[n], sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_p
                ket_m = Ggauss_sample(nt.dt, Asigtau[a, b, c], -x[n], sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_m
                n += 1
    for c in [0, 1, 2]:
        ket_p = Ggauss_sample(nt.dt, Atau[c], x[n], tau0vec[c], tau1vec[c]) * ket_p
        ket_m = Ggauss_sample(nt.dt, Atau[c], -x[n], tau0vec[c], tau1vec[c]) * ket_m
        n += 1
    ket_p = Ggauss_sample(nt.dt, 0.25 * Vcoul, x[n], tauz0, tauz1) * ket_p
    ket_m = Ggauss_sample(nt.dt, 0.25 * Vcoul, -x[n], tauz0, tauz1) * ket_m
    return 0.5 * (bra * ket_p + bra * ket_m)


def gaussian_brackets_parallel(n_samples=100, mix=False, plot=False, disable_tqdm=False):
    print('HS brackets')
    Asig, Asigtau, Atau, Vcoul = nt.make_potentials(random=True)
    bra, ket = nt.make_test_states()
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()

    # correct answer via pade
    b_exact = bra * Gpade_sigma(nt.dt, Asig) * Gpade_sigmatau(nt.dt, Asigtau) * Gpade_tau(nt.dt, Atau) * Gpade_coul(nt.dt, Vcoul) * ket

    n_aux = 9 + 28 + 3 + 1
    x_set = rng.standard_normal((n_samples, n_aux))  # different x for each x,y,z
    with Pool(processes=nt.n_procs) as pool:
        b_list = pool.starmap_async(gauss_task, tqdm([(x, bra, ket, Asig, Asigtau, Atau, Vcoul) for x in x_set], disable=disable_tqdm, leave=True)).get()

    if plot:
        nt.plot_samples(b_list, range=(-2, 2), filename=f'hsprop_mb{nt.run_tag}.pdf', title='HS (MBB)')

    b_gauss = np.mean(b_list)
    s_gauss = np.std(b_list) / np.sqrt(n_samples)
    print('exact = ', b_exact)
    print(f'gauss = {b_gauss}  +/-  {s_gauss}')
    print('error = ', b_exact - b_gauss)


def Grbm_sample(dt, A, h, opi, opj):
    norm = np.exp(-0.5 * dt * np.abs(A)) / 2
    W = np.arctanh(np.sqrt(np.tanh(0.5 * dt * np.abs(A))))
    arg = W * (2 * h - 1)
    qi = np.cosh(arg) * one + np.sinh(arg) * opi
    qj = np.cosh(arg) * one - np.sign(A) * np.sinh(arg) * opj
    return 2 * norm * qi * qj


def rbm_task(h, bra, ket, Asig, Asigtau, Atau, Vcoul):
    ket_p = ket.copy()
    ket_m = ket.copy()
    n = 0
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            ket_p = Grbm_sample(nt.dt, Asig[a, b], h[n], sig0vec[a], sig1vec[b]) * ket_p
            ket_m = Grbm_sample(nt.dt, Asig[a, b], 1 - h[n], sig0vec[a], sig1vec[b]) * ket_m
            n += 1
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            for c in [0, 1, 2]:
                ket_p = Grbm_sample(nt.dt, Asigtau[a, b, c], h[n], sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_p
                ket_m = Grbm_sample(nt.dt, Asigtau[a, b, c], 1 - h[n], sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_m
                n += 1
    for c in [0, 1, 2]:
        ket_p = Grbm_sample(nt.dt, Atau[c], h[n], tau0vec[c], tau1vec[c]) * ket_p
        ket_m = Grbm_sample(nt.dt, Atau[c], 1 - h[n], tau0vec[c], tau1vec[c]) * ket_m
        n += 1

    ket_p = Grbm_sample(nt.dt, 0.25 * Vcoul, h[n], tauz0, tauz1) * ket_p
    ket_m = Grbm_sample(nt.dt, 0.25 * Vcoul, 1 - h[n], tauz0, tauz1) * ket_m
    n += 1
    return 0.5 * (bra * ket_p + bra * ket_m)


def rbm_brackets_parallel(n_samples=100, mix=False, plot=False, disable_tqdm=False):
    print('RBM brackets')
    Asig, Asigtau, Atau, Vcoul = nt.make_potentials(random=True)
    bra, ket = nt.make_test_states()
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()


    # correct answer via pade
    b_exact = bra * Gpade_sigma(nt.dt, Asig) * Gpade_sigmatau(nt.dt, Asigtau) * Gpade_tau(nt.dt, Atau) * Gpade_coul(nt.dt, Vcoul) * ket

    # make population of identical wfns

    n_aux = 9 + 28 + 3 + 1
    h_set = rng.integers(0, 2, size=(n_samples, n_aux))

    with Pool(processes=nt.n_procs) as pool:
        b_list = pool.starmap_async(rbm_task, tqdm([(h, bra, ket, Asig, Asigtau, Atau, Vcoul) for h in h_set], disable=disable_tqdm, leave=True)).get()

    if plot:
        nt.plot_samples(b_list, range=(-2, 2), filename=f'rbmprop_mb{nt.run_tag}.pdf', title='RBM (MBB)')

    b_rbm = np.mean(b_list)
    s_rbm = np.std(b_list) / np.sqrt(n_samples)
    print('exact = ', b_exact)
    print(f'rbm = {b_rbm}  +/-  {s_rbm}')
    print('error = ', b_exact - b_rbm)


if __name__ == "__main__":
    n_samples = nt.n_samples
    plot = True
    disable_tqdm = False
    with Profile() as profile:
        nt.ic(nt.make_potentials(random=True))
        gaussian_brackets_parallel(n_samples=n_samples, plot=plot, disable_tqdm=disable_tqdm)
        rbm_brackets_parallel(n_samples=n_samples, plot=plot, disable_tqdm=disable_tqdm)
        # Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()
    print('DONE')
