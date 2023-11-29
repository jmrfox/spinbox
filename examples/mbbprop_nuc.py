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

def g_pade_sig(dt, asig):
    out = ManyBodyBasisSpinIsospinOperator(2).zeros()
    for a in range(3):
        for b in range(3):
            out += asig[a, b] * sig0vec[a] * sig1vec[b]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_sigtau(dt, asigtau):
    out = ManyBodyBasisSpinIsospinOperator(2).zeros()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                out += asigtau[a, b, c] * sig0vec[a] * sig1vec[b] * tau0vec[c] * tau1vec[c]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_tau(dt, atau):
    out = ManyBodyBasisSpinIsospinOperator(2).zeros()
    for c in range(3):
        out += atau[c] * tau0vec[c] * tau1vec[c]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_coul(dt, v):
    out = ManyBodyBasisSpinIsospinOperator(2)
    out += tauz0 + tauz1 + tauz0 * tauz1
    out = -0.125 * v * dt * out
    return out.exponentiate()

def g_coul_onebody(dt,v):
    """just the one-body part of the expanded coulomb propagator
    for use along with auxiliary field propagators"""
    out = ManyBodyBasisSpinIsospinOperator(2)
    out += tauz0 + tauz1
    out = - 0.125 * v * dt * out
    return out.exponentiate()

def g_gauss_sample(dt, a, x, opi, opj):
    k = np.sqrt(-0.5 * dt * a, dtype=complex)
    norm = np.exp(0.5 * dt * a)
    gi = np.cosh(k * x) * one + np.sinh(k * x) * opi
    gj = np.cosh(k * x) * one + np.sinh(k * x) * opj
    return norm * gi * gj


def gauss_task(x, bra, ket, pot_dict):
    ket_p = ket.copy()
    ket_m = ket.copy()

    asig = pot_dict['asig']
    asigtau = pot_dict['asigtau']
    atau = pot_dict['atau']
    vcoul = pot_dict['vcoul']
    bls = pot_dict['bls']

    n = 0
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            ket_p = g_gauss_sample(nt.dt, asig[a, b], x[n], sig0vec[a], sig1vec[b]) * ket_p
            ket_m = g_gauss_sample(nt.dt, asig[a, b], -x[n], sig0vec[a], sig1vec[b]) * ket_m
            n += 1
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            for c in [0, 1, 2]:
                ket_p = g_gauss_sample(nt.dt, asigtau[a, b, c], x[n], sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_p
                ket_m = g_gauss_sample(nt.dt, asigtau[a, b, c], -x[n], sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_m
                n += 1
    for c in [0, 1, 2]:
        ket_p = g_gauss_sample(nt.dt, atau[c], x[n], tau0vec[c], tau1vec[c]) * ket_p
        ket_m = g_gauss_sample(nt.dt, atau[c], -x[n], tau0vec[c], tau1vec[c]) * ket_m
        n += 1

    ket_p = g_coul_onebody(nt.dt, vcoul) * g_gauss_sample(nt.dt, 0.25 * vcoul, x[n], tauz0, tauz1) * ket_p
    ket_m = g_coul_onebody(nt.dt, vcoul) * g_gauss_sample(nt.dt, 0.25 * vcoul, -x[n], tauz0, tauz1) * ket_m
    return 0.5 * (bra * ket_p + bra * ket_m)


def gaussian_brackets_parallel(n_samples=100, mix=False, plot=False, disable_tqdm=False):
    print('HS brackets')
    bra, ket = nt.make_test_states()
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()

    pot_dict = nt.make_potentials()
    asig = pot_dict['asig']
    asigtau = pot_dict['asigtau']
    atau = pot_dict['atau']
    vcoul = pot_dict['vcoul']
    bls = pot_dict['bls']

    # correct answer via pade
    b_exact = bra * g_pade_sig(nt.dt, asig) * g_pade_sigtau(nt.dt, asigtau) * g_pade_tau(nt.dt, atau) * g_pade_coul(nt.dt, vcoul) * ket

    n_aux = 9 + 27 + 3 + 1
    x_set = rng.standard_normal((n_samples, n_aux))  # different x for each x,y,z
    with Pool(processes=nt.n_procs) as pool:
        b_list = pool.starmap_async(gauss_task, tqdm([(x, bra, ket, pot_dict) for x in x_set], disable=disable_tqdm, leave=True)).get()

    if plot:
        range = None
        nt.plot_samples(b_list, range=range, filename=f'hsprop_mb{nt.run_tag}.pdf', title='HS (MBB)')

    b_gauss = np.mean(b_list)
    s_gauss = np.std(b_list) / np.sqrt(n_samples)
    print('exact = ', b_exact)
    print(f'gauss = {b_gauss}  +/-  {s_gauss}')
    print('error = ', b_exact - b_gauss)


def g_rbm_sample(dt, a, h, opi, opj):
    norm = np.exp(-0.5 * dt * np.abs(a)) / 2
    W = np.arctanh(np.sqrt(np.tanh(0.5 * dt * np.abs(a))))
    arg = W * (2 * h - 1)
    qi = np.cosh(arg) * one + np.sinh(arg) * opi
    qj = np.cosh(arg) * one - np.sign(a) * np.sinh(arg) * opj
    return 2 * norm * qi * qj


def rbm_task(h, bra, ket, pot_dict):
    ket_p = ket.copy()
    ket_m = ket.copy()
    
    asig = pot_dict['asig']
    asigtau = pot_dict['asigtau']
    atau = pot_dict['atau']
    vcoul = pot_dict['vcoul']
    bls = pot_dict['bls']
    
    n = 0
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            ket_p = g_rbm_sample(nt.dt, asig[a, b], h[n], sig0vec[a], sig1vec[b]) * ket_p
            ket_m = g_rbm_sample(nt.dt, asig[a, b], 1 - h[n], sig0vec[a], sig1vec[b]) * ket_m
            n += 1
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            for c in [0, 1, 2]:
                ket_p = g_rbm_sample(nt.dt, asigtau[a, b, c], h[n], sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_p
                ket_m = g_rbm_sample(nt.dt, asigtau[a, b, c], 1 - h[n], sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_m
                n += 1
    for c in [0, 1, 2]:
        ket_p = g_rbm_sample(nt.dt, atau[c], h[n], tau0vec[c], tau1vec[c]) * ket_p
        ket_m = g_rbm_sample(nt.dt, atau[c], 1 - h[n], tau0vec[c], tau1vec[c]) * ket_m
        n += 1

    ket_p = g_coul_onebody(nt.dt, vcoul) * g_rbm_sample(nt.dt, 0.25 * vcoul, h[n], tauz0, tauz1) * ket_p
    ket_m = g_coul_onebody(nt.dt, vcoul) * g_rbm_sample(nt.dt, 0.25 * vcoul, 1 - h[n], tauz0, tauz1) * ket_m
    n += 1
    return 0.5 * (bra * ket_p + bra * ket_m)


def rbm_brackets_parallel(n_samples=100, mix=False, plot=False, disable_tqdm=False):
    print('RBM brackets')
    bra, ket = nt.make_test_states()
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()

    pot_dict = nt.make_potentials()
    asig = pot_dict['asig']
    asigtau = pot_dict['asigtau']
    atau = pot_dict['atau']
    vcoul = pot_dict['vcoul']
    bls = pot_dict['bls']

    # correct answer via pade
    b_exact = bra * g_pade_sig(nt.dt, asig) * g_pade_sigtau(nt.dt, asigtau) * g_pade_tau(nt.dt, atau) * g_pade_coul(nt.dt, vcoul) * ket

    # make population of identical wfns

    n_aux = 9 + 27 + 3 + 1
    h_set = rng.integers(0, 2, size=(n_samples, n_aux))

    with Pool(processes=nt.n_procs) as pool:
        b_list = pool.starmap_async(rbm_task, tqdm([(h, bra, ket, pot_dict) for h in h_set], disable=disable_tqdm, leave=True)).get()

    if plot:
        range = None
        nt.plot_samples(b_list, range=range, filename=f'rbmprop_mb{nt.run_tag}.pdf', title='RBM (MBB)')

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
        gaussian_brackets_parallel(n_samples=n_samples, plot=plot, disable_tqdm=disable_tqdm)
        rbm_brackets_parallel(n_samples=n_samples, plot=plot, disable_tqdm=disable_tqdm)
        # Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()
    print('DONE')
