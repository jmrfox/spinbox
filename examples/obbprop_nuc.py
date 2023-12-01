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




def g_coul_onebody(dt, v):
    """just the one-body part of the expanded coulomb propagator
    for use along with auxiliary field propagators"""
    k = - 0.125 * v * dt
    norm = np.exp(k)
    ck, sk = np.cosh(k), np.sinh(k)
    out = one.scalar_mult(0, ck).scalar_mult(1, ck) + tauz0.scalar_mult(0, sk) * tauz1.scalar_mult(1, sk)
    return out.spread_scalar_mult(norm)

def g_gauss_sample(dt: float, a: float, x, i: int, j: int, opi: OneBodyBasisSpinIsospinOperator, opj: OneBodyBasisSpinIsospinOperator):
    k = np.sqrt(-0.5 * dt * a, dtype=complex)
    norm = np.exp(0.5 * dt * a)
    out = one.scalar_mult(i, np.cosh(k * x)).scalar_mult(j, np.cosh(k * x)) + opi.scalar_mult(i, np.sinh(k * x)) * opj.scalar_mult(j, np.sinh(k * x))
    return out.spread_scalar_mult(norm)


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
            ket_p = g_gauss_sample(nt.dt, asig[a, b], x[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_p
            ket_m = g_gauss_sample(nt.dt, asig[a, b], -x[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_m
            n += 1
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            for c in [0, 1, 2]:
                ket_p = g_gauss_sample(nt.dt, asigtau[a, b, c], x[n], 0, 1, sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_p
                ket_m = g_gauss_sample(nt.dt, asigtau[a, b, c], -x[n], 0, 1, sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_m
                n += 1
    for c in [0, 1, 2]:
        ket_p = g_gauss_sample(nt.dt, atau[c], x[n], 0, 1, tau0vec[c], tau1vec[c]) * ket_p
        ket_m = g_gauss_sample(nt.dt, atau[c], -x[n], 0, 1, tau0vec[c], tau1vec[c]) * ket_m
        n += 1
    ket_p = g_coul_onebody(nt.dt, vcoul) * g_gauss_sample(nt.dt, 0.25 * vcoul, x[n], 0, 1, tauz0, tauz1) * ket_p
    ket_m = g_coul_onebody(nt.dt, vcoul) * g_gauss_sample(nt.dt, 0.25 * vcoul, -x[n], 0, 1, tauz0, tauz1) * ket_m
    # ket_p = g_coul_onebody(nt.dt, vcoul) * ket_p
    # ket_m = g_coul_onebody(nt.dt, vcoul) * ket_m
    return 0.5 * (bra * ket_p + bra * ket_m)

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
            ket_p = g_gauss_sample(nt.dt, asig[a, b], x[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_p
            ket_m = g_gauss_sample(nt.dt, asig[a, b], -x[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_m
            n += 1
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            for c in [0, 1, 2]:
                ket_p = g_gauss_sample(nt.dt, asigtau[a, b, c], x[n], 0, 1, sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_p
                ket_m = g_gauss_sample(nt.dt, asigtau[a, b, c], -x[n], 0, 1, sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_m
                n += 1
    for c in [0, 1, 2]:
        ket_p = g_gauss_sample(nt.dt, atau[c], x[n], 0, 1, tau0vec[c], tau1vec[c]) * ket_p
        ket_m = g_gauss_sample(nt.dt, atau[c], -x[n], 0, 1, tau0vec[c], tau1vec[c]) * ket_m
        n += 1
    ket_p = g_coul_onebody(nt.dt, vcoul) * g_gauss_sample(nt.dt, 0.25 * vcoul, x[n], 0, 1, tauz0, tauz1) * ket_p
    ket_m = g_coul_onebody(nt.dt, vcoul) * g_gauss_sample(nt.dt, 0.25 * vcoul, -x[n], 0, 1, tauz0, tauz1) * ket_m
    # ket_p = g_coul_onebody(nt.dt, vcoul) * ket_p
    # ket_m = g_coul_onebody(nt.dt, vcoul) * ket_m
    return 0.5 * (bra * ket_p + bra * ket_m)


def gaussian_brackets_parallel(n_samples=100, mix=False, plot=False, disable_tqdm=False):
    print('HS brackets')
    bra, ket = nt.make_test_states()

    pot_dict = nt.make_all_potentials(rng=default_rng(seed=nt.global_seed))
    
    n_aux = 9 + 27 + 3 + 1
    rng = default_rng(seed=nt.global_seed)
    x_set = rng.standard_normal((n_samples, n_aux))  # different x for each x,y,z
    with Pool(processes=nt.n_procs) as pool:
        b_list = pool.starmap_async(gauss_task, tqdm([(x, bra, ket, pot_dict) for x in x_set], disable=disable_tqdm)).get()

    if plot:
        nt.plot_samples(b_list, range=(-2, 2), filename=f'hsprop_ob{nt.run_tag}.pdf', title='HS (OBB)')

    b = np.mean(b_list)
    s = np.std(b_list) / np.sqrt(n_samples)
    print(f'gauss =  {b}  +/- {s}')


def g_rbm_sample(dt, a, h, i, j, opi, opj):
    norm = np.exp(-0.5 * dt * np.abs(a)) / 2
    W = np.arctanh(np.sqrt(np.tanh(0.5 * dt * np.abs(a))))
    arg = W * (2 * h - 1)
    out = one.scalar_mult(i, np.cosh(arg)).scalar_mult(j, np.cosh(arg)) + opi.scalar_mult(i, np.sinh(arg)) * opj.scalar_mult(j, -np.sign(a) * np.sinh(arg))
    return out.spread_scalar_mult(2 * norm)
    # return out.scalar_mult(0, 2*norm)


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
            ket_p = g_rbm_sample(nt.dt, asig[a, b], h[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_p
            ket_m = g_rbm_sample(nt.dt, asig[a, b], 1 - h[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_m
            n += 1
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            for c in [0, 1, 2]:
                ket_p = g_rbm_sample(nt.dt, asigtau[a, b, c], h[n], 0, 1, sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_p
                ket_m = g_rbm_sample(nt.dt, asigtau[a, b, c], 1 - h[n], 0, 1, sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_m
                n += 1
    for c in [0, 1, 2]:
        ket_p = g_rbm_sample(nt.dt, atau[c], h[n], 0, 1, tau0vec[c], tau1vec[c]) * ket_p
        ket_m = g_rbm_sample(nt.dt, atau[c], 1 - h[n], 0, 1, tau0vec[c], tau1vec[c]) * ket_m
        n += 1

    # ket_p = g_coul_onebody(nt.dt, vcoul) * g_rbm_sample(nt.dt, 0.25 * vcoul, h[n], 0, 1, tauz0, tauz1) * ket_p
    # ket_m = g_coul_onebody(nt.dt, vcoul) * g_rbm_sample(nt.dt, 0.25 * vcoul, 1 - h[n], 0, 1, tauz0, tauz1) * ket_m
    ket_p = g_coul_onebody(nt.dt, vcoul) * ket_p
    ket_m = g_coul_onebody(nt.dt, vcoul) * ket_m
    return 0.5 * (bra * ket_p + bra * ket_m)


def rbm_brackets_parallel(n_samples=100, mix=False, plot=False, disable_tqdm=False):
    print('RBM brackets')
    bra, ket = nt.make_test_states()

    pot_dict = nt.make_all_potentials(rng=default_rng(seed=nt.global_seed))
    
    n_aux = 9 + 27 + 3 + 1
    rng = default_rng(seed=nt.global_seed)
    h_set = rng.integers(0, 2, size=(n_samples, n_aux))

    with Pool(processes=nt.n_procs) as pool:
        b_list = pool.starmap_async(rbm_task, tqdm([(h, bra, ket, pot_dict) for h in h_set], disable=disable_tqdm)).get()

    if plot:
        nt.plot_samples(b_list, range=(-2, 2), filename=f'rbmprop_ob{nt.run_tag}.pdf', title='RBM (OBB)')

    b = np.mean(b_list)
    s = np.std(b_list) / np.sqrt(n_samples)
    print(f'rbm =  {b}  +/- {s}')


if __name__ == "__main__":
    n_samples = nt.n_samples
    plot = True
    disable_tqdm = False
    with Profile() as profile:
        gaussian_brackets_parallel(n_samples=n_samples, plot=plot, disable_tqdm=disable_tqdm)
        rbm_brackets_parallel(n_samples=n_samples, plot=plot, disable_tqdm=disable_tqdm)
        bra, ket = nt.make_test_states()
        instant_bracket = bra * ket
        print(f'<G(t=0)> = {instant_bracket}')
        # Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()

    print('DONE')
