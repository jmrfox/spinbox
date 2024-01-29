import nuctest as nt
from quap import *
from tqdm import tqdm
from cProfile import Profile
from pstats import SortKey, Stats
from multiprocessing.pool import Pool

num_particles = 2
ident = ManyBodyBasisSpinIsospinOperator(num_particles)
# list constructors make generating operators more streamlined
sig = [[ManyBodyBasisSpinIsospinOperator(num_particles).sigma(i,a) for a in [0, 1, 2]] for i in range(num_particles)]
tau = [[ManyBodyBasisSpinIsospinOperator(num_particles).tau(i,a) for a in [0, 1, 2]] for i in range(num_particles)]
# access like sig[particle][xyz]

def g_pade_sig(dt, asig):
    out = ManyBodyBasisSpinIsospinOperator(2).zeros()
    for a in range(3):
        for b in range(3):
            out += asig[a, b] * sig[0][a] * sig[1][b]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_sigtau(dt, asigtau):
    out = ManyBodyBasisSpinIsospinOperator(2).zeros()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                out += asigtau[a, b] * sig[0][a] * sig[1][b] * tau[0][c] * tau[1][c]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_tau(dt, atau):
    out = ManyBodyBasisSpinIsospinOperator(2).zeros()
    for c in range(3):
        out += atau * tau[0][c] * tau[1][c]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_coul(dt, v):
    out = ident + tau[0][2] + tau[1][2] + tau[0][2] * tau[1][2]
    out = -0.125 * v * dt * out
    return out.exponentiate()

def g_ls_linear(gls_ai, i, a):
    # linear approx to LS
    out = ident - 1.j * gls_ai * sig[i][a] 
    return out

def g_ls_onebody(gls_ai, i, a):
    # one-body part of the LS propagator factorization
    out = - 1.j * gls_ai * sig[i][a]
    return out.exponentiate()

def g_ls_twobody(gls_ai, gls_bj, i, j, a, b):
    # one-body part of the LS propagator factorization
    out = 0.5 * gls_ai * gls_bj * sig[i][a] * sig[j][b]
    return out.exponentiate()

def g_coul_onebody(dt,v):
    """just the one-body part of the expanded coulomb propagator
    for use along with auxiliary field propagators"""
    out = - 0.125 * v * dt * (ident + tau[0][2] + tau[1][2])
    return out.exponentiate()

def g_gauss_sample(dt, a, x, opi, opj):
    k = np.sqrt(-0.5 * dt * a, dtype=complex)
    norm = np.exp(0.5 * dt * a)
    gi = np.cosh(k * x) * ident + np.sinh(k * x) * opi
    gj = np.cosh(k * x) * ident + np.sinh(k * x) * opj
    return norm * gi * gj


def gauss_task(x, bra, ket, pot_dict, rng_mix=None):
    ket_p = ket.copy()
    ket_m = ket.copy()

    asig = pot_dict['asig']
    asigtau = pot_dict['asigtau']
    atau = pot_dict['atau']
    vcoul = pot_dict['vcoul']
    bls = pot_dict['bls']

    n = 0
    idx = [[0, 1, 2] for _ in range(6)]
    if rng_mix is not None: # no evidence that mixing helps.
        idx = rng_mix.choice(2,size=(6,3))
    # SIGMA
    for a in idx[0]:
        for b in idx[1]:
            ket_p = g_gauss_sample(nt.dt, asig[a, b], x[n], sig[0][a], sig[1][b]) * ket_p
            ket_m = g_gauss_sample(nt.dt, asig[a, b], -x[n], sig[0][a], sig[1][b]) * ket_m
            n += 1
    # SIGMA TAU
    for a in idx[2]:
        for b in idx[3]:
            for c in idx[4]:
                ket_p = g_gauss_sample(nt.dt, asigtau[a, b], x[n], sig[0][a] * tau[0][c], sig[1][b] * tau[1][c]) * ket_p
                ket_m = g_gauss_sample(nt.dt, asigtau[a, b], -x[n], sig[0][a] * tau[0][c], sig[1][b] * tau[1][c]) * ket_m
                n += 1
    # TAU
    for c in idx[5]:
        ket_p = g_gauss_sample(nt.dt, atau, x[n], tau[0][c], tau[1][c]) * ket_p
        ket_m = g_gauss_sample(nt.dt, atau, -x[n], tau[0][c], tau[1][c]) * ket_m
        n += 1

    # COULOMB
    ket_p = g_coul_onebody(nt.dt, vcoul) * g_gauss_sample(nt.dt, 0.25 * vcoul, x[n], tau[0][2], tau[1][2]) * ket_p
    ket_m = g_coul_onebody(nt.dt, vcoul) * g_gauss_sample(nt.dt, 0.25 * vcoul, -x[n], tau[0][2], tau[1][2]) * ket_m
    
    # LS

    return 0.5 * (bra * ket_p + bra * ket_m)


def gaussian_brackets_parallel(n_samples=100, mix=False, plot=False, disable_tqdm=False):
    print('HS brackets')
    bra, ket = nt.make_test_states()
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()

    pot_dict = nt.make_all_potentials(rng=default_rng(seed=nt.global_seed))
    asig = pot_dict['asig']
    asigtau = pot_dict['asigtau']
    atau = pot_dict['atau']
    vcoul = pot_dict['vcoul']
    bls = pot_dict['bls']

    # correct answer via pade
    b_exact = bra * g_pade_sig(nt.dt, asig) * g_pade_sigtau(nt.dt, asigtau) * g_pade_tau(nt.dt, atau) * g_pade_coul(nt.dt, vcoul) * ket

    n_aux = 9 + 27 + 3 + 1
    rng = default_rng(seed=nt.global_seed)
    x_set = rng.standard_normal((n_samples, n_aux))  # different x for each x,y,z
    with Pool(processes=nt.n_procs) as pool:
        b_list = pool.starmap_async(gauss_task, tqdm([(x, bra, ket, pot_dict) for x in x_set], disable=disable_tqdm, leave=True)).get()

    if plot:
        nt.plot_samples(b_list, filename=f'hsprop_mb{nt.run_tag}.pdf', title='HS (MBB)')

    b_gauss = np.mean(b_list)
    s_gauss = np.std(b_list) / np.sqrt(n_samples)
    print('exact = ', b_exact)
    print(f'gauss = {b_gauss}  +/-  {s_gauss}')
    print('error = ', b_exact - b_gauss)


def g_rbm_sample(dt, a, h, opi, opj):
    norm = np.exp(-0.5 * dt * np.abs(a))
    W = np.arctanh(np.sqrt(np.tanh(0.5 * dt * np.abs(a))))
    arg = W * (2 * h - 1)
    qi = np.cosh(arg) * ident + np.sinh(arg) * opi
    qj = np.cosh(arg) * ident - np.sign(a) * np.sinh(arg) * opj
    return norm * qi * qj


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
            ket_p = g_rbm_sample(nt.dt, asig[a, b], h[n], sig[0][a], sig[1][b]) * ket_p
            ket_m = g_rbm_sample(nt.dt, asig[a, b], 1 - h[n], sig[0][a], sig[1][b]) * ket_m
            n += 1
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            for c in [0, 1, 2]:
                ket_p = g_rbm_sample(nt.dt, asigtau[a, b], h[n], sig[0][a] * tau[0][c], sig[1][b] * tau[1][c]) * ket_p
                ket_m = g_rbm_sample(nt.dt, asigtau[a, b], 1 - h[n], sig[0][a] * tau[0][c], sig[1][b] * tau[1][c]) * ket_m
                n += 1
    for c in [0, 1, 2]:
        ket_p = g_rbm_sample(nt.dt, atau, h[n], tau[0][c], tau[1][c]) * ket_p
        ket_m = g_rbm_sample(nt.dt, atau, 1 - h[n], tau[0][c], tau[1][c]) * ket_m
        n += 1

    ket_p = g_coul_onebody(nt.dt, vcoul) * g_rbm_sample(nt.dt, 0.25 * vcoul, h[n], tau[0][2], tau[1][2]) * ket_p
    ket_m = g_coul_onebody(nt.dt, vcoul) * g_rbm_sample(nt.dt, 0.25 * vcoul, 1 - h[n], tau[0][2], tau[1][2]) * ket_m
    return 0.5 * (bra * ket_p + bra * ket_m)


def rbm_brackets_parallel(n_samples=100, mix=False, plot=False, disable_tqdm=False):
    print('RBM brackets')
    bra, ket = nt.make_test_states()
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()

    pot_dict = nt.make_all_potentials(rng=default_rng(seed=nt.global_seed))
    asig = pot_dict['asig']
    asigtau = pot_dict['asigtau']
    atau = pot_dict['atau']
    vcoul = pot_dict['vcoul']
    bls = pot_dict['bls']

    # correct answer via pade
    b_exact = bra * g_pade_sig(nt.dt, asig) * g_pade_sigtau(nt.dt, asigtau) * g_pade_tau(nt.dt, atau) * g_pade_coul(nt.dt, vcoul) * ket

    # make population of identical wfns

    n_aux = 9 + 27 + 3 + 1
    rng = default_rng(seed=nt.global_seed)
    h_set = rng.integers(0, 2, size=(n_samples, n_aux))

    with Pool(processes=nt.n_procs) as pool:
        b_list = pool.starmap_async(rbm_task, tqdm([(h, bra, ket, pot_dict) for h in h_set], disable=disable_tqdm, leave=True)).get()

    if plot:
        nt.plot_samples(b_list, filename=f'rbmprop_mb{nt.run_tag}.pdf', title='RBM (MBB)')

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
        bra, ket = nt.make_test_states()
        bracket_t0 = bra * ket
        print(f'<G(t=0)> = {bracket_t0}')
        # Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()
    print('DONE')
