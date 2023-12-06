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
                out += asigtau[a, b, c] * sig[0][a] * sig[1][b] * tau[0][c] * tau[1][c]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_tau(dt, atau):
    out = ManyBodyBasisSpinIsospinOperator(2).zeros()
    for c in range(3):
        out += atau[c] * tau[0][c] * tau[1][c]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_coul(dt, v):
    out = ident + tau[0][2] + tau[1][2] + tau[0][2] * tau[1][2]
    # out = one + tau[0][2] + tau[1][2]
    out = -0.125 * v * dt * out
    return out.exponentiate()


def g_coul_onebody(dt,v):
    """just the one-body part of the expanded coulomb propagator
    for use along with auxiliary field propagators"""
    out = - 0.125 * v * dt * (ident + tau[0][2] + tau[1][2])
    return out.exponentiate()


def g_linear_ls(dt, bls):
    # linear approx to LS
    out = ident
    for a in range(3):
         out += 0.5j * bls[a] * ( sig[0][a] + sig[1][a] ) 
    return out

def g_ls_onebody(dt, bls):
    # one-body part of the LS propagator factorization
    out = ident.copy()
    for i in range(num_particles):
        for a in range(3):
            k = bls[a]   # for 2 particles, B_ija = g_ia
            out = (np.cos(k) * ident + 1.0j * np.sin(k) * sig[i][a]) * out
    return out


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
    for a in idx[0]:
        for b in idx[1]:
            ket_p = g_gauss_sample(nt.dt, asig[a, b], x[n], sig[0][a], sig[1][b]) * ket_p
            ket_m = g_gauss_sample(nt.dt, asig[a, b], -x[n], sig[0][a], sig[1][b]) * ket_m
            n += 1
    for a in idx[2]:
        for b in idx[3]:
            for c in idx[4]:
                ket_p = g_gauss_sample(nt.dt, asigtau[a, b, c], x[n], sig[0][a] * tau[0][c], sig[1][b] * tau[1][c]) * ket_p
                ket_m = g_gauss_sample(nt.dt, asigtau[a, b, c], -x[n], sig[0][a] * tau[0][c], sig[1][b] * tau[1][c]) * ket_m
                n += 1
    for c in idx[5]:
        ket_p = g_gauss_sample(nt.dt, atau[c], x[n], tau[0][c], tau[1][c]) * ket_p
        ket_m = g_gauss_sample(nt.dt, atau[c], -x[n], tau[0][c], tau[1][c]) * ket_m
        n += 1

    ket_p = g_coul_onebody(nt.dt, vcoul) * g_gauss_sample(nt.dt, 0.25 * vcoul, x[n], tau[0][2], tau[1][2]) * ket_p
    ket_m = g_coul_onebody(nt.dt, vcoul) * g_gauss_sample(nt.dt, 0.25 * vcoul, -x[n], tau[0][2], tau[1][2]) * ket_m

    ket_p = g_ls_onebody(nt.dt, bls) * ket_p
    ket_m = g_ls_onebody(nt.dt, bls) * ket_m

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
    b_exact = bra * g_pade_sig(nt.dt, asig) * g_pade_sigtau(nt.dt, asigtau) * g_pade_tau(nt.dt, atau) * g_pade_coul(nt.dt, vcoul) * g_linear_ls(nt.dt, bls) * ket

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
    norm = np.exp(-0.5 * dt * np.abs(a)) / 2
    W = np.arctanh(np.sqrt(np.tanh(0.5 * dt * np.abs(a))))
    arg = W * (2 * h - 1)
    qi = np.cosh(arg) * ident + np.sinh(arg) * opi
    qj = np.cosh(arg) * ident - np.sign(a) * np.sinh(arg) * opj
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
            ket_p = g_rbm_sample(nt.dt, asig[a, b], h[n], sig[0][a], sig[1][b]) * ket_p
            ket_m = g_rbm_sample(nt.dt, asig[a, b], 1 - h[n], sig[0][a], sig[1][b]) * ket_m
            n += 1
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            for c in [0, 1, 2]:
                ket_p = g_rbm_sample(nt.dt, asigtau[a, b, c], h[n], sig[0][a] * tau[0][c], sig[1][b] * tau[1][c]) * ket_p
                ket_m = g_rbm_sample(nt.dt, asigtau[a, b, c], 1 - h[n], sig[0][a] * tau[0][c], sig[1][b] * tau[1][c]) * ket_m
                n += 1
    for c in [0, 1, 2]:
        ket_p = g_rbm_sample(nt.dt, atau[c], h[n], tau[0][c], tau[1][c]) * ket_p
        ket_m = g_rbm_sample(nt.dt, atau[c], 1 - h[n], tau[0][c], tau[1][c]) * ket_m
        n += 1

    ket_p = g_coul_onebody(nt.dt, vcoul) * g_rbm_sample(nt.dt, 0.25 * vcoul, h[n], tau[0][2], tau[1][2]) * ket_p
    ket_m = g_coul_onebody(nt.dt, vcoul) * g_rbm_sample(nt.dt, 0.25 * vcoul, 1 - h[n], tau[0][2], tau[1][2]) * ket_m
    # ket_p = g_coul_onebody(nt.dt, vcoul) * ket_p
    # ket_m = g_coul_onebody(nt.dt, vcoul) * ket_m
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
