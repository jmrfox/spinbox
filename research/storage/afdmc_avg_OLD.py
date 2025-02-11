from spinbox import *
from tqdm import tqdm
from cProfile import Profile
from pstats import SortKey, Stats
from multiprocessing.pool import Pool

ident = AFDMCSpinIsospinOperator(num_particles)
# list constructors make generating operators more streamlined
sig = [[AFDMCSpinIsospinOperator(num_particles).sigma(i,a) for a in [0, 1, 2]] for i in range(num_particles)]
tau = [[AFDMCSpinIsospinOperator(num_particles).tau(i,a) for a in [0, 1, 2]] for i in range(num_particles)]
# access like sig[particle][xyz]


def g_coulomb_onebody(dt, v, i):
    """just the one-body part of the factored coulomb propagator
    for use along with auxiliary field propagator for 2 body part"""
    k = - 0.125 * v * dt
    out = AFDMCSpinIsospinOperator(nt.n_particles)
    out.op_stack[i] = ccosh(k) * ident + csinh(k) * tau[2]
    return out

def g_ls_onebody(gls_ai, i, a):
    """just the one-body part of the factored LS propagator
    """
    k = - 1.j * gls_ai
    ck, sk = ccosh(k), csinh(k)
    out = AFDMCSpinIsospinOperator(nt.n_particles)
    out.op_stack[i] = ck * ident + sk * sig[a]
    return out


def g_gauss_sample(dt: float, a: float, x, i: int, j: int, opi: AFDMCSpinIsospinOperator, opj: AFDMCSpinIsospinOperator):
    k = csqrt(-0.5 * dt * a)
    norm = cexp(0.5 * dt * a)
    # out = ident.scalar_mult(i, ccosh(k * x)).scalar_mult(j, ccosh(k * x)) + opi.scalar_mult(i, csinh(k * x)) * opj.scalar_mult(j, csinh(k * x))
    # return out.spread_scalar_mult(norm)
    out = AFDMCSpinIsospinOperator(nt.n_particles)
    out.op_stack[i] = ccosh(k) * ident + csinh(k) * opi
    out.op_stack[j] = ccosh(k) * ident + csinh(k) * opj
    out.op_stack[i] *= csqrt(norm)
    out.op_stack[j] *= csqrt(norm)
    return out


def g_rbm_sample(dt, a, h, i, j, opi, opj):
    norm = cexp(-0.5 * dt * np.abs(a))
    W = carctanh(csqrt(ctanh(0.5 * dt * np.abs(a))))
    arg = W * (2 * h - 1)
    # out = ident.scalar_mult(i, ccosh(arg)).scalar_mult(j, ccosh(arg)) + opi.scalar_mult(i, csinh(arg)) * opj.scalar_mult(j, -np.sign(a) * csinh(arg))
    out = AFDMCSpinIsospinOperator(nt.n_particles)
    # return out
    out.op_stack[i] = ccosh(arg) * ident + csinh(arg) * opi
    out.op_stack[j] = ccosh(arg) * ident - np.sign(a) * csinh(arg) * opj
    # out.op_stack[i] *= csqrt(norm)
    # out.op_stack[j] *= csqrt(norm)
    out = out.spread_scalar_mult(norm)
    return out





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
        nt.plot_samples(b_list, filename=f'hsprop_ob{nt.run_tag}.pdf', title='HS (OBB)')

    b = np.mean(b_list)
    s = np.std(b_list) / np.sqrt(n_samples)
    print(f'gauss =  {b}  +/- {s}')


def g_rbm_sample(dt, a, h, i, j, opi, opj):
    norm = np.exp(-0.5 * dt * np.abs(a))
    W = np.arctanh(np.sqrt(np.tanh(0.5 * dt * np.abs(a))))
    arg = W * (2 * h - 1)
    out = ident.scalar_mult(i, np.cosh(arg)).scalar_mult(j, np.cosh(arg)) + opi.scalar_mult(i, np.sinh(arg)) * opj.scalar_mult(j, -np.sign(a) * np.sinh(arg))
    return out.spread_scalar_mult(norm)


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
            ket_p = g_rbm_sample(nt.dt, asig[a, b], h[n], 0, 1, sig[0][a], sig[1][b]) * ket_p
            ket_m = g_rbm_sample(nt.dt, asig[a, b], 1 - h[n], 0, 1, sig[0][a], sig[1][b]) * ket_m
            n += 1
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            for c in [0, 1, 2]:
                ket_p = g_rbm_sample(nt.dt, asigtau[a, b], h[n], 0, 1, sig[0][a] * tau[0][c], sig[1][b] * tau[1][c]) * ket_p
                ket_m = g_rbm_sample(nt.dt, asigtau[a, b], 1 - h[n], 0, 1, sig[0][a] * tau[0][c], sig[1][b] * tau[1][c]) * ket_m
                n += 1
    for c in [0, 1, 2]:
        ket_p = g_rbm_sample(nt.dt, atau, h[n], 0, 1, tau[0][c], tau[1][c]) * ket_p
        ket_m = g_rbm_sample(nt.dt, atau, 1 - h[n], 0, 1, tau[0][c], tau[1][c]) * ket_m
        n += 1

    ket_p = g_coul_onebody(nt.dt, vcoul) * g_rbm_sample(nt.dt, 0.25 * vcoul, h[n], 0, 1, tau[0][2], tau[1][2]) * ket_p
    ket_m = g_coul_onebody(nt.dt, vcoul) * g_rbm_sample(nt.dt, 0.25 * vcoul, 1 - h[n], 0, 1, tau[0][2], tau[1][2]) * ket_m
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
        nt.plot_samples(b_list, filename=f'rbmprop_ob{nt.run_tag}.pdf', title='RBM (OBB)')

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
        bracket_t0 = bra * ket
        print(f'<G(t=0)> = {bracket_t0}')
        # Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()

    print('DONE')
