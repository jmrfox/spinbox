from quap import *
import nuclear as nt
from quap.extras import chistogram
from itertools import starmap

#  this script does the same as mbbprop_averaging except for several values of B_LS and plots

ident = GFMCSpinIsospinOperator(nt.n_particles)
# list constructors make generating operators more streamlined
sig = [[GFMCSpinIsospinOperator(nt.n_particles).sigma(i,a) for a in [0, 1, 2]] for i in range(nt.n_particles)]
tau = [[GFMCSpinIsospinOperator(nt.n_particles).tau(i,a) for a in [0, 1, 2]] for i in range(nt.n_particles)]
# access like sig[particle][xyz]


def g_pade_sig(dt, asig, i, j):
    out = GFMCSpinIsospinOperator(nt.n_particles).zeros()
    for a in range(3):
        for b in range(3):
            out += asig[a, i, b, j] * sig[i][a] * sig[j][b]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_sigtau(dt, asigtau, i, j):
    out = GFMCSpinIsospinOperator(nt.n_particles).zeros()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                out += asigtau[a, i, b, j] * sig[i][a] * sig[j][b] * tau[i][c] * tau[j][c]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_tau(dt, atau, i, j):
    out = GFMCSpinIsospinOperator(nt.n_particles).zeros()
    for c in range(3):
        out += atau[i, j] * tau[i][c] * tau[j][c]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_coul(dt, v, i, j):
    out = ident + tau[i][2] + tau[j][2] + tau[i][2] * tau[j][2]
    out = -0.125 * v[i, j] * dt * out
    return out.exponentiate()


def g_coulomb_onebody(dt, v, i):
    """just the one-body part of the expanded coulomb propagator
    for use along with auxiliary field propagators"""
    out = - 0.125 * v * dt * tau[i][2]
    return out.exponentiate()


def g_ls_linear(gls, i):
    # linear approx to LS
    out = GFMCSpinIsospinOperator(nt.n_particles)
    for a in range(3):
        out = (ident - 1.j * gls[a, i] * sig[i][a]) * out 
    return out

def g_ls_onebody(gls_ai, i, a):
    # one-body part of the LS propagator factorization
    out = - 1.j * gls_ai * sig[i][a]
    return out.exponentiate()

def g_ls_twobody(gls_ai, gls_bj, i, j, a, b):
    # two-body part of the LS propagator factorization
    out = 0.5 * gls_ai * gls_bj * sig[i][a] * sig[j][b]
    return out.exponentiate()

def make_g_exact(n_particles, dt, pot, controls):
    # compute exact bracket
    g_exact = ident.copy()
    pairs_ij = interaction_indices(n_particles)
    for i,j in pairs_ij:
        if controls['sigma']:
            g_exact = g_pade_sig(dt, pot.sigma, i, j) * g_exact
        if controls['sigmatau']:
            g_exact = g_pade_sigtau(dt, pot.sigmatau, i, j) * g_exact 
        if controls['tau']:
            g_exact = g_pade_tau(dt, pot.tau, i, j) * g_exact
        if controls['coulomb']:
            g_exact = g_pade_coul(dt, pot.coulomb, i, j) * g_exact
    #  LS
    if controls['spinorbit']:
        for i in range(nt.n_particles):
            g_exact = g_ls_linear(pot.spinorbit, i) * g_exact
        # for i in range(nt.n_particles):
        #     for a in range(3):
        #         g_exact = g_ls_onebody(gls[a, i], i, a) * g_exact
        # for i in range(nt.n_particles):
        #     for j in range(nt.n_particles):
        #         for a in range(3):
        #             for b in range(3):
        #                 g_exact = g_ls_twobody(gls[a, i], gls[b, j], i, j, a, b) * g_exact
    return g_exact


#  GAUSSIAN

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
    gls = pot_dict['gls']

    n = 0
    idx = [[0, 1, 2] for _ in range(9)]
    if rng_mix is not None: # no evidence that mixing helps.
        idx = rng_mix.choice(2, size=(9,3))
    for i,j in nt.pairs_ij:
        # SIGMA
        for a in idx[0]:
            for b in idx[1]:
                ket_p = g_gauss_sample(nt.dt, asig[a, i, b, j], x[n], sig[i][a], sig[j][b]) * ket_p
                ket_m = g_gauss_sample(nt.dt, asig[a, i, b, j], -x[n], sig[i][a], sig[j][b]) * ket_m
                n += 1
        # SIGMA TAU
        for a in idx[2]:
            for b in idx[3]:
                for c in idx[4]:
                    ket_p = g_gauss_sample(nt.dt, asigtau[a, i, b, j], x[n], sig[i][a] * tau[i][c], sig[j][b] * tau[j][c]) * ket_p
                    ket_m = g_gauss_sample(nt.dt, asigtau[a, i, b, j], -x[n], sig[i][a] * tau[i][c], sig[j][b] * tau[j][c]) * ket_m
                    n += 1
        # TAU
        for c in idx[5]:
            ket_p = g_gauss_sample(nt.dt, atau[i, j], x[n], tau[i][c], tau[j][c]) * ket_p
            ket_m = g_gauss_sample(nt.dt, atau[i, j], -x[n], tau[i][c], tau[j][c]) * ket_m
            n += 1

        # COULOMB
        ket_p = g_coulomb_onebody(nt.dt, vcoul[i, j], i) * g_coulomb_onebody(nt.dt, vcoul[i, j], j) * ket_p
        ket_m = g_coulomb_onebody(nt.dt, vcoul[i, j], i) * g_coulomb_onebody(nt.dt, vcoul[i, j], j) * ket_m
        ket_p = g_gauss_sample(nt.dt, 0.25 * vcoul[i, j], x[n], tau[i][2], tau[j][2]) * ket_p
        ket_m = g_gauss_sample(nt.dt, 0.25 * vcoul[i, j], -x[n], tau[i][2], tau[j][2]) * ket_m

    # LS
    for i in range(nt.n_particles):
        for a in range(3):
            ket_p = g_ls_onebody(gls[a, i], i, a) * ket_p
            ket_m = g_ls_onebody(gls[a, i], i, a) * ket_m
    for i,j in nt.pairs_ij:
        for a in range(3):
            for b in range(3):
                asigls = - gls[a, i]* gls[b, j]
                ket_p = g_gauss_sample(1, asigls, x[n], sig[i][a], sig[j][b]) * ket_p
                ket_m = g_gauss_sample(1, asigls, -x[n], sig[i][a], sig[j][b]) * ket_m
    trace_factor = cexp( 0.5 * np.sum(gls**2))
    ket_p = trace_factor * ket_p
    ket_m = trace_factor * ket_m
    
    return 0.5 * scalar(bra * ket_p + bra * ket_m)




def gauss_brackets_parallel(bra, ket, pot_dict, n_samples=100):
    print('HS brackets')
    n_aux = (nt.n_particles**2 + nt.n_particles) * (9 + 27 + 3 + 1 + 3 + 9)
    rng = default_rng(seed=nt.global_seed)
    x_set = rng.standard_normal((n_samples, n_aux))  # different x for each x,y,z

    print(f'# PROCESSES = {nt.n_procs}')
    do_parallel = True
    if do_parallel:
        with Pool(processes=nt.n_procs) as pool:
            b_array = pool.starmap_async(gauss_task, tqdm([(x, bra, ket, pot_dict) for x in x_set], leave=True)).get()
    else:
        input("SERIAL...")
        # b_array = []
        # for args in tqdm([(x, bra, ket, pot_dict) for x in x_set], leave=True):
        #     b_array.append(gauss_task(*args))
        b_array = list(starmap(gauss_task, tqdm([(x, bra, ket, pot_dict) for x in x_set])))

    b_array = np.array(b_array)
    return b_array




#################


def main():
    n_particles=2
    dt = 0.001
    n_samples = 10000

    ket = AFDMCSpinIsospinState(n_particles,'ket', read_from_file("./data/h2/fort.770",complex=True, shape=(2,4,1))).to_manybody_basis()
    bra = ket.copy().dagger()

    pot = ArgonnePotential(n_particles)
    pot.read_sigma("./data/h2/fort.7701")
    pot.read_sigmatau("./data/h2/fort.7702")
    pot.read_tau("./data/h2/fort.7703")
    pot.read_coulomb("./data/h2/fort.7704")
    pot.read_spinorbit("./data/h2/fort.7705")

    # prop = GFMCPropagatorHS(n_particles, dt, include_prefactor=True)
    prop = GFMCPropagatorRBM(n_particles, dt, include_prefactor=True)
    controls = {'sigma': True, 'sigmatau': False, 'tau': False, 'coulomb': False, 'spinorbit': False}
    
    integ = Integrator(pot, prop)
    integ.controls = controls 
    integ.setup(n_samples=n_samples, balanced=True)
    
    b_array = integ.run(bra, ket, parallel=True)
    b_m = np.mean(b_array)
    b_s = np.std(b_array)/np.sqrt(n_samples)
    print(f'bracket = {b_m} +/- {b_s}')
    # chistogram(b_array, filename='hs_test.pdf', title='HS test')

    g_exact = make_g_exact(n_particles, dt, pot, controls)
    b_exact = bra * g_exact * ket
    print('exact = ',b_exact)

    print("ratio = ", np.abs(b_m)/np.abs(b_exact) )
    print("abs error = ", abs(1-np.abs(b_m)/np.abs(b_exact)) )

if __name__ == "__main__":
    main()
    