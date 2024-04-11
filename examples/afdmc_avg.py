from quap import *


n_particles = 2
dt = 0.001
n_samples = 100000
seed = 2



#  this script does the same as mbbprop_averaging except for several values of B_LS and plots

ident = GFMCSpinIsospinOperator(n_particles)
# list constructors make generating operators more streamlined
sig = [[GFMCSpinIsospinOperator(n_particles).sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
tau = [[GFMCSpinIsospinOperator(n_particles).tau(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
# access like sig[particle][xyz]


def g_pade_sig(dt, asig, i, j):
    out = GFMCSpinIsospinOperator(n_particles).zeros()
    for a in range(3):
        for b in range(3):
            out += asig[a, i, b, j] * sig[i][a] * sig[j][b]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_sigtau(dt, asigtau, i, j):
    out = GFMCSpinIsospinOperator(n_particles).zeros()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                out += asigtau[a, i, b, j] * sig[i][a] * sig[j][b] * tau[i][c] * tau[j][c]
    out = -0.5 * dt * out
    return out.exponentiate()


def g_pade_tau(dt, atau, i, j):
    out = GFMCSpinIsospinOperator(n_particles).zeros()
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
    out = GFMCSpinIsospinOperator(n_particles)
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
        for i in range(n_particles):
            g_exact = g_ls_linear(pot.spinorbit, i) * g_exact
        # for i in range(n_particles):
        #     for a in range(3):
        #         g_exact = g_ls_onebody(gls[a, i], i, a) * g_exact
        # for i in range(n_particles):
        #     for j in range(n_particles):
        #         for a in range(3):
        #             for b in range(3):
        #                 g_exact = g_ls_twobody(gls[a, i], gls[b, j], i, j, a, b) * g_exact
    return g_exact




def main():
    ket = AFDMCSpinIsospinState(n_particles,'ket', read_from_file("./data/h2/fort.770",complex=True, shape=(2,4,1)))
    bra = ket.copy().dagger()

    pot = ArgonnePotential(n_particles)
    pot.read_sigma("./data/h2/fort.7701")
    pot.read_sigmatau("./data/h2/fort.7702")
    pot.read_tau("./data/h2/fort.7703")
    pot.read_coulomb("./data/h2/fort.7704")
    pot.read_spinorbit("./data/h2/fort.7705")

    prop = AFDMCPropagatorHS(n_particles, dt, include_prefactor=True, mix=True, seed=seed)
    # prop = GFMCPropagatorRBM(n_particles, dt, include_prefactor=True, mix=True)
    
    integ = Integrator(pot, prop)
    integ.controls["sigma"] = True
    integ.controls["sigmatau"] = False
    integ.controls["tau"] = False
    integ.controls["coulomb"] = False
    integ.controls["spinorbit"] = False
    integ.controls["balanced"] = True 
    integ.setup(n_samples=n_samples, seed=seed)
    
    b_array = integ.run(bra, ket, parallel=True)
    b_m = np.mean(b_array)
    b_s = np.std(b_array)/np.sqrt(n_samples)
    print(f'bracket = {b_m} +/- {b_s}')
    # chistogram(b_array, filename='hs_test.pdf', title='HS test')

    g_exact = make_g_exact(n_particles, dt, pot, integ.controls)
    b_exact = bra * g_exact * ket
    print('exact = ',b_exact)

    print("ratio = ", np.abs(b_m)/np.abs(b_exact) )
    print("abs error = ", abs(1-np.abs(b_m)/np.abs(b_exact)) )

if __name__ == "__main__":
    main()
    