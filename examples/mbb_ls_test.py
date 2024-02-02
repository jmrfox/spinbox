import nuctest as nt
from quap import *

ident = ManyBodyBasisSpinIsospinOperator(nt.n_particles)
# list constructors make generating operators more streamlined
sig = [[ManyBodyBasisSpinIsospinOperator(nt.n_particles).sigma(i,a) for a in [0, 1, 2]] for i in range(nt.n_particles)]
tau = [[ManyBodyBasisSpinIsospinOperator(nt.n_particles).tau(i,a) for a in [0, 1, 2]] for i in range(nt.n_particles)]
# sig[particle][xyz]

def g_ls_linear(gls_ai, i, a):
    # linear approx to LS
    out = ident - 1.j * gls_ai * sig[i][a] 
    return out

def g_ls_onebody(gls_ai, i, a):
    # one-body part of the LS propagator factorization
    out = - 1.j * gls_ai * sig[i][a]
    return out.exponentiate()

def g_ls_twobody(gls_ai, gls_bj, i, j, a, b):
    # two-body part of the LS propagator factorization
    out = 0.5 * gls_ai * gls_bj * sig[i][a] * sig[j][b]
    return out.exponentiate()

def ratio_linear_vs_factored(pot_scale):
    print('MBB LS TEST')
    # this looks like it works
    bra, ket = nt.make_test_states(manybody=True)
    pots = nt.make_all_potentials(scale = pot_scale)
    gls = np.sum(pots['bls'], axis = 2) 

    ket_lin = ket.copy()
    for i in range(nt.n_particles):
        for a in range(3):
            ket_lin = g_ls_linear(gls[a, i], i, a) * ket_lin
                    
    ket_fac = ket.copy()
    for i in range(nt.n_particles):
            for a in range(3):
                ket_fac = g_ls_onebody(gls[a, i], i, a) * ket_fac
    for i in range(nt.n_particles):
        for j in range(nt.n_particles):
            for a in range(3):
                for b in range(3):
                    ket_fac = g_ls_twobody(gls[a, i], gls[b, j], i, j, a, b) * ket_fac
    

    b_lin = bra * ket_lin
    b_fac = bra * ket_fac
    ratio = np.linalg.norm(b_lin) / np.linalg.norm(b_fac)
    # print('linear approx, <G> = ', b_lin )
    # print('factorization, <G> = ', b_fac)
    # print('square ratio = ', ratio)
    return ratio


def ratio_linear_vs_onebody(pot_scale):
    print('MBB LS TEST')
    # this looks like it works
    bra, ket = nt.make_test_states(manybody=True)
    pots = nt.make_all_potentials(scale = pot_scale)
    gls = np.sum(pots['bls'], axis = 2) 

    ket_lin = ket.copy()
    for i in range(nt.n_particles):
        for a in range(3):
            ket_lin = g_ls_linear(gls[a, i], i, a) * ket_lin
                    
    ket_fac = ket.copy()
    for i in range(nt.n_particles):
            for a in range(3):
                ket_fac = g_ls_onebody(gls[a, i], i, a) * ket_fac
    # for i in range(nt.n_particles):
    #     for j in range(nt.n_particles):
    #         for a in range(3):
    #             for b in range(3):
    #                 ket_fac = g_ls_twobody(gls[a, i], gls[b, j], i, j, a, b) * ket_fac
    

    b_lin = bra * ket_lin
    b_fac = bra * ket_fac
    ratio = np.linalg.norm(b_lin) / np.linalg.norm(b_fac)
    # print('linear approx, <G> = ', b_lin )
    # print('factorization, <G> = ', b_fac)
    # print('square ratio = ', ratio)
    return ratio

if __name__=="__main__":
    # ratios = []
    # gls = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
    # for pot_scale in gls:
    #     ratios.append(abs(1-ratio_linear_vs_factored(pot_scale)))

    # print(ratios, gls)
    # plt.loglog(gls, ratios, ls='-', c='k', marker='o')
    # plt.xlabel(r"$B_{LS}$")
    # plt.ylabel("error = |1 - ratio^2|")
    # plt.title('Error in LS prop: linear vs exact (factored)')
    # plt.savefig('error_linear_vs_factored.pdf')


    ratios_full = []
    ratios_ob = []
    gls = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
    for pot_scale in gls:
        ratios_full.append(abs(1-ratio_linear_vs_factored(pot_scale)))
        ratios_ob.append(abs(1-ratio_linear_vs_onebody(pot_scale)))

    plt.loglog(gls, ratios_full, ls='-', c='k', marker='o', label='exact')
    plt.loglog(gls, ratios_ob, ls='-', c='r', marker='o', label='one-body')
    plt.xlabel(r"$B_{LS}$")
    plt.ylabel("error = |1 - ratio^2|")
    plt.title(f'Error in LS prop, A = {nt.n_particles}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'errors_linear_vs_ob_A{nt.n_particles}.pdf')