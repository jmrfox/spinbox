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


if __name__=="__main__":
    print('MBB LS TEST')
    # this looks like it works
    bra, ket = nt.make_test_states(manybody=True)
    pots = nt.make_all_potentials(scale = 0.1)
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
    print('linear approx, <G> = ', b_lin )
    print('factorization, <G> = ', b_fac)
    print('square ratio = ', np.linalg.norm(b_lin) / np.linalg.norm(b_fac))


    print('DONE')