from quap import *

# test spin+isospin states and operators

def test_states():
    print('TESTING SPIN-ISOSPIN STATES')
    print('INITIALIZING ONE-BODY')
    coeffs_uu = np.concatenate([spinor4('up', 'ket'), spinor4('up', 'ket')], axis=0)
    coeffs_ud = np.concatenate([spinor4('up', 'ket'), spinor4('down', 'ket')], axis=0)
    sp_uu = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_uu)
    sp_ud = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_ud)
    print("|uu> = \n", sp_uu)
    print("|ud> = \n", sp_ud)
    print('INNER PRODUCTS')
    print("<uu|uu> = \n", sp_uu.transpose() * sp_uu)
    print("<ud|ud> = \n", sp_ud.transpose() * sp_ud)
    print("<uu|ud> = \n", sp_uu.transpose() * sp_ud)
    print("|uu> = \n", sp_uu)
    print("|ud> = \n", sp_ud)
    print('OUTER PRODUCTS')
    print("|uu><uu| = \n", sp_uu * sp_uu.transpose())
    print("|ud><ud| = \n", sp_ud * sp_ud.transpose())
    print("|uu><ud| = \n", sp_uu * sp_ud.transpose())
    print('TO MANYBODY')
    sp_uu = sp_uu.to_many_body_state()
    sp_ud = sp_ud.to_many_body_state()
    print("|uu> = \n", sp_uu)
    print("|ud> = \n", sp_ud)
    print('INNER PRODUCTS')
    print("<uu|uu> = \n", sp_uu.transpose() * sp_uu)
    print("<ud|ud> = \n", sp_ud.transpose() * sp_ud)
    print("<uu|ud> = \n", sp_uu.transpose() * sp_ud)
    print('OUTER PRODUCTS')
    print("|uu><uu| = \n", sp_uu * sp_uu.transpose())
    print("|ud><ud| = \n", sp_ud * sp_ud.transpose())
    print("|uu><ud| = \n", sp_uu * sp_ud.transpose())

    print('RANDOM TENSOR PRODUCTS')
    coeffs_0 = np.concatenate([spinor4('random', 'ket'), spinor4('random', 'ket')], axis=0)
    coeffs_1 = np.concatenate([spinor4('random', 'ket'), spinor4('random', 'ket')], axis=0)
    sp_0 = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_0)
    sp_1 = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_1)
    print("|0> = \n", sp_0)
    print("|1> = \n", sp_1)
    print('INNER PRODUCTS')
    print("<0|0> = \n", sp_0.transpose() * sp_0)
    print("<1|1> = \n", sp_1.transpose() * sp_1)
    print("<0|1> = \n", sp_0.transpose() * sp_1)
    print('OUTER PRODUCTS')
    print("|0><0| = \n", sp_0 * sp_0.transpose())
    print("|1><1| = \n", sp_1 * sp_1.transpose())
    print("|0><1| = \n", sp_0 * sp_1.transpose())
    print('TO MANYBODY')
    sp_0 = sp_0.to_many_body_state()
    sp_1 = sp_1.to_many_body_state()
    print("|0> = \n", sp_0)
    print("|1> = \n", sp_1)
    print('INNER PRODUCTS')
    print("<0|0> = \n", sp_0.transpose() * sp_0)
    print("<1|1> = \n", sp_1.transpose() * sp_1)
    print("<0|1> = \n", sp_0.transpose() * sp_1)
    print('OUTER PRODUCTS')
    print("|0><0| = \n", sp_0 * sp_0.transpose())
    print("|1><1| = \n", sp_1 * sp_1.transpose())
    print("|0><1| = \n", sp_0 * sp_1.transpose())
    print('DONE TESTING STATES')


def test_operators():
    print('TESTING OPERATORS')
    print('TENSOR PRODUCT STATES')
    coeffs_uu = np.concatenate([spinor4('up', 'ket'), spinor4('up', 'ket')], axis=0)
    coeffs_ud = np.concatenate([spinor4('up', 'ket'), spinor4('down', 'ket')], axis=0)
    sp_uu = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_uu)
    sp_ud = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_ud)
    print("|uu> = \n", sp_uu)
    print("|ud> = \n", sp_ud)
    print('TENSOR PRODUCT OPERATORS')
    sigx0 = OneBodyBasisSpinIsospinOperator(2).sigma(0, 'x')
    sigy0 = OneBodyBasisSpinIsospinOperator(2).sigma(0, 'y')
    sigz0 = OneBodyBasisSpinIsospinOperator(2).sigma(0, 'z')
    print("sigx0 = \n", sigx0)
    print("sigy0 = \n", sigy0)
    print("sigz0 = \n", sigz0)
    print("sigx0 |uu> = \n", sigx0 * sp_uu)
    print("sigy0 |uu> = \n", sigy0 * sp_uu)
    print("sigz0 |uu> = \n", sigz0 * sp_uu)
    print("sigx0 |ud> = \n", sigx0 * sp_ud)
    print("sigy0 |ud> = \n", sigy0 * sp_ud)
    print("sigz0 |ud> = \n", sigz0 * sp_ud)
    print('EXCHANGE P(i,j)')
    P01 = OneBodyBasisSpinIsospinOperator(2).exchange(0, 1)
    print('P(0,1) = \n', P01)
    print("|ud> = \n", sp_ud)
    print("P(0,1) |ud> = \n", P01 * sp_ud)
    print("SCALAR MULTIPLICATION")
    five0 = OneBodyBasisSpinIsospinOperator(2).scalar_mult(0, 5)
    three1 = OneBodyBasisSpinIsospinOperator(2).scalar_mult(1, 3)
    print("5(0) = \n", five0)
    print("3(1) = \n", three1)
    print("5(0) |uu> = \n", five0 * sp_uu)
    print("3(1) |ud> = \n", three1 * sp_ud)

    print('MANYBODY STATES')
    sp_uu = sp_uu.to_many_body_state()
    sp_ud = sp_ud.to_many_body_state()
    print("|uu> = \n", sp_uu)
    print("|ud> = \n", sp_ud)
    print('MANYBODY OPERATORS')
    sigx0 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'x')
    sigy0 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'y')
    sigz0 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'z')
    print("sigx0 = \n", sigx0)
    print("sigy0 = \n", sigy0)
    print("sigz0 = \n", sigz0)
    print("sigx0 |uu> = \n", sigx0 * sp_uu)
    print("sigy0 |uu> = \n", sigy0 * sp_uu)
    print("sigz0 |uu> = \n", sigz0 * sp_uu)
    print("sigx0 |ud> = \n", sigx0 * sp_ud)
    print("sigy0 |ud> = \n", sigy0 * sp_ud)
    print("sigz0 |ud> = \n", sigz0 * sp_ud)
    print('EXCHANGE P(i,j)')
    P01 = ManyBodyBasisSpinIsospinOperator(2).exchange(0, 1)
    print('P(0,1) = \n', P01)
    print("|ud> = \n", sp_ud)
    print("P(0,1) |ud> = \n", P01 * sp_ud)
    print("SCALAR MULTIPLICATION")
    five0 = ManyBodyBasisSpinIsospinOperator(2).scalar_mult(0, 5)
    three1 = ManyBodyBasisSpinIsospinOperator(2).scalar_mult(1, 3)
    print("5(0) = \n", five0)
    print("3(1) = \n", three1)
    print("5(0) |uu> = \n", five0 * sp_uu)
    print("3(1) |ud> = \n", three1 * sp_ud)

    print('TENSOR PRODUCT TEST: sigma(i) dot sigma(j) = 2P(i,j) - 1')
    bra, ket = random_spin_bra_ket(2)
    sigx01 = OneBodyBasisSpinIsospinOperator(2).sigma(0, 'x').sigma(1, 'x')
    sigy01 = OneBodyBasisSpinIsospinOperator(2).sigma(0, 'y').sigma(1, 'y')
    sigz01 = OneBodyBasisSpinIsospinOperator(2).sigma(0, 'z').sigma(1, 'z')
    P01 = OneBodyBasisSpinOperator(2).exchange(0, 1)
    lhs = bra * (sigx01 * ket) + bra * (sigy01 * ket) + bra * (sigz01 * ket)
    rhs = 2 * (bra * (P01 * ket)) - bra * ket
    print("sigma(i) dot sigma(j) = \n", lhs)
    print("2P(i,j) - 1 \n", rhs)

    print('MANY BODY TEST: sigma(i) dot sigma(j) = 2P(i,j) - 1')
    bra, ket = random_spin_bra_ket(2)
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()
    sigx01 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'x').sigma(1, 'x')
    sigy01 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'y').sigma(1, 'y')
    sigz01 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'z').sigma(1, 'z')
    P01 = ManyBodyBasisSpinIsospinOperator(2).exchange(0, 1)
    lhs = prod([bra, sigx01, ket]) + prod([bra, sigy01, ket]) + prod([bra, sigz01, ket])
    rhs = 2 * (bra * (P01 * ket)) - bra * ket
    print("sigma(i) dot sigma(j) = \n", lhs)
    print("2P(i,j) - 1 \n", rhs)

    print('DONE TESTING OPERATORS')



def test_propagator_easy():
    print('TESTING PROPAGATOR (EASY)')
    # bra, ket = random_bra_ket()
    coeffs_bra = np.concatenate([spinor4('max', 'bra'), spinor4('max', 'bra')], axis=1)
    coeffs_ket = np.concatenate([spinor4('max', 'ket'), spinor4('max', 'ket')], axis=0)
    bra = OneBodyBasisSpinIsospinState(2, 'bra', coeffs_bra).to_many_body_state()
    ket = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_ket).to_many_body_state()
    dt = 0.01
    A = 1.0
    Sx = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'x').sigma(1, 'x')
    Gx = (-dt * A * Sx).exponentiate()
    print("Gx = \n", Gx)
    print("<s| = \n", bra)
    print("|s> = \n", ket)
    print("<s|Gx|s> = \n", prod([bra, Gx, ket]))

# def test_product_states():
#     import matplotlib.pyplot as plt
#     state_x = CoordinateState(2,'ket','1Dradial',{'n':5})
#     domain = np.linspace(0,3)
#     y = state_x.psi(domain)
#     plt.plot(domain,y)
#     plt.show()



if __name__ == "__main__":
    print('TEST 1 start')
    test_states()
    test_operators()
    test_propagator_easy()
    print('TEST 1 complete')
