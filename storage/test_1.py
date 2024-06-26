from spinbox import *

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
    print("<uu|uu> = \n", sp_uu.dagger() * sp_uu)
    print("<ud|ud> = \n", sp_ud.dagger() * sp_ud)
    print("<uu|ud> = \n", sp_uu.dagger() * sp_ud)
    print("|uu> = \n", sp_uu)
    print("|ud> = \n", sp_ud)
    print('OUTER PRODUCTS')
    print("|uu><uu| = \n", sp_uu * sp_uu.dagger())
    print("|ud><ud| = \n", sp_ud * sp_ud.dagger())
    print("|uu><ud| = \n", sp_uu * sp_ud.dagger())
    print('TO MANYBODY')
    sp_uu = sp_uu.to_many_body_state()
    sp_ud = sp_ud.to_many_body_state()
    print("|uu> = \n", sp_uu)
    print("|ud> = \n", sp_ud)
    print('INNER PRODUCTS')
    print("<uu|uu> = \n", sp_uu.dagger() * sp_uu)
    print("<ud|ud> = \n", sp_ud.dagger() * sp_ud)
    print("<uu|ud> = \n", sp_uu.dagger() * sp_ud)
    print('OUTER PRODUCTS')
    print("|uu><uu| = \n", sp_uu * sp_uu.dagger())
    print("|ud><ud| = \n", sp_ud * sp_ud.dagger())
    print("|uu><ud| = \n", sp_uu * sp_ud.dagger())

    print('RANDOM TENSOR PRODUCTS')
    coeffs_0 = np.concatenate([spinor4('random', 'ket'), spinor4('random', 'ket')], axis=0)
    coeffs_1 = np.concatenate([spinor4('random', 'ket'), spinor4('random', 'ket')], axis=0)
    sp_0 = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_0)
    sp_1 = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_1)
    print("|0> = \n", sp_0)
    print("|1> = \n", sp_1)
    print('INNER PRODUCTS')
    print("<0|0> = \n", sp_0.dagger() * sp_0)
    print("<1|1> = \n", sp_1.dagger() * sp_1)
    print("<0|1> = \n", sp_0.dagger() * sp_1)
    print('OUTER PRODUCTS')
    print("|0><0| = \n", sp_0 * sp_0.dagger())
    print("|1><1| = \n", sp_1 * sp_1.dagger())
    print("|0><1| = \n", sp_0 * sp_1.dagger())
    print('TO MANYBODY')
    sp_0 = sp_0.to_many_body_state()
    sp_1 = sp_1.to_many_body_state()
    print("|0> = \n", sp_0)
    print("|1> = \n", sp_1)
    print('INNER PRODUCTS')
    print("<0|0> = \n", sp_0.dagger() * sp_0)
    print("<1|1> = \n", sp_1.dagger() * sp_1)
    print("<0|1> = \n", sp_0.dagger() * sp_1)
    print('OUTER PRODUCTS')
    print("|0><0| = \n", sp_0 * sp_0.dagger())
    print("|1><1| = \n", sp_1 * sp_1.dagger())
    print("|0><1| = \n", sp_0 * sp_1.dagger())
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
    # print('EXCHANGE P(i,j)')
    # P01 = OneBodyBasisSpinIsospinOperator(2).exchange(0, 1)
    # print('P(0,1) = \n', P01)
    # print("|ud> = \n", sp_ud)
    # print("P(0,1) |ud> = \n", P01 * sp_ud)
    # print("SCALAR MULTIPLICATION")
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

    print('MANY BODY TEST: sigma(i) dot sigma(j) = 2P(i,j) - 1')
    bra, ket = random_spinisospin_bra_ket(2)
    bra_mb = bra.to_many_body_state()
    ket_mb = ket.to_many_body_state()
    sigx01 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'x').sigma(1, 'x')
    sigy01 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'y').sigma(1, 'y')
    sigz01 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'z').sigma(1, 'z')
    P01 = ManyBodyBasisSpinIsospinOperator(2).exchange(0, 1)
    lhs = bra_mb * sigx01 * ket_mb + bra_mb * sigy01 * ket_mb + bra_mb * sigz01 * ket_mb
    rhs = 2 * (bra_mb * P01 * ket_mb) - bra_mb * ket_mb
    print("sigma(i) dot sigma(j) = \n", lhs)
    print("2P(i,j) - 1 \n", rhs)

    # print("ONE-BODY CHECK: 2P(i,j) - 1")
    # P01 = OneBodyBasisSpinIsospinOperator(2).exchange(0, 1)
    # print(2 * (bra * P01 * ket) - bra * ket)

    print('DONE TESTING OPERATORS')



def test_propagator_x():
    print('TESTING PROPAGATOR X')
    coeffs_bra = np.concatenate([spinor4('max', 'bra'), spinor4('max', 'bra')], axis=1)
    coeffs_ket = np.concatenate([spinor4('up', 'ket'), spinor4('down', 'ket')], axis=0)
    bra = OneBodyBasisSpinIsospinState(2, 'bra', coeffs_bra).to_many_body_state()
    ket = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_ket).to_many_body_state()
    dt = 0.01
    A = 1.0
    Sx = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'x').sigma(1, 'x')
    Gx = (-dt/2 * A * Sx).exponentiate()
    print("Gx = \n", Gx)
    print("<s| = \n", bra)
    print("|s> = \n", ket)
    print("<s|Gx|s> = \n", bra * Gx * ket)

def test_propagator_sigma():
    print('TESTING PROPAGATOR SIGMA')
    coeffs_bra = np.concatenate([spinor4('max', 'bra'), spinor4('max', 'bra')], axis=1)
    coeffs_ket = np.concatenate([spinor4('up', 'ket'), spinor4('down', 'ket')], axis=0)
    bra = OneBodyBasisSpinIsospinState(2, 'bra', coeffs_bra).to_many_body_state()
    ket = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_ket).to_many_body_state()
    dt = 0.01
    A = 1.0
    Sx = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'x').sigma(1, 'x')
    Sy = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'y').sigma(1, 'y')
    Sz = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'z').sigma(1, 'z')
    G = (-dt/2 * A * (Sx + Sy + Sz)).exponentiate()
    print("G = \n", G)
    print("<s| = \n", bra)
    print("|s> = \n", ket)
    print("<s|G|s> = \n", bra * G * ket)

if __name__ == "__main__":
    print('TEST 1 start')
    # test_states()
    test_operators()
    # test_propagator_x()
    # test_propagator_sigma()
    print('TEST 1 complete')
