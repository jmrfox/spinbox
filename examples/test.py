from spinbox import *
from spinbox.extras import *
import pytest

print(pytest.__version__)

epsilon = 10**-12

def eq(x,y,verbose=False):
    if verbose:
        print("x = ", x)
        print("y = ", y)
        print("|x-y| = ", abs(x-y))
    return abs(x-y)<epsilon


###

def test_overlaps_1():
    n_particles = 3
    isospin = True
    s0 = ProductState(n_particles, isospin=isospin).randomize(100)
    s1 = ProductState(n_particles, isospin=isospin).randomize(101)
    assert eq(1.0, s0.dagger().inner(s0))
    b_prod = s0.dagger().inner(s1)
    b_hilb = s0.dagger().to_manybody_basis().inner(s1.to_manybody_basis())
    assert eq(b_prod,b_hilb)

def test_overlaps_2():
    n_particles = 3
    isospin = True
    s0 = HilbertState(n_particles, isospin=isospin).randomize(100)
    s1 = HilbertState(n_particles, isospin=isospin).randomize(101)
    assert eq(1.0, s0.dagger().inner(s0))
    
    
def test_2Pij_minus_1():
    n_particles = 3
    isospin = False
    s0 = ProductState(n_particles, isospin=isospin).randomize(100)
    s1 = ProductState(n_particles, isospin=isospin).randomize(101)
    s0_exch = s0.copy()
    temp = s0_exch.coefficients[0].copy()
    s0_exch.coefficients[0] = s0_exch.coefficients[1].copy()
    s0_exch.coefficients[1] = temp
    b_exch = 2* s1.dagger().inner(s0_exch) - s1.dagger().inner(s0)
    sig_x = ProductOperator(n_particles=n_particles, isospin=isospin).apply_sigma(0,0).apply_sigma(1,0)
    sig_y = ProductOperator(n_particles=n_particles, isospin=isospin).apply_sigma(0,1).apply_sigma(1,1)
    sig_z = ProductOperator(n_particles=n_particles, isospin=isospin).apply_sigma(0,2).apply_sigma(1,2)
    b_sig = s1.dagger().inner(sig_x.multiply_state(s0)) + s1.dagger().inner(sig_y.multiply_state(s0)) + s1.dagger().inner(sig_z.multiply_state(s0))
    assert eq(b_exch, b_sig, verbose=True)


def test_densmat():
    n_particles = 3
    isospin = True
    s0 = ProductState(n_particles, isospin=isospin).randomize(100)
    s1 = ProductState(n_particles, isospin=isospin).randomize(101)
    r0 = s0.outer(s0.dagger())
    r1 = s1.outer(s1.dagger())
    assert eq(s0.dagger().inner(r1.multiply_state(s0)) , s1.dagger().inner(r0.multiply_state(s1)) )
    

def test_amplitudes_1():
    n_particles = 3
    isospin = True
    dt = 0.01
    ###
    ket = ProductState(n_particles, isospin=isospin).randomize().to_manybody_basis()
    bra = ket.copy().dagger()
    op = HilbertOperator(n_particles=n_particles, isospin=isospin).apply_sigma(0,0).apply_tau(1,1)
    b_hilb = bra.inner(op.multiply_state(ket))
    ###
    ket = ProductState(n_particles, isospin=isospin).randomize()
    bra = ket.copy().dagger()
    op = ProductOperator(n_particles=n_particles, isospin=isospin).apply_sigma(0,0).apply_tau(1,1)
    b_prod = bra.inner(op.multiply_state(ket))
    assert eq(b_hilb, b_prod)
    

def test_hs_sigma():
    n_particles = 3
    isospin = True
    dt = 0.01
    pot = ArgonnePotential(n_particles)
    pot.sigma.generate(1.0)
    ###
    ket = ProductState(n_particles, isospin=isospin).randomize().to_manybody_basis()
    bra = ket.copy().dagger()
    prop = HilbertPropagatorHS(n_particles, dt, isospin=isospin)
    aux = np.ones(prop.n_aux_sigma).flatten()
    factors = prop.factors_sigma(pot.sigma, aux)
    ket_prop = ket.copy()
    for f in factors:
        ket_prop = f.multiply_state(ket_prop)
    b_hilb = bra.inner(ket_prop)
    ###
    ket = ProductState(n_particles, isospin=isospin).randomize()
    bra = ket.copy().dagger()
    prop = ProductPropagatorHS(n_particles, dt, isospin=isospin)
    aux = np.ones(prop.n_aux_sigma).flatten()
    factors = prop.factors_sigma(pot.sigma, aux)
    ket_prop = ket.copy()
    for f in factors:
        ket_prop = f.multiply_state(ket_prop)
    b_prod = bra.inner(ket_prop)
    assert eq(b_hilb, b_prod, verbose=True)
    
    
def test_rbm_sigma():
    n_particles = 3
    isospin = True
    dt = 0.01
    pot = ArgonnePotential(n_particles)
    pot.sigma.generate(1.0)
    ###
    ket = ProductState(n_particles, isospin=isospin).randomize().to_manybody_basis()
    bra = ket.copy().dagger()
    ket_prop = ket.copy()
    prop = HilbertPropagatorRBM(n_particles, dt, isospin=isospin)
    aux = np.ones(prop.n_aux_sigma).flatten()
    factors = prop.factors_sigma(pot.sigma, aux)
    for f in factors:
        ket_prop = f.multiply_state(ket_prop)
    b_hilb = bra.inner(ket_prop)
    ###
    ket = ProductState(n_particles, isospin=isospin).randomize()
    bra = ket.copy().dagger()
    ket_prop = ket.copy()
    prop = ProductPropagatorRBM(n_particles, dt, isospin=isospin)
    aux = np.ones(prop.n_aux_sigma).flatten()
    factors = prop.factors_sigma(pot.sigma, aux)
    for f in factors:
        ket_prop = f.multiply_state(ket_prop)
    b_prod = bra.inner(ket_prop)
    assert eq(b_hilb, b_prod, verbose=True)
    
    