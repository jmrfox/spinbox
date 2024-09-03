import numpy as np

def exp(x):
    return np.exp(x,dtype=complex)

def sqrt(x):
    return np.sqrt(x,dtype=complex)

def log(x):
    return np.log(x,dtype=complex)

def cabs(x):
    return np.abs(np.real(x)) + 1j*np.abs(np.imag(x))

##

def threebody_prop(sig, a3):
    out = exp( -a3 * sig[0] * sig[1] * sig[2])
    return out

def partition(sig, w, c, n):
    out = exp( - c - w*sig[0] - w*sig[1] - w*sig[2] ) + 1
    return n * out

def correction(sig, a2, a1):
    out = exp(a2*sig[0]*sig[1] + a2*sig[0]*sig[2] + a2*sig[1]*sig[2] +
                  a1*sig[0] + a1*sig[1] + a1*sig[2])
    return out

### repulsive: a3>0

def soln_c_repulsive(a3):
    x = sqrt(exp(8*a3) - 1)
    x = sqrt( 2*exp(4*a3)*( exp(4*a3)*x + exp(8*a3) - 1  ) - x)
    x = x + exp(6*a3) + exp(2*a3)*sqrt(exp(8*a3) - 1)
    x = x*2*exp(2*a3) - 1
    x = 0.5*log(x)
    return x


def soln_a2_repulsive(a3):
    c = soln_c_repulsive(a3)
    x = 0.125*( 2*c - log(exp(4*c) + 1) + log(2) )
    return x

def soln_a1_repulsive(a3):
    c = soln_c_repulsive(a3)
    x = 0.125*( 6*c - log(exp(4*c) + 1) + log(2) )
    return x


def soln_n_repulsive(a3):
    c = soln_c_repulsive(a3)
    top = exp( 5 * c / 4)
    bottom = 2**(3/8) * sqrt(exp(2*c) + 1) * (exp(4*c) + 1)**0.125
    return top/bottom

###  attractive a3<0

def soln_c_attractive(a3):
    x = sqrt(1 - exp(8*a3))
    x = sqrt( 2*(x + 1) - exp(8*a3) * ( x + 2) )
    x = x + 1 + sqrt(1 - exp(8*a3))
    x = 0.5 * log(2*exp(-8*a3)*x - 1)
    return x

def soln_a2_attractive(a3):
    c = soln_c_attractive(a3)
    x = 0.125*( 2*c - log(exp(4*c) + 1) + log(2) )
    return x

def soln_a1_attractive(a3):
    c = soln_c_attractive(a3)
    x = 0.125*( log(0.5*(exp(4*c) + 1)) - 6*c )
    return x

def soln_n_attractive(a3):
    c = soln_c_attractive(a3)
    top = exp( c / 4)
    bottom = 2**(3/8) * sqrt(exp(-2*c) + 1) * (exp(4*c) + 1)**0.125
    return top/bottom


    
def test_real(sig, a3):
    lhs = threebody_prop(sig,a3)
    if a3>0:   # repulsive force
        c = soln_c_repulsive(a3)
        w = c
        a2 = soln_a2_repulsive(a3)
        a1 = soln_a1_repulsive(a3)
        n = soln_n_repulsive(a3)
        rhs = partition(sig, w, c, n) * correction(sig, a2, a1)
    else:  # attractive force
        c = soln_c_attractive(a3)
        w = -c
        a2 = soln_a2_attractive(a3)
        a1 = soln_a1_attractive(a3)
        n = soln_n_attractive(a3)
        rhs = partition(sig, w, c, n) * correction(sig, a2, a1)
    
    print(f"LHS = {lhs}")
    print(f"RHS = {rhs}")

def test_imag(sig, a3):
    lhs = threebody_prop(sig,a3)
    if np.imag(a3)>0:   # repulsive force
        c = soln_c_repulsive(a3)
        w = c
        a2 = soln_a2_repulsive(a3)
        a1 = soln_a1_repulsive(a3)
        n = soln_n_repulsive(a3)
        rhs = partition(sig, w, c, n) * correction(sig, a2, a1)
    else:  # attractive force
        c = soln_c_attractive(a3)
        w = -c
        a2 = soln_a2_attractive(a3)
        a1 = soln_a1_attractive(a3)
        n = soln_n_attractive(a3)
        rhs = partition(sig, w, c, n) * correction(sig, a2, a1)
    
    print(f"LHS = {lhs}")
    print(f"RHS = {rhs}")


def main():
    sig = [-1,1,1]
    a3 = 3.14j
    test_imag(sig, a3)

if __name__=="__main__":
    main()