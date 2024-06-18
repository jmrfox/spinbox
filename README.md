# spinbox
The core components of `spinbox` are many-body spinor states and operators, including imaginary-time propagation and averaging used in quantum Monte Carlo simulations for atomic nuclei. 
A significant focus is placed on the choice of basis: a calculation can be done either in the full basis of orthonormal product states, or in a restricted "no entanglement" basis. The former is typical of Green's function Monte Carlo (GFMC), and the latter is used for nuclear auxiliary field diffusion Monte Carlo (AFDMC).
The relations between these two bases can be subtle, and thus, this package could be useful as a pedagogical resource for those new to these problems. 

The motivation for this package is my need to check auxiliary field diffusion Monte Carlo simulations of atomic nuclei. Or rather, check certain aspects of that simulation; at present, `spinbox` is not for performing a full Monte Carlo calculation, but rather testing important parts of the larger calculation on their own.
This includes inner products, matrix elements, operator composition, imaginary-time propagation, auxiliary fields, and averaging.


