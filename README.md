# spinbox : tools for many-fermion spin states and Monte Carlo simulations of atomic nuclei

On the front lines of research in the nuclear many-body problem stand a number of supercomputer-scale simulation codes. 
These codes produce invaluable results but can often be hard to understand, especially for those without intimate knowledge of the relevant theoretical methods. 
Hence, robust tools that fill pedagogical roles are extremely valuable. 
`spinbox` makes it easy for one to replicate and analyze the computational processes relevant to a Quantum Monte Carlo (QMC) simulation that may be difficult to understand, debug, or analyze due to the scale of the corresponding simulation software.

### Features:
- numerical representation of samples of the many-body wavefunctions, including  tensor-product states (used in AFDMC)
- numerical representation of many-body operators, including tensor-product operators: general matrix representations, spin, imaginary-time propagation, etc.
- arithmetical and algebraic operations implemented as class methods
- classes for representing realistic nuclear two- and three-body Hamiltonians (e.g. Argonne V18, Illinois NNN)
- large-scale parallel integration over random variables, crucial for the AFDMC method
- Hubbard-Stratonovich (HS) propagators, central to AFDMC
- restricted Boltzmann machine (RBM) propagators, an alternative to HS and the subject of <a href="https://arxiv.org/abs/2407.14632">this paper</a>.
  
**Why not use one of the many other Python packages for many-fermion calculations?**

While a number of Python packages exist that are suited to quantum many-body calculations, the motivation of `spinbox` is relatively particular. In Diffusion Monte Carlo methods (DMC, GFMC, AFDMC), rather than work with the wavefunction itself, one uses a set of points which together make a possible sampling of the many-body wavefunction's probability density. Furthermore, the basis used may be highly constrained and require special treatment, such is the case in AFDMC. 
These traits are somewhat unusual, and other Python packages are typically not set up to do this easily.
`spinbox` was designed specifically for these methods.
Furthermore, the software has built-in options for nuclear systems assuming spin-isospin symmetry. Isospin is an additional "spin-like" variable particular to nuclear and sub-nuclear physics. Isospin can certainly be set up with other quantum mechanics libraries, but it is usually a nontrivial process to do so. 
In `spinbox`, isospin symmetry is included by default and may be optionally turned off.


This package is open source so that anyone may use it and contribute to it.

## Documentation

Documentation is made with `sphinx`. If you have it, you should be able to go into the `docs` directory and run `make html`, for example. The resulting docs would appear in `docs/build/html`.