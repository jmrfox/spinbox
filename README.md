# spinbox
## tools for many-fermion spin states in the context of Monte Carlo simulation

At the front lines of research on the nuclear many-body problem are a large number of supercomputer-scale simulation codes. 
These codes produce valuable results but can be hard to understand, especially for those without intimate knowledge of the relevant theoretical methods. 
Thus, tools that fill pedagogical roles are extremely valuable. 
`spinbox` makes it easy for one to replicate and analyze the computational processes relevant to a Quantum Monte Carlo (QMC) simulation that may be difficult to understand/debug/analyze due to the scale of the corresponding simulation software.

`spinbox` uses other state-of-the-art Python modules for numerical calculations. 
While a number of Python libraries exist that are suited to general quantum many-body calculations, the motivation of `spinbox` is quite particular.  
In Diffusion Monte Carlo methods (DMC, GFMC, AFDMC), the central calculation is the imaginary-time propagation of individual samples of the many-body wavefunction. 
Although quantum wavefunctions generally must be described by a probability distribution over a basis, DMC imbues particles (within one sample) with classical spatial coordinates. 
This method is unusual, so other Python packages are typically not set up to do this easily.

Furthermore, the software has built-in options for nuclear systems assuming spin-isospin symmetry. Isospin can certainly be set up with other quantum mechanics libraries, but it is a nontrivial process to do so. 
In `spinbox`, isospin symmetry is included by default and may be optionally turned off.

### Features:
- numerical representation of samples of the many-body wavefunctions, including  tensor-product states (used in AFDMC)
- numerical representation of many-body operators, including tensor-product operators: general, spin, imaginary-time propagation, etc.
- the correct associated arithmetic and algebra, implemented as class methods
- classes for representing realistic nuclear two- and three-body Hamiltonians (e.g. Argonne V18, Illinois NNN)
- large-scale parallel integration over random variables, crucial for the AFDMC method 

This package is open source so that anyone may use it and contribute to it. My hope is this is useful to other researchers doing QMC calculations.


