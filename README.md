# quap
The core components of `quap` are many-body fermion states and operators allowing for comparing calculations in the full many-body basis to those in an uncorrelated "one-body" basis.

The motivation for this package is my need to check auxiliary field diffusion Monte Carlo simulations of atomic nuclei. Or rather, check certain aspects of that simulation; at present, `quap` is not for performing a full Monte Carlo calculation, but rather testing important parts of the larger calculation on their own. 
Thus, while many parts of the package are general, some aspects correspond to nuclear AFDMC methods. The one-body basis states, for instance, are the basis states used in AFDMC, and spin-isospin states are used to simulate nuclear states. For a single-species fermionic calculation, like electronic systems, one can use jus the spin states with no isospin.

`quap` is short for "quantum playground", indicative of my foolish ambition to accommodate general quantum mechanical calculations. However, it's also a loose reference to Qwop, a stupid and utterly <a href="https://www.foddy.net/Athletics.html">legendary video game</a> by Bennett Foddy.