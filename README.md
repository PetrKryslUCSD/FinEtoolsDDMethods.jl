[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)

# FinEtoolsDDMethods.jl

Domain decomposition methods: solvers and algorithms used with FinEtools applications. 

## News

- 08/29/2024: Cleanup of the interface. Sequential, MPI, threaded execution supported.

## Capabilities and limitations

Only linear static problems are addressed at the moment. `FinEtools` discretization only.

- Schur-based decomposition for conjugate gradient iteration. 
- Overlapping Schwarz two-level preconditioning of conjugate gradient iteration with coherent node clusters.

Both of the above can now be expressed as MPI-parallel algorithms.

## Examples

Please refer to the `README` in the `examples` folder for instructions 
on how to get the examples to run.

## References

Paper [submitted](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4902156) for 
the coherent node cluster conjugate gradient preconditioner with applications 
to almost incompressible elasticity and general shells.

Paper in the [accepted form](https://www.sciencedirect.com/science/article/pii/S0045782524008120)
is compatible with version 0.4.2.
