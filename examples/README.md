# Examples

This is the `examples` folder.
There are two sub folders: 
- `schur`: Schur-complement-based solver using domain decomposition and conjugate gradients
on the complement matrix. Some of the examples are parallelized with MPI.
- `conc`: coherent node cluster (CoNC) model reduction is used as a global solver in a preconditioned conjugate gradient based on the decomposition at two levels:
local (classical additive Schwarz based on overlapping subdomains), and global (reduced model based on coherent clusters). The examples do not run in parallel yet.

## How to run a Schur-complement CG parallel example

Please do:

- Clone the package, change into the `examples` folder, activate and instantiate this environment.
- Request the binary `OpenMPI_jll`
```
using MPIPreferences
MPIPreferences.use_jll_binary("OpenMPI_jll")
```
Or, use the trampoline
```
MPIPreferences.use_jll_binary( "MPItrampoline_jll")
```
- Install the helper script `mpiexecjl`
```
using MPI
MPI.install_mpiexecjl()
```
Note the folder where the executable is installed.
- Run the example. `mpiexecjl` likely needs to be specified using the path of the folder in which it was installed.
```
 mpiexecjl -n 4 --project=. julia schur/heat/Poisson2D_cg_mpi_driver.jl
```

On Windows 11, the following would work:
```
mpiexec -n 3 julia --project=. .\conc\heat\Poisson2D_mpi_driver.jl
```


## How to run a CoNC-preconditioned CG (sequential) example

Two classes of problems solved here:
- Three dimensional elasticity (composite of matrix with embedded fibers).
- General three dimensional shells.

There are shell scripts to run the studies reported in the paper.

The three examples for the elasticity problem are:
- Moderately compressible matrix. Driver `cc.jl` (shell script `cc.sh`).
- Strongly compressible matrix. Driver `css.jl` (shell script `css.sh`).
- Nearly incompressible matrix. Driver `cni.jl` (shell script `cni.sh`).

The three examples for the shell problems are:
- Single-sheet hyperboloid with cosine pressure loading and free edge. Driver `hyp.jl` (shell script `hyp.sh`).
- Z-section cantilever under torsional loading. Driver `z.jl` (shell script `z.sh`).
- Barrel with stiffeners.  Driver `barrel.jl` (shell script `barrel.sh`).

For instance, when in the folder `FinEtoolsDDMethods.jl/examples`, execute
```
$ bash conc/shells/hyp.sh
```
Warning: Running the examples may require a beefy machine,
and it may take a long time to finish the simulations: some of the scripts run through many
combinations of the parameters.

To run one particular example with just selected input parameters, first change into the `examples` folder.
```
PS C:\temp\FinEtoolsDDMethods.jl> cd .\examples\
```
Then fire up Julia 
```
PS C:\temp\FinEtoolsDDMethods.jl\examples> julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.10.4 (2024-06-04)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |
```
and run
```
julia> using Pkg; Pkg.activate("."); Pkg.instantiate();
  Activating project at `C:\temp\FinEtoolsDDMethods.jl\examples`

julia> include("conc/shells/hyp.jl")
Current folder: C:\temp\FinEtoolsDDMethods.jl\examples
  Activating project at `C:\temp\FinEtoolsDDMethods.jl\examples`
Refinement factor: 2
Number of elements per partition: 200
Number of 1D basis functions: 5
Number of fine grid partitions: 2
Overlap: 1
Number of elements: 8192
Number of free dofs = 24767
Number coarse grid partitions: 41
Size of the reduced problem: (3690, 3690)
Mean fine partition size: 1.29775e+04
Number of iterations:  11
true
```

Alternatively, a top level script may be run as
```
$ julia conc/shells/barrel.jl --nfpartitions 32
```
Run
```
$ julia conc/shells/barrel.jl --help
```
to see the available options.

## How to run a CoNC-preconditioned CG (MPI-parallel) example

At the moment only the heat conduction and shell analysis examples have been cast in this form. Try
```
mpiexec -n 5 julia --project=. .\conc\heat\Poisson2D_mpi_driver.jl
```
or
```
mpiexec -n 5 julia --project=. .\conc\shells\barrel_mpi_driver.jl
```
