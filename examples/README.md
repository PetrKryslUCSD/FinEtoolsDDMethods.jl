# Examples

This is the `examples` folder.
There are two sub folders: 
- `schur`: Schur-complement-based solver using domain decomposition and conjugate gradients
on the complement matrix. Some of the examples are parallelized with MPI.
- `conc`: coherent node cluster (CoNC) model reduction is used as a global solver in a preconditioned conjugate gradient based on the decomposition at two levels:
local (classical additive Schwarz based on overlapping subdomains), and global (reduced model based on coherent clusters). The examples do not run in parallel yet.

## Setting up the MPI environment

For parallel execution, the Julia environment first needs to be set up.
This needs to be done for the architecture on which the code will execute.

Please do:

- Clone the package, change into the `examples` folder, activate and instantiate this environment.
- Request the binary `OpenMPI_jll`
```
using MPIPreferences
MPIPreferences.use_jll_binary("OpenMPI_jll")
```
or `MPICH`
```
using MPIPreferences
MPIPreferences.use_jll_binary("MPICH_jll")
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
mpiexecjl -n 4julia  --project=. schur/heat/Poisson2D_cg_mpi_driver.jl
```

On Windows 11, in the bash, the following would work:
```
mpiexecjl -n 2 julia --project=. conc/shells/zc_mpi_driver.jl 
```


## How to run a Schur-complement CG parallel example


## How to run a CoNC-preconditioned CG (sequential) example

Two classes of problems solved here:
- Three-dimensional elasticity (composite of matrix with embedded fibers).
- General three-dimensional shells.

There are shell scripts to run the studies reported in the paper.

For a strong scaling simulation of an elasticity problem, there are:
- Moderately compressible matrix. Driver `fib-strong-***T***-cc-***X***.jl`.
- Nearly incompressible matrix. Driver `fib-strong-***T***-cni-***X***.jl`.


Here `***T***` stands for the type of the mesh (`hex` or `tet`), and `***X***`
stands for the coarse-grid evolution strategy: `const`, `grow`, `match`.

For the shells, there are scripts for strong and weak scaling:

- Z-section cantilever under torsional loading. Driver `zc-***S***-***X***.jl`

Here `***S***` stands for the scaling type (`strong` or `weak`), 
and `***X***`
stands for the coarse-grid evolution strategy: `const`, `growlin`, `growlin2`, `growlog`, `growsqr`, `match`.


For instance, when in the folder `FinEtoolsDDMethods.jl/examples`, execute
```
$ julia --project=. conc/shells/zc-weak-growlin.jl
```

*Warning*: Running the examples may require a beefy machine,
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

At the moment only the heat conduction and shell analysis examples have been cast in this form. In an interactive run on Windows, try
```
mpiexec -n 5 julia --project=. .\conc\heat\Poisson2D_mpi_driver.jl
```
or
```
mpiexec -n 5 julia --project=. .\conc\shells\zc_mpi_driver.jl
```

Batch execution on the Ookami A64FX nodes is described with the following `sbatch` script
for MPICH:
```
#!/usr/bin/env bash

#SBATCH --job-name=job_barrel
#SBATCH --output=job_barrel.log
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=24
#SBATCH --time=00:15:00
#SBATCH -p short

export JULIA_DEPOT_PATH="~/a64fx/depot"
module load julia
module load gcc/13.1.0
module load slurm
module load openmpi/gcc8/4.1.2

export JULIA_NUM_THREADS=1
export BLAS_THREADS=2

cd FinEtoolsDDMethods.jl/examples
/lustre/home/pkrysl/a64fx/depot/bin/mpiexecjl julia --project=. conc/shells/barrel_mpi_driver.jl
```
The above executes on 24 nodes (root + 23 partitions), using 2 BLAS threads per process.

And this one for OpenMPI:
```
#!/usr/bin/env bash

#SBATCH --job-name=job_zc
#SBATCH --output=job_zc_Np=8.log
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=9
#SBATCH --time=00:15:00
#SBATCH -p short

# Load OpenMPI and Julia
export JULIA_DEPOT_PATH="~/a64fx/depot"
module load julia
module load gcc/13.1.0
module load slurm
module load openmpi/gcc8/4.1.2

# Automatically set the number of Julia threads depending on number of Slurm threads
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export BLAS_THREADS=2

cd FinEtoolsDDMethods.jl/examples
mpiexec julia --project=. conc/shells/zc_mpi_driver.jl --n1 6 --Nc 100 --ref 100
```