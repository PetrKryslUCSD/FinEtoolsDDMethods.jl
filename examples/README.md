# Examples

This is the `examples` folder.
There are two sub folders: 
- `schur`: Schur-complement-based solver using domain decomposition and
conjugate gradients on the complement matrix. Some of the examples are
parallelized with MPI.
- `conc`: coherent node cluster (CoNC) model reduction is used as a global
solver in a preconditioned conjugate gradient based on the decomposition at two
levels: local (classical additive Schwarz based on overlapping subdomains), and
global (reduced model based on coherent clusters). The examples do not run in
parallel yet.

## Setting up the MPI environment

For parallel execution, the Julia environment first needs to be set up. This
needs to be done for the architecture on which the code will execute.

Please do:

- Clone the package, change into the `examples` folder, activate and instantiate
  this environment.
- Request the binary `OpenMPI_jll`
```
using MPIPreferences
MPIPreferences.use_jll_binary("OpenMPI_jll")
```
or `MPICH` (preferred on Ookami)
```
using MPIPreferences
MPIPreferences.use_jll_binary("MPICH_jll")
```
Or, use the trampoline
```
MPIPreferences.use_jll_binary( "MPItrampoline_jll")
```
Nothing needs to be done to test with MPI on Windows 11.
- Install the helper script `mpiexecjl`
```
using MPI
MPI.install_mpiexecjl()
```
Note the folder where the executable is installed.
- Run the example. `mpiexecjl` likely needs to be specified using the path of the folder in which it was installed.
```
mpiexecjl -n 4 julia  --project=. schur/heat/Poisson2D_cg_mpi_driver.jl
```

On Windows 11, in the bash, the following would work:
```
mpiexecjl -n 2 julia --project=. conc/shells/zc_mpi_driver.jl 
```


## How to run a Schur-complement CG parallel example

Only a few examples are provided.
```
mpiexecjl -n 4 julia  --project=. schur/heat/Poisson2D_cg_mpi_driver.jl
```



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
Then fire up Julia giving it the name of the driver:
```
julia --project=. conc/shells/zc_seq_driver.jl
```
There are also `hyp_seq_driver.jl` (hyperbolic paraboloid shell with varying thickness), and
`barrel_seq_driver.jl` (internally pressurized barrel).

Run
```
$ julia conc/shells/barrel_seq_driver.jl --help
```
to see the available options.

## How to run a CoNC-preconditioned CG (MPI-parallel) example

At the moment  the heat conduction, linear elasticity,  and shell analysis examples have been cast in this form. In an interactive run on Windows, try
```
mpiexecjl -n 5 julia --project=. ./conc/heat/Poisson2D_mpi_driver.jl
```
or
```
mpiexec -n 5 julia --project=. ./conc/shells/zc_mpi_driver.jl
```

### Ookami

Batch execution on the Ookami A64FX nodes is described with a `sbatch` script.
The batch file may be generated with the script `make-weak-zc-const.sh`
and submitted with `sbatch`
```
for Np in 384 ; do 
  bash conc/shells/make-weak-zc-const.sh --machine ookami --Nepp 1000 --Np $Np --Ntpn 64 > do-$Np.sh; 
  sbatch do-$Np.sh; 
done
```
For OpenMPI replace `mpiexec` with `/lustre/home/pkrysl/a64fx/depot/bin/mpiexecjl `.

In order to precompile Julia, use interactive command line:
```
srun -N 1 -n 48 -t 00:30:00 -p short --pty bash
```

### Expanse

Expanse's standard compute nodes are each powered by two 64-core AMD EPYC 7742 processors and contain 256 GB of DDR4 memory.

Batch execution on the Expanse A64FX nodes is described with a `sbatch` script.
The batch file may be generated with the script `make-weak-zc-const.sh`
and submitted with `sbatch` (note the machine specification)
```
for Np in 384 ; do 
  bash conc/shells/make-weak-zc-const.sh --machine expanse --Nepp 1000 --Np $Np --Ntpn 64 > do-$Np.sh; 
  sbatch do-$Np.sh; 
done
```
