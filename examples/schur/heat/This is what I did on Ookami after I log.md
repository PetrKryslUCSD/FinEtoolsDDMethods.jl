This is what I did on Ookami after I logged on. 

My setup of the environment looks like this:
```
export JULIA_DEPOT_PATH="~/a64fx/depot"
module load julia
module load gcc/13.1.0
module load slurm
# module load mvapich2/arm22/2.3.7
module load openmpi/gcc8/4.1.2
```

1. I cloned the repository FinEtoolsDDMethods.jl from GitHub.
2. I activated the environment and instantiated the packages.
 `julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'`
3. I ran julia interactively to install mpiexecjl.
 `julia --project=.`
4. I ran srun to get an interactive session.
 `srun -p short -n 10 --ntasks-per-node=1 --pty bash`
5. I ran the example.
 cd FinEtoolsDDMethods.jl/examples/
 `~/a64fx/depot/bin/mpiexecjl -n 4 julia --project=. heat/Poisson2D_cg_mpi_driver.jl`
 
After several minutes, the job was terminated. The error message is below.
Note well: On my laptop this example runs to completion in ~70 seconds.
```
julia: symbol lookup error: /lustre/home/pkrysl/a64fx/depot/artifacts/58dcf187642cdfbafb3581993ca3d8de565acc78/lib/openmpi/mca_pmix_pmix3x.so: undefined symbol: opal_libevent2022_evthread_use_pthreads
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
[1972752] signal (15): Terminated
in expression starting at /lustre/home/pkrysl/a64fx/FinEtoolsDDMethods.jl/examples/heat/Poisson2D_cg_mpi_driver.jl:160
_ZN12_GLOBAL__N_117InterleavedAccess13runOnFunctionERN4llvm8FunctionE at /lustre/software/julia/julia-1.10.3/lib/julia/libLLVM-15jl.so (unknown line)
_ZN12_GLOBAL__N_117InterleavedAccess13runOnFunctionERN4llvm8FunctionE at /lustre/software/julia/julia-1.10.3/lib/julia/libLLVM-15jl.so (unknown line)
unknown function (ip: (nil))
Allocations: 14621931 (Pool: 14605380; Big: 16551); GC: 19
julia: symbol lookup error: /lustre/home/pkrysl/a64fx/depot/artifacts/58dcf187642cdfbafb3581993ca3d8de565acc78/lib/openmpi/mca_pmix_pmix3x.so: undefined symbol: opal_libevent2022_evthread_use_pthreads
julia: symbol lookup error: /lustre/home/pkrysl/a64fx/depot/artifacts/58dcf187642cdfbafb3581993ca3d8de565acc78/lib/openmpi/mca_pmix_pmix3x.so: undefined symbol: opal_libevent2022_evthread_use_pthreads
--------------------------------------------------------------------------
mpiexec detected that one or more processes exited with non-zero status, thus causing
the job to be terminated. The first process to do so was:
  Process name: [[7833,1],1]
  Exit code:    127
```
