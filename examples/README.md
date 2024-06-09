# Examples

This is the README folder.

This is how to run an example:

- Clone the package, change into the `examples` folder, activate and instantiate this environment.
- Request the binary `OpenMPI_jll`
```
using MPIPreferences
MPIPreferences.use_jll_binary("OpenMPI_jll")
```
- Install the helper script `mpiexecjl`
```
using MPI
MPI.install_mpiexecjl()
```
Note the folder where the executable is installed.
- Run the example. `mpiexecjl` likely needs to be specified using the path of the folder in which it was installed.
```
 mpiexecjl -n 4 --project=. julia heat/Poisson2D_cg_mpi_driver.jl
```



