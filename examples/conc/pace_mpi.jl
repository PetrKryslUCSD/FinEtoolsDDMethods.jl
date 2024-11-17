mpiexecjl = "mpiexec"

cmd = `$(mpiexecjl) -n 7 julia --project=. ./conc/heat/Poisson2D_mpi_driver.jl`
run(cmd)
@info "======================================\nExpected 13 iterations, 5.86281e-04 error"

cmd = `$(mpiexecjl) -n 7 julia --project=. ./conc/shells/zc_mpi_driver.jl`
run(cmd)
@info "======================================\nExpected 26 iterations"





cmd = `$(mpiexecjl) -n 7 julia --project=. ./conc/shells/hyp_mpi_driver.jl`
run(cmd)
@info "======================================\nExpected 28 iterations"


cmd = `$(mpiexecjl) -n 7 julia --project=. ./conc/lindef/fib_mpi_driver.jl`
run(cmd)
@info "======================================\nExpected 67 iterations"

cmd = `$(mpiexecjl) -n 7 julia --project=. ./conc/shells/barrel_mpi_driver.jl`
run(cmd)
@info "======================================\nExpected 69 iterations" 