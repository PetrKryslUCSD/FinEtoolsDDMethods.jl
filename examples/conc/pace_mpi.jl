mpiexecjl = "mpiexec"

cmd = `$(mpiexecjl) -n 7 julia --project=. ./conc/heat/Poisson2D_mpi_driver.jl`
run(cmd)
@info "======================================\nExpected 13 iterations, nearly zero error"

cmd = `$(mpiexecjl) -n 7 julia --project=. ./conc/shells/zc_mpi_driver.jl`
run(cmd)
@info "======================================\nExpected 37 iterations"


cmd = `$(mpiexecjl) -n 7 julia --project=. ./conc/shells/barrel_mpi_driver.jl`
run(cmd)
@info "======================================\nExpected 69 iterations" # why? 68 in serial


cmd = `$(mpiexecjl) -n 7 julia --project=. ./conc/shells/hyp_mpi_driver.jl`
run(cmd)
@info "======================================\nExpected 28 iterations"


cmd = `$(mpiexecjl) -n 7 julia --project=. ./conc/lindef/fib_mpi_driver.jl`
run(cmd)
@info "======================================\nExpected 67 iterations"

