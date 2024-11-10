cmd = `julia --project=. ./conc/heat/Poisson2D_seq_driver.jl`
run(cmd)
@info "======================================\nExpected 13 iterations, 5.86281e-04 error"

cmd = `julia --project=. ./conc/shells/zc_seq_driver.jl`
run(cmd)
@info "======================================\nExpected 26 iterations"


cmd = `julia --project=. ./conc/shells/barrel_seq_driver.jl`
run(cmd)
@info "======================================\nExpected 69 iterations"


cmd = `julia --project=. ./conc/shells/hyp_seq_driver.jl`
run(cmd)
@info "======================================\nExpected 28 iterations"


cmd = `julia --project=. ./conc/lindef/fib_seq_driver.jl --Np 7`
run(cmd)
@info "======================================\nExpected 67 iterations"

