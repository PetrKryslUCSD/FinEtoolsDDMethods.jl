cmd = `julia --project=. ./conc/heat/Poisson2D_seq_driver.jl`
run(cmd)
@info "======================================\nExpected 13 iterations, nearly zero error"

cmd = `julia --project=. ./conc/shells/zc_seq_driver.jl`
run(cmd)
@info "======================================\nExpected 37 iterations"


cmd = `julia --project=. ./conc/shells/barrel_seq_driver.jl`
run(cmd)
@info "======================================\nExpected 68 iterations"


cmd = `julia --project=. ./conc/shells/hyp_seq_driver.jl`
run(cmd)
@info "======================================\nExpected 68 iterations"


cmd = `julia --project=. ./conc/lindef/fib_seq_driver.jl`
run(cmd)
@info "======================================\nExpected 36 iterations"

