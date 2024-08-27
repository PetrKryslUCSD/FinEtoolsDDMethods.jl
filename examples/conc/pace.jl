# Take the examples from the paper and pace them through the default test.
include(raw"lindef/fibers_examples.jl")
fibers_examples.test()

include(raw"lindef/fibers_examples.jl")
fibers_examples.test(kind="tet")

include(raw"lindef/fibers_seq_examples.jl")
fibers_seq_examples.test()

include(raw"lindef/fibers_seq_examples.jl")
fibers_seq_examples.test(kind="tet")

include(raw"lindef/fibers_thr_examples.jl")
fibers_thr_examples.test()

include(raw"lindef/fibers_thr_examples.jl")
fibers_thr_examples.test(kind="tet")

include(raw"shells/barrel_examples.jl")
barrel_examples.test(stabilize=false)
barrel_examples.test(stabilize=true)

include(raw"shells/cos_2t_p_hyp_free_examples.jl")
cos_2t_p_hyp_free_examples.test()

include(raw"shells/LE5_Z_cantilever_examples.jl")
LE5_Z_cantilever_examples.test()