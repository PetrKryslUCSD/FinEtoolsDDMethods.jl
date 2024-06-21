
# for ne in 200 400 800; do for n1 in 2 3 4 5; do julia conc/lindef/sim.jl $ne $n1; done; done

println("Current folder: $(pwd())")

using Pkg

Pkg.activate(".")
# Pkg.instantiate()

nelperpart = 200
if length(ARGS) > 0
    nelperpart = parse(Int, ARGS[1])
end
nbf1max = 3
if length(ARGS) > 1
    nbf1max = parse(Int, ARGS[2])
end
ref = 1
if length(ARGS) > 2
    ref = parse(Int, ARGS[3])
end
kind = "hex"
if length(ARGS) > 3
    kind = ARGS[4]
end

include(raw"fibers_soft_hard_examples.jl")
using .fibres_soft_hard_examples; 

fibres_soft_hard_examples.test(; kind = kind, nelperpart = nelperpart, nbf1max = nbf1max, ref = ref)

