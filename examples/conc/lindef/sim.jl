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
refmultiplier = 1
if length(ARGS) > 2
    refmultiplier = parse(Int, ARGS[3])
end

include(raw"fibers_soft_hard_tet_examples.jl")
using .fibres_soft_hard_tet_examples; 

fibres_soft_hard_tet_examples.test(; nelperpart = nelperpart, nbf1max = nbf1max, refmultiplier = refmultiplier)

