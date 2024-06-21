
# for ne in 200 400 800; do for n1 in 2 3 4 5; do julia conc/lindef/sim.jl $ne $n1; done; done

println("Current folder: $(pwd())")
println("Arguments: nelperpart nbf1max ref nfpartitions kind")

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
nfpartitions = 2
if length(ARGS) > 3
    nfpartitions = parse(Int, ARGS[4])
end
kind = "hex"
if length(ARGS) > 4
    kind = ARGS[5]
end

include(raw"fibers_examples.jl")
using .fibers_examples; 

fibers_examples.test("soft_hard";
    kind=kind,
    Em=1.0, num=0.3, Ef=1.20e5, nuf=0.3,
    nelperpart=nelperpart, nbf1max=nbf1max, 
    nfpartitions=nfpartitions, ref=ref)

