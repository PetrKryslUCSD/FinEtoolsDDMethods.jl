println("Current folder: $(pwd())")

using Pkg

Pkg.activate(".")
Pkg.instantiate()

using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--nelperpart"
        help = "Number of elements per partition"
        arg_type = Int
        default = 200
        "--nbf1max"
        help = "Number 1D basis functions"
        arg_type = Int
        default = 5
        "--nfpartitions"
        help = "Number fine grid partitions"
        arg_type = Int
        default = 2
        "--overlap"
        help = "Overlap"
        arg_type = Int
        default = 1
        "--ref"
        help = "Refinement factor"
        arg_type = Int
        default = 2
        "--aspect"
        help = "Aspect ratio, length divided by thickness"
        arg_type = Float64
        default = 10
        "--visualize"
        help = "Write out visualization files?"
        arg_type = Bool
        default = false
    end
    return parse_args(s)
end

p = parse_commandline()

include(raw"LE5_Z_cantilever_overlapped_examples.jl")
using .LE5_Z_cantilever_overlapped_examples; 


LE5_Z_cantilever_overlapped_examples.test(;
    aspect=p["aspect"], nelperpart=p["nelperpart"], nbf1max=p["nbf1max"],
    nfpartitions=p["nfpartitions"], overlap=p["overlap"], ref=p["ref"],
    visualize=p["visualize"])

