println("Current folder: $(pwd())")

using Pkg

Pkg.activate(".")
Pkg.instantiate()

using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--nelperpart"
        help = "Number of elements per partition"
        arg_type = Int
        default = 1200
        "--nbf1max"
        help = "Number 1D basis functions"
        arg_type = Int
        default = 5
        "--nfpartitions"
        help = "Number fine grid partitions"
        arg_type = Int
        default = 16
        "--overlap"
        help = "Overlap"
        arg_type = Int
        default = 1
        "--ref"
        help = "Refinement factor"
        arg_type = Int
        default = 2
        "--itmax"
        help = "Maximum number of iterations allowed"
        arg_type = Int
        default = 200
        "--relrestol"
        help = "Relative residual tolerance"
        arg_type = Float64
        default = 1.0e-6
        "--stabilize"
        help = "Should the axial rotation be stabilized? factor"
        arg_type = Bool
        default = false
        "--visualize"
        help = "Write out visualization files?"
        arg_type = Bool
        default = false
    end
    return parse_args(s)
end

p = parse_commandline()

include(raw"barrel_overlapped_examples.jl")
using .barrel_overlapped_examples; 


barrel_overlapped_examples.test(;
    nelperpart=p["nelperpart"], nbf1max=p["nbf1max"],
    nfpartitions=p["nfpartitions"], overlap=p["overlap"], ref=p["ref"],
    itmax=p["itmax"], relrestol=p["relrestol"],
    stabilize=p["stabilize"],
    visualize=p["visualize"])

