println("Current folder: $(pwd())")

using Pkg

Pkg.activate(".")
Pkg.instantiate()

using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--Nc"
        help = "Number of clusters"
        arg_type = Int
        default = 2
        "--n1"
        help = "Number 1D basis functions"
        arg_type = Int
        default = 5
        "--Np"
        help = "Number fine grid partitions"
        arg_type = Int
        default = 2
        "--No"
        help = "Number of overlaps"
        arg_type = Int
        default = 1
        "--ref"
        help = "Refinement factor"
        arg_type = Int
        default = 2
        "--itmax"
        help = "Maximum number of iterations allowed"
        arg_type = Int
        default = 2000
        "--relrestol"
        help = "Relative residual tolerance"
        arg_type = Float64
        default = 1.0e-6
        "--visualize"
        help = "Write out visualization files?"
        arg_type = Bool
        default = false
    end
    return parse_args(s)
end

p = parse_commandline()

include(raw"z_cantilever_seq_examples.jl")
using .z_cantilever_seq_examples; 


z_cantilever_seq_examples.test(;
    Nc=p["Nc"], n1=p["n1"],
    Np=p["Np"], No=p["No"], ref=p["ref"],
    itmax=p["itmax"], relrestol=p["relrestol"],
    visualize=p["visualize"])

