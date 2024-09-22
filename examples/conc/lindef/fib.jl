println("Current folder: $(pwd())")

using Pkg

Pkg.activate(".")
Pkg.instantiate()

using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--filename"
        help = "Use filename to name the output files"
        arg_type = String
        default = ""
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
        "--N"
        help = "Number of element edges in one direction"
        arg_type = Int
        default = 2
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
        "--kind"
        help = "hex or tet"
        arg_type = String
        default = "hex"
        "--Ef"
        help = "Young's modulus of the fibres"
        arg_type = Float64
        default = 1.00e5
        "--nuf"
        help = "Poisson ratio of the fibres"
        arg_type = Float64
        default = 0.3
        "--Em"
        help = "Young's modulus of the matrix"
        arg_type = Float64
        default = 1.00e3
        "--num"
        help = "Poisson ratio of the matrix"
        arg_type = Float64
        default = 0.4999
        "--peek"
        help = "Peek at the iterations?"
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

include(raw"fib_seq_examples.jl")
using .fib_seq_examples; 

fib_seq_examples.test(;
    filename=p["filename"],
    kind=p["kind"], 
    ref=p["ref"], 
    Em=p["Em"], num=p["num"], Ef=p["Ef"], nuf=p["nuf"],
    Nc=p["Nc"], n1=p["n1"], Np=p["Np"], No=p["No"], 
    itmax=p["itmax"], relrestol=p["relrestol"],
    peek=p["peek"],
    visualize = p["visualize"])

