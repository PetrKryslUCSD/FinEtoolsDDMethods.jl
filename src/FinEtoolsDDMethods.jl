module FinEtoolsDDMethods

# Enable LSP look up in test modules
if false
    include("../test/runtests.jl")
end

using FinEtools

include("FENodeToPartitionMapModule.jl")
using .FENodeToPartitionMapModule: FENodeToPartitionMap
export FENodeToPartitionMap

include("PartitionSchurDDModule.jl")
using .PartitionSchurDDModule: DOF_KIND_INTERFACE, PartitionSchurDD, mark_interfaces!
export DOF_KIND_INTERFACE, PartitionSchurDD, mark_interfaces!

include("PartitionCoNCDDModule.jl")
using .PartitionCoNCDDModule: coarse_grid_partitioning
export coarse_grid_partitioning

include("cg.jl")

end # module FinEtoolsDDMethods
