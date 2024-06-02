module FinEtoolsDDParallel

# Enable LSP look up in test modules
if false
    include("../test/runtests.jl")
end

using FinEtools

include("FENodeToPartitionMapModule.jl")
using .FENodeToPartitionMapModule: FENodeToPartitionMap
export FENodeToPartitionMap
include("mesh.jl")
export make_partition_mesh
include("PartitionSchurDDModule.jl")
using .PartitionSchurDDModule: DOF_KIND_INTERFACE, PartitionSchurDD, mark_interfaces!
export DOF_KIND_INTERFACE, PartitionSchurDD, mark_interfaces!
include("cg.jl")

end # module FinEtoolsDDParallel
