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

include("CoNCUtilitiesModule.jl")
using .CoNCUtilitiesModule: cluster_partitioning, shell_cluster_partitioning
export cluster_partitioning, shell_cluster_partitioning

include("PartitionCoNCDDSEQModule.jl")
include("PartitionCoNCDDMPIModule.jl")
using .PartitionCoNCDDMPIModule: CoNCPartitioningInfo, CoNCPartitionData
using .PartitionCoNCDDMPIModule: partition_multiply!, precondition_global_solve!, precondition_local_solve!
export CoNCPartitioningInfo, CoNCPartitionData
export partition_multiply!, precondition_global_solve!, precondition_local_solve!

include("cg.jl")

end # module FinEtoolsDDMethods
