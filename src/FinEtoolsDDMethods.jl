module FinEtoolsDDMethods

# Enable LSP look up in test modules
if false
    include("../test/runtests.jl")
end

using FinEtools

include("utility.jl")
export mebibytes

include("FENodeToPartitionMapModule.jl")
using .FENodeToPartitionMapModule: FENodeToPartitionMap
export FENodeToPartitionMap

include("PartitionSchurDDModule.jl")
using .PartitionSchurDDModule: DOF_KIND_INTERFACE, PartitionSchurDD, mark_interfaces!
export DOF_KIND_INTERFACE, PartitionSchurDD, mark_interfaces!

include("CoNCUtilitiesModule.jl")
using .CoNCUtilitiesModule: cluster_partitioning, shell_cluster_partitioning
export cluster_partitioning, shell_cluster_partitioning

include("CompatibilityModule.jl")

include("cg.jl")
using .FinEtoolsDDMethods.CGModule: KSP_NORM_UNPRECONDITIONED, KSP_NORM_NATURAL
export KSP_NORM_UNPRECONDITIONED, KSP_NORM_NATURAL

include("PartitionCoNCModule.jl")
using .PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData
export CoNCPartitioningInfo, CoNCPartitionData
using .PartitionCoNCModule: partition_size, mean_partition_size
export partition_size, mean_partition_size

include("DDCoNCSeqModule.jl")
include("DDCoNCMPIModule.jl")

end # module FinEtoolsDDMethods
