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
using .PartitionCoNCDDModule: cluster_partitioning, shell_cluster_partitioning
using .PartitionCoNCDDModule: fine_grid_partitions
using .PartitionCoNCDDModule: preconditioner
export cluster_partitioning, shell_cluster_partitioning, fine_grid_partitions, preconditioner

include("PartitionCoNCDDMPIModule.jl")

include("cg.jl")

end # module FinEtoolsDDMethods
