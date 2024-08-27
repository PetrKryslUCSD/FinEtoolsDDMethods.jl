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

include("CompatibilityModule.jl")

include("cg.jl")
using .FinEtoolsDDMethods.CGModule: KSP_NORM_UNPRECONDITIONED, KSP_NORM_NATURAL
export KSP_NORM_UNPRECONDITIONED, KSP_NORM_NATURAL

include("PartitionCoNCModule.jl")
include("DDCoNCSeqModule.jl")
include("DDCoNCThrModule.jl")

function meminfo_julia(where = "")
    toint(n) = Int(ceil(n))
    @info """Memory: $(where)
    GC total:  $(toint(Base.gc_total_bytes(Base.gc_num())/2^20)) [MiB]
    GC live:   $(toint(Base.gc_live_bytes()/2^20)) [MiB]
    JIT:       $(toint(Base.jit_total_bytes()/2^20)) [MiB]
    Max. RSS:  $(toint(Sys.maxrss()/2^20)) [MiB]
    """
end

end # module FinEtoolsDDMethods
