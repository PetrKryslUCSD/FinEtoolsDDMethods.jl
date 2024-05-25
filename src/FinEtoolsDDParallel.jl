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
include("PartitionSchurDDModule.jl")

end # module FinEtoolsDDParallel
