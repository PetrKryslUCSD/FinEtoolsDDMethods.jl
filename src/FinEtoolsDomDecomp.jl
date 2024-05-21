module FinEtoolsDomDecomp

# Enable LSP look up in test modules
if false
    include("../test/runtests.jl")
end

using FinEtools

include("FENodeToPartitionMapModule.jl")

export FENodeToPartitionMap

end # module FinEtoolsDomDecomp
