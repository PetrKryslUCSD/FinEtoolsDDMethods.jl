"""
    FENodeToPartitionMapModule  

Module to construct a map from finite element nodes to partitions to which they belong.
"""
module FENodeToPartitionMapModule

__precompile__(true)

using FinEtools

function _make_map(fens, fes, element_partitioning)
    map = Vector{Vector{eltype(element_partitioning)}}(undef, count(fens))
    for i in eachindex(fes) 
        for j in fes.conn[i]
            if map[j] === nothing
                map[j] = Vector{eltype(element_partitioning)}()
            end
            push!(map[j], element_partitioning[i])
        end
    end
    return map
end


function _make_map(n2e, element_partitioning)
    map = Vector{Vector{eltype(element_partitioning)}}(undef, length(n2e.map))
    Threads.@threads for i in eachindex(n2e.map) 
        for j in n2e.map[i]
            if map[i] === nothing
                map[i] = Vector{eltype(element_partitioning)}()
            end
            push!(map[i], element_partitioning[j])
        end
    end
    return map
end


"""
    FENodeToPartitionMap

Map from finite element nodes to the partitions to which they belong.

- `map` = map as a vector of vectors. If a node belongs to a single partition,
  the inner vector will have a single element.
"""
struct FENodeToPartitionMap{IT}
    # Map as a vector of vectors.
    map::Vector{Vector{IT}}
end

"""
    FENodeToPartitionMap(n2e::N2EMAP, element_partitioning::Vector{IT}) where {N2EMAP<:FENodeToFEMap,N,IT<:Integer}

Map from finite element nodes to the partitions to which they belong.

# Arguments
- `n2e::N2EMAP`: A mapping between finite element nodes and finite elements.
- `element_partitioning::Vector{IT}`: A vector specifying the partitioning of elements.

"""
function FENodeToPartitionMap(
    n2e::N2EMAP,
    element_partitioning::Vector{IT},
) where {N2EMAP<:FENodeToFEMap,IT<:Integer}
    return FENodeToPartitionMap(_make_map(n2e, element_partitioning))
end

"""
    FENodeToPartitionMap(
        fens, fes::FE, element_partitioning
    ) where {FE<:AbstractFESet}

Map from finite element nodes to the nodes connected to them by elements.

Convenience constructor.
"""
function FENodeToPartitionMap(
    fens, fes::FE, element_partitioning
) where {FE<:AbstractFESet}
    n2e = FENodeToFEMap(fes, count(fens))
    return FENodeToPartitionMap(_make_map(n2e, element_partitioning))
end

end
