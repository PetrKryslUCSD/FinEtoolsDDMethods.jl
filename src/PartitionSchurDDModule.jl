"""
    PartitionSchurDDModule  

Module for operations on partitions of finite element models four solves based
on the Schur complement.
"""
module PartitionSchurDDModule

__precompile__(true)

using FinEtools
const DOF_KIND_INTERFACE::KIND_INT = 3

"""
    PartitionSchurDD

Map from finite element nodes to the partitions to which they belong.

- `map` = map as a vector of vectors. If a node belongs to a single partition,
  the inner vector will have a single element.
"""
struct PartitionSchurDD{T, IT}
    K_ff::SparseMatrixCSC{T, IT}
    K_fi::SparseMatrixCSC{T, IT}
    K_fd::SparseMatrixCSC{T, IT}
    K_if::SparseMatrixCSC{T, IT}
    K_ii::SparseMatrixCSC{T, IT}
    K_id::SparseMatrixCSC{T, IT}
end

function PartitionSchurDD{T, IT}(K::SparseMatrixCSC{T, IT}, u::F) where {F<:NodalField,T<:Number,IT<:Integer}
    fr = dofrange(Temp, DOF_KIND_FREE)
    ir = dofrange(Temp, DOF_KIND_INTERFACE)
    dr = dofrange(Temp, DOF_KIND_DATA)
    return PartitionSchurDD(K[fr, fr], K[fr, ir], K[fr, dr], K[ir, fr], K[ir, ir], K[ir, dr])
end



end # module PartitionSchurDDModule
