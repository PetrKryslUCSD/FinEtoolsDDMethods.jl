"""
    PartitionSchurDDModule  

Module for operations on partitions of finite element models four solves based
on the Schur complement.
"""
module PartitionSchurDDModule

__precompile__(true)

using FinEtools
using SparseArrays
using Krylov
using LinearOperators
using SparseArrays
using LinearAlgebra
import Base: size, eltype
import LinearAlgebra: mul!
using ..FENodeToPartitionMapModule: FENodeToPartitionMap

# function spy_matrix(A::SparseMatrixCSC, name="")
#     I, J, V = findnz(A)
#     p = PlotlyLight.Plot()
#     p(x=J, y=I, mode="markers")
#     p.layout.title = name
#     p.layout.yaxis.title = "Row"
#     p.layout.yaxis.range = [size(A, 1) + 1, 0]
#     p.layout.xaxis.title = "Column"
#     p.layout.xaxis.range = [0, size(A, 2) + 1]
#     p.layout.xaxis.side = "top"
#     p.layout.margin.pad = 10
#     display(p)
# end

struct MatrixCache{T, IT}
    temp_f::Vector{T}
    temp_i::Vector{T}
    b_i::Vector{T}
    K_ii::SparseMatrixCSC{T, IT}
    K_fi::SparseMatrixCSC{T, IT}
    K_if::SparseMatrixCSC{T, IT}
    K_ff_factor::SparseArrays.CHOLMOD.Factor{T, IT}
    K_ii_factor::SparseArrays.CHOLMOD.Factor{T, IT}
    result_i::Vector{T}
end

function MatrixCache(K::SparseMatrixCSC{T, IT}, b::Vector{T}, u::F) where {F<:NodalField,T, IT}
    fr = dofrange(u, DOF_KIND_FREE)
    ir = dofrange(u, DOF_KIND_INTERFACE)
    dr = dofrange(u, DOF_KIND_DATA)
    K_ff = K[fr, fr]
    K_ii = K[ir, ir]
    K_ff_factor = cholesky(K_ff)
    K_ii_factor = cholesky(K_ii)
    temp_f, temp_i, result_i = zeros(T, size(K_ff, 1)), zeros(T, size(K_ii, 1)), zeros(T, size(K_ii, 1))
    u_d = gathersysvec(u, DOF_KIND_DATA)
    K_fd = K[fr, dr]
    K_id = K[ir, dr]
    K_if = K[ir, fr]
    K_fi = K[fr, ir]
    b_i = (b[ir] - K_id * u_d - K_if * (K_ff_factor \ (b[fr] - K_fd * u_d)))
    MatrixCache{T, IT}(temp_f, temp_i, b_i, K_ii, K_fi, K_if, K_ff_factor, K_ii_factor, result_i)
end

function mul!(y, mc::MatrixCache{T}, v) where {T}
    mul!(mc.temp_f, mc.K_fi, v)
    mul!(mc.temp_i, mc.K_if, (mc.K_ff_factor \ mc.temp_f))
    mul!(y, mc.K_ii, v) 
    y .-= mc.tempi
    y
end

# function mul!(y::Vector{Float64}, F::SparseArrays.CHOLMOD.Factor{Float64, Int64}, v::Vector{Float64})
#     y .= F \ v
# end

const DOF_KIND_INTERFACE::KIND_INT = 3

"""
    PartitionSchurDD

Partition for the Schur complement solver for a partitioned finite element model.
"""
struct PartitionSchurDD{T, IT, FEN<:FENodeSet, FES<:AbstractFESet, F<:NodalField}
    fens::FEN
    fes::FES 
    u::F
    global_node_numbers::Vector{IT}
    global_u::F
    mc::MatrixCache{T, IT}
end

function sum_load_vectors(I, B, u)
    ar = dofrange(u, DOF_KIND_ALL)
    if length(I) != length(ar)
        I = zeros(eltype(I), length(ar))
    end        
    if length(B) != length(ar)
        B = zeros(eltype(B), length(ar))
    end     
    return I + B
end

function PartitionSchurDD(fens::FEN, fes::FES, global_node_numbers::Vector{IT}, global_u::F, 
    make_partition_fields, make_partition_femm, make_partition_matrix, make_partition_interior_load, make_partition_boundary_load   
    ) where {FEN<:FENodeSet,FES<:AbstractFESet,F<:NodalField{T,IT} where {T<:Number,IT<:Integer},IT<:Integer}
    # validate_mesh(fens, fes);
    geom, u = make_partition_fields(fens)
    global_field_to_partition_field!(u, global_u, global_node_numbers)
    femm = make_partition_femm(fes)
    K = make_partition_matrix(femm, geom, u)
    I = make_partition_interior_load(femm, geom, u, global_node_numbers)
    B = make_partition_boundary_load(geom, u, global_node_numbers)
    mc = MatrixCache(K, sum_load_vectors(I, B, u), u)
    return PartitionSchurDD(fens, fes, u, global_node_numbers, global_u, mc)
end

function global_field_to_partition_field!(partition_u, global_u, global_node_numbers)
    # Transfer the kinds and values from the global field. Not the degree of freedom numbers.
    for i in 1:nents(partition_u)
        partition_u.kind[i, :] .= global_u.kind[global_node_numbers[i], :]
        partition_u.values[i, :] .= global_u.values[global_node_numbers[i], :]
    end
    numberdofs!(partition_u, 1:nents(partition_u), [DOF_KIND_FREE, DOF_KIND_INTERFACE, DOF_KIND_DATA])
    return partition_u
end

function mark_interfaces!(u::F, n2p::FENodeToPartitionMap) where {F<:NodalField}
    for i in eachindex(n2p.map)
        if length(n2p.map[i]) > 1
            u.kind[i, :] .= DOF_KIND_INTERFACE
        end
    end
end

# function partition_v_from_global_field!(v, global_u, global_node_numbers, partition_u)
#     for i in eachindex(global_node_numbers)
#         g = global_node_numbers[i]
#         for j in 1:ndofs(global_u)
#             if global_u.kind[g, j] == DOF_KIND_INTERFACE
#                 l = partition_u.dofnums[i, j]
#                 v[l] = global_u.values[g, j]
#             end
#         end
#     end
#     return u
# end

function partition_v_from_global_v!(partition_v, global_u, global_node_numbers, partition_u)
    for i in eachindex(global_node_numbers)
        g = global_node_numbers[i]
        for j in 1:ndofs(global_u)
            if global_u.kind[g, j] == DOF_KIND_INTERFACE
                l = partition_u.dofnums[i, j]
                gl = global_u.dofnums[g, j]
                partition_v[l] = global_v[gl]
            end
        end
    end
    return u
end

# function add_v_to_global_field!(v, global_u, global_node_numbers, partition_u)
#     for i in eachindex(global_node_numbers)
#         g = global_node_numbers[i]
#         for j in 1:ndofs(global_u)
#             if global_u.kind[g, j] == DOF_KIND_INTERFACE
#                 l = partition_u.dofnums[i, j]
#                 global_u.values[g, j] += v[l] 
#             end
#         end
#     end
#     return u
# end

function add_partition_v_to_global_v!(global_v, global_u, global_node_numbers, partition_u, partition_v)
    for i in eachindex(global_node_numbers)
        g = global_node_numbers[i]
        for j in 1:ndofs(global_u)
            if global_u.kind[g, j] == DOF_KIND_INTERFACE
                l = partition_u.dofnums[i, j]
                gl = global_u.dofnums[g, j]
                global_v[gl] += partition_v[l] 
            end
        end
    end
    return global_v
end

function mul_S_v!(partition::PartitionSchurDD, v)
    mc = partition.mc
    partition_v_from_global_v!(mc.temp_i, partition.global_u, partition.global_node_numbers, partition.u)
    mul!(mc.result_i, mc, mc.temp_i)
    return partition
end

function assemble_sol!(y,  partition::PartitionSchurDD)
    mc = partition.mc
    return add_partition_v_to_global_v!(y, partition.global_u, partition.global_node_numbers, partition.u, mc.result_i)
end

function assemble_rhs!(y,  partition::PartitionSchurDD)
    mc = partition.mc
    return add_partition_v_to_global_v!(y, partition.global_u, partition.global_node_numbers, partition.u, mc.b_i)
end

end # module PartitionSchurDDModule
