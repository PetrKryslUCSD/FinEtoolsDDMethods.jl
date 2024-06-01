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

struct MatrixCache{T,IT,FACTOR}
    temp_f::Vector{T}
    temp_s::Vector{T}
    temp_i::Vector{T}
    temp_v::Vector{T}
    b_i_til::Vector{T}
    b_f::Vector{T}
    K_ii::SparseMatrixCSC{T,IT}
    K_fi::SparseMatrixCSC{T,IT}
    K_if::SparseMatrixCSC{T,IT}
    K_ff_factor::FACTOR
    dofnums_i::Vector{IT}
    result_i::Vector{T}
end

function MatrixCache(K::SparseMatrixCSC{T,IT}, b::Vector{T},
    u::F, global_u::F, global_node_numbers) where {F<:NodalField,T,IT}
    fr = dofrange(u, DOF_KIND_FREE)
    ir = dofrange(u, DOF_KIND_INTERFACE)
    dr = dofrange(u, DOF_KIND_DATA)
    K_ff = K[fr, fr]
    K_ii = K[ir, ir]
    K_ff_factor = lu(K_ff)
    temp_f = zeros(T, size(K_ff, 1))
    temp_s = zeros(T, size(K_ff, 1))
    temp_i = zeros(T, size(K_ii, 1))
    temp_v = zeros(T, size(K_ii, 1))
    result_i = zeros(T, size(K_ii, 1))
    u_d = gathersysvec(u, DOF_KIND_DATA)
    K_fd = K[fr, dr]
    K_id = K[ir, dr]
    K_if = K[ir, fr]
    K_fi = K[fr, ir]
    b_f = b[fr] - K_fd * u_d
    b_i_til = (b[ir] - K_id * u_d - K_if * (K_ff_factor \ b_f))
    dofnums_i = interface_degrees_of_freedom(global_u, global_node_numbers, u)
    MatrixCache{T,IT,typeof(K_ff_factor)}(temp_f, temp_s, temp_i, temp_v, b_i_til, b_f, K_ii, K_fi, K_if, K_ff_factor, dofnums_i, result_i)
end

function mul!(y, mc::MatrixCache{T}, v) where {T}
    mul!(mc.temp_f, mc.K_fi, v)
    ldiv!(mc.temp_s, mc.K_ff_factor, mc.temp_f)
    # mc.temp_s .= mc.K_ff_factor \ mc.temp_f
    mul!(mc.temp_i, mc.K_if, mc.temp_s)
    mul!(y, mc.K_ii, v)
    y .-= mc.temp_i
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
struct PartitionSchurDD{T,IT,FEN<:FENodeSet,FES<:AbstractFESet,F<:NodalField}
    fens::FEN
    fes::FES
    u::F
    global_node_numbers::Vector{IT}
    global_u::F
    mc::MatrixCache{T,IT}
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
    init_partition_field!(u, global_u, global_node_numbers)
    femm = make_partition_femm(fes)
    K = make_partition_matrix(femm, geom, u)
    I = make_partition_interior_load(femm, geom, u, global_node_numbers)
    B = make_partition_boundary_load(geom, u, global_node_numbers)
    mc = MatrixCache(K, sum_load_vectors(I, B, u), u, global_u, global_node_numbers)
    return PartitionSchurDD(fens, fes, u, global_node_numbers, global_u, mc)
end

function init_partition_field!(partition_u, global_u, global_node_numbers)
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

function interface_degrees_of_freedom(global_u, global_node_numbers, partition_u)
    global_from = first(dofrange(global_u, DOF_KIND_INTERFACE))
    partition_from = first(dofrange(partition_u, DOF_KIND_INTERFACE))
    partition_to = last(dofrange(partition_u, DOF_KIND_INTERFACE))
    id = Vector{Int}(undef, partition_to - partition_from + 1)
    p = 1
    for i in eachindex(global_node_numbers)
        g = global_node_numbers[i]
        for j in 1:ndofs(global_u)
            if global_u.kind[g, j] == DOF_KIND_INTERFACE
                gl = global_u.dofnums[g, j]
                id[p] = gl - global_from + 1
                p += 1
            end
        end
    end
    return id
end

function gather_partition_v_from_global_v!(partition_v, dofnums_i, global_v)
    partition_v .= @view global_v[dofnums_i]
    return partition_v
end

function scatter_partition_v_to_global_v!(global_v, dofnums_i, partition_v)
    global_v[dofnums_i] .+= partition_v
    return global_v
end

function mul_S_v!(partition::PartitionSchurDD, v)
    mc = partition.mc
    gather_partition_v_from_global_v!(mc.temp_v, mc.dofnums_i, v)
    mul!(mc.result_i, mc, mc.temp_v)
    return partition
end

function assemble_sol!(y, partition::PartitionSchurDD)
    mc = partition.mc
    return scatter_partition_v_to_global_v!(y, mc.dofnums_i, mc.result_i)
end

function assemble_rhs!(y, partition::PartitionSchurDD)
    mc = partition.mc
    return scatter_partition_v_to_global_v!(y, mc.dofnums_i, mc.b_i_til)
end

function partition_field_to_global_field!(global_u, global_node_numbers, partition_u,)
    @inbounds for i in 1:nents(partition_u)
        g = global_node_numbers[i]
        global_u.values[g, :] .= partition_u.values[i, :]
    end
    return global_u
end

function global_field_to_partition_field!(partition_u, global_node_numbers, global_u,)
    @inbounds for i in 1:nents(partition_u)
        g = global_node_numbers[i]
        partition_u.values[i, :] .= global_u.values[g, :]
    end
    return partition_u
end

function reconstruct_free!(partition::PartitionSchurDD)
    mc = partition.mc
    global_field_to_partition_field!(partition.u, partition.global_node_numbers, partition.global_u,)
    T_i = gathersysvec(partition.u, DOF_KIND_INTERFACE)
    T_f = mc.K_ff_factor \ (mc.b_f - mc.K_fi * T_i)
    scattersysvec!(partition.u, T_f, DOF_KIND_FREE)
    partition_field_to_global_field!(partition.global_u, partition.global_node_numbers, partition.u,)
    return partition.global_u
end

function partition_complement_diagonal!(y, partition::PartitionSchurDD)
    # S_ii = K_ii - K_if * (K_ff \ K_fi)
    mc = partition.mc
    mc.temp_i .= diag(mc.K_ii)
    for j  in 1:size(mc.K_ii, 1)
        ldiv!(mc.temp_s, mc.K_ff_factor, @view(mc.K_fi[:, j]))
        mc.temp_i[j] -= dot(@view(mc.K_if[j, :]), mc.temp_s)
    end
    return scatter_partition_v_to_global_v!(y, mc.dofnums_i, mc.temp_i)
end

function assemble_interface_matrix!(K_ii, partition::PartitionSchurDD)
    mc = partition.mc
    K_ii[mc.dofnums_i, mc.dofnums_i] += mc.K_ii
    return K_ii
end

end # module PartitionSchurDDModule
