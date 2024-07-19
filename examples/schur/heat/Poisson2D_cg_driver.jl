module Poisson2D_cg_driver
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsHeatDiff
using FinEtoolsDDMethods
using FinEtoolsDDMethods.PartitionSchurDDModule: mul_S_v!, assemble_rhs!, assemble_sol!
using FinEtoolsDDMethods.PartitionSchurDDModule: reconstruct_free!, partition_complement_diagonal!
using FinEtoolsDDMethods.PartitionSchurDDModule: assemble_interface_matrix!
using FinEtoolsDDMethods.CGModule: pcg_seq
using Metis
using Test
import Base: size, eltype
import LinearAlgebra: mul!
using LinearAlgebra
using LinearOperators
using SparseArrays
using Krylov


struct PartitionedSOp{T, IT} 
    s::IT
    z::T
    partitions::Vector{PartitionSchurDD}
end

function mul!(y::Vector{T}, Sop::SOP, v::Vector{T})  where {T, SOP<:PartitionedSOp}
    y .= zero(eltype(y))
    for p in Sop.partitions
        mul_S_v!(p, v)
        assemble_sol!(y,  p)
    end
    y
end
size(Sop::PartitionedSOp) = (Sop.s, Sop.s)
eltype(Sop::PartitionedSOp) = typeof(Sop.z)

function test()

    # println("""
    #
    # Heat conduction example described by Amuthan A. Ramabathiran
    # http://www.codeproject.com/Articles/579983/Finite-Element-programming-in-Julia:
    # Unit square, with known temperature distribution along the boundary,
    # and uniform heat generation rate inside.  Mesh of regular linear TRIANGLES,
    # in a grid of 1000 x 1000 edges (2M triangles, 1M degrees of freedom).
    # Version: 05/29/2017
    # """
    # )
    A = 1.0 # dimension of the domain (length of the side of the square)
    thermal_conductivity = [i == j ? one(Float64) : zero(Float64) for i = 1:2, j = 1:2] # conductivity matrix
    Q = -6.0 # internal heat generation rate
    tempf(x) = (1.0 .+ x[:, 1] .^ 2 .+ 2 * x[:, 2] .^ 2)#the exact distribution of temperature
    N = 100 # number of subdivisions along the sides of the square domain
    ndoms = 3

    fens, fes = T3block(A, A, N, N)

    geom = NodalField(fens.xyz)
    Temp = NodalField(zeros(count(fens), 1))

    
    # Partition the finite element mesh.
    femm = FEMMBase(IntegDomain(fes, TriRule(1)))
    C = dualconnectionmatrix(femm, fens, nodesperelem(boundaryfe(fes)))
    g = Metis.graph(C; check_hermitian=true)
    element_partitioning = Metis.partition(g, ndoms; alg=:KWAY)
    n2p = FENodeToPartitionMap(fens, fes, element_partitioning)
    # Mark all degrees of freedom at the interfaces
    mark_interfaces!(Temp, n2p)
    # Apply Dirichlet boundary conditions. This will change some of the
    # interface degrees of freedom to "data" degrees of freedom. 
    l1 = selectnode(fens; box=[0.0 0.0 0.0 A], inflate=1.0 / N / 100.0)
    l2 = selectnode(fens; box=[A A 0.0 A], inflate=1.0 / N / 100.0)
    l3 = selectnode(fens; box=[0.0 A 0.0 0.0], inflate=1.0 / N / 100.0)
    l4 = selectnode(fens; box=[0.0 A A A], inflate=1.0 / N / 100.0)
    List = vcat(l1, l2, l3, l4)
    setebc!(Temp, List, true, 1, tempf(geom.values[List, :])[:])
    applyebc!(Temp)
    numberdofs!(Temp, 1:count(fens), [DOF_KIND_FREE, DOF_KIND_INTERFACE, DOF_KIND_DATA])

    material = MatHeatDiff(thermal_conductivity)

    function make_partition_fields(partition_fens)
        partition_geom = NodalField(partition_fens.xyz)
        partition_Temp = NodalField(zeros(size(partition_fens.xyz, 1), 1))
        return partition_geom, partition_Temp
    end

    function make_partition_femm(partition_fes)
        return FEMMHeatDiff(IntegDomain(partition_fes, TriRule(1), 100.0), material)
    end

    function make_partition_matrix(partition_femm, partition_geom, partition_Temp)
        return conductivity(partition_femm, partition_geom, partition_Temp)
    end

    function make_partition_interior_load(partition_femm, partition_geom, partition_Temp, global_node_numbers)
        fi = ForceIntensity(Float64[Q])
        return distribloads(partition_femm, partition_geom, partition_Temp, fi, 3)
    end

    function make_partition_boundary_load(partition_geom, partition_Temp, global_node_numbers)
        return Float64[]
    end

    # Create partitions
    @info "Creating partitions"
    partitions = PartitionSchurDD[]
    for p  in 1:ndoms
        push!(partitions,
            PartitionSchurDD(make_partition_mesh(fens, fes, element_partitioning, p)..., Temp,
                make_partition_fields, make_partition_femm, make_partition_matrix, make_partition_interior_load, make_partition_boundary_load
            )
        )
    end

    @info "Assembling the righthand side"
    Sop = PartitionedSOp(length(dofrange(Temp, DOF_KIND_INTERFACE)), zero(eltype(Temp.values)), partitions)
    b = gathersysvec(Temp, DOF_KIND_INTERFACE)
    b .= 0.0
    for p in Sop.partitions
        assemble_rhs!(b,  p)
    end

    @info "Creating preconditioning matrix"
    # S_diag = zeros(size(b))
    # for p in Sop.partitions
    #     partition_complement_diagonal!(S_diag, p)
    # end
    K_ii = spzeros(size(b, 1), size(b, 1))
    for p in Sop.partitions
        assemble_interface_matrix!(K_ii, p)
    end
    K_ii_factor = lu(K_ii)
    @info "Solving the Linear System using CG"
    # @time (T_i, stats) = cg(Sop, b)
    # @time (T_i, stats) = cg(Sop, b; M=DiagonalPreconditioner(S_diag))
    @time (T_i, stats) = cg(Sop, b; ldiv=true, M=K_ii_factor)
    @show stats
    @info "Reconstructing value of free degrees of freedom"
    scattersysvec!(Temp, T_i, DOF_KIND_INTERFACE)
    for p in Sop.partitions
        reconstruct_free_dofs!(p)
    end

    @info "Exporting visualization"
    approx_T = Temp.values
    VTK.vtkexportmesh("approx.vtk", fes.conn, [geom.values approx_T], VTK.T3; scalars=[("Temperature", approx_T,)])

    @info "Computing error"
    Error = 0.0
    for k = 1:size(fens.xyz, 1)
        Error = Error .+ abs.(approx_T[k] .- tempf(reshape(fens.xyz[k, :], (1, 2))))
        # Error = Error .+ abs.(Temp.values[k, 1] .- tempf(reshape(fens.xyz[k, :], (1, 2))))
    end
    println("Error =$Error")


    # File =  "a.vtk"
    # MeshExportModule.vtkexportmesh (File, fes.conn, [geom.values Temp.values], MeshExportModule.T3; scalars=Temp.values, scalars_name ="Temperature")

    # @test Error[1] < 1.e-5

    true
end
test()
nothing
end
