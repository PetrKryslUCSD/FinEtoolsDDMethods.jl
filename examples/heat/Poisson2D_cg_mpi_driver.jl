"""
mpiexec -n 3 julia --project=. ./heat/Poisson2D_cg_mpi_driver.jl
"""
module Poisson2D_cg_mpi_driver
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsHeatDiff
using FinEtoolsDDParallel
using FinEtoolsDDParallel.PartitionSchurDDModule: mul_S_v!, assemble_rhs!, assemble_sol!
using FinEtoolsDDParallel.PartitionSchurDDModule: reconstruct_free!, partition_complement_diagonal!
using FinEtoolsDDParallel.PartitionSchurDDModule: assemble_interface_matrix!
using FinEtoolsDDParallel.CGModule: pcg_seq, pcg_mpi
using Metis
using Test
using LinearAlgebra
using SparseArrays
using PlotlyLight
using Krylov
using LinearOperators
using SparseArrays
using LinearAlgebra
import Base: size, eltype
import LinearAlgebra: mul!
using MPI

function mul_y_S_v!(y, partition, v) 
    y .= zero(eltype(y))
    if partition !== nothing
        mul_S_v!(partition, v)
        assemble_sol!(y, partition)
    end
    y
end

function test()
    A = 1.0 # dimension of the domain (length of the side of the square)
    thermal_conductivity = [i == j ? one(Float64) : zero(Float64) for i = 1:2, j = 1:2] # conductivity matrix
    Q = -6.0 # internal heat generation rate
    tempf(x) = (1.0 .+ x[:, 1] .^ 2 .+ 2 * x[:, 2] .^ 2) #the exact distribution of temperature
    N = 100 # number of subdivisions along the sides of the square domain
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    npartitions = nprocs - 1

    if rank == 0
        @info "Number of processes: $nprocs"
        @info "Number of partitions: $npartitions"
    end

    fens, fes = T3block(A, A, N, N)
    geom = NodalField(fens.xyz)
    Temp = NodalField(zeros(count(fens), 1))
    
    
    femm = FEMMBase(IntegDomain(fes, TriRule(1)))
    C = dualconnectionmatrix(femm, fens, nodesperelem(boundaryfe(fes)))
    g = Metis.graph(C; check_hermitian=true)
    element_partitioning = Metis.partition(g, npartitions; alg=:KWAY)
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
    
    partition = nothing
    if rank > 0
        partnum = rank 
        @info "Creating partition $(partnum)"
        partition = PartitionSchurDD(make_partition_mesh(fens, fes, element_partitioning, partnum)..., Temp,
                    make_partition_fields, make_partition_femm, make_partition_matrix, make_partition_interior_load, make_partition_boundary_load
                )
    end

    if rank == 0
        @info "Building the right hand side"
    end
    b = gathersysvec(Temp, DOF_KIND_INTERFACE)
    b .= 0.0
    
    if rank > 0
        assemble_rhs!(b, partition)
    end
    MPI.Reduce!(b, MPI.SUM, comm; root=0)
    
    # if rank == 0
    #     @info "Building the preconditioning matrix"
    # end
    # K_ii = spzeros(size(b, 1), size(b, 1))
    # if rank > 0
    #     assemble_interface_matrix!(K_ii, partition)
    # end
    # MPI.Reduce!(K_ii, MPI.SUM, comm; root=0)

    if rank == 0
        @info "Solving the Linear System using CG"
    end  
    (T_i, stats) = pcg_mpi((q, p) -> mul_y_S_v!(q, partition, p), b, zeros(size(b)))
    @show stats
    

    MPI.Finalize()

    # @info "Solving the Linear System using CG"
    # # @time (T_i, stats) = cg(Sop, b)
    # # @time (T_i, stats) = cg(Sop, b; M=DiagonalPreconditioner(S_diag))
    # @time (T_i, stats) = cg(Sop, b; M=K_ii_factor)
    # @show stats
    # @info "Reconstructing value of free degrees of freedom"
    # scattersysvec!(Temp, T_i, DOF_KIND_INTERFACE)
    # for p in Sop.partitions
    #     reconstruct_free!(p)
    # end

    # @info "Exporting visualization"
    # approx_T = Temp.values
    # VTK.vtkexportmesh("approx.vtk", fes.conn, [geom.values approx_T], VTK.T3; scalars=[("Temperature", approx_T,)])

    # @info "Computing error"
    # Error = 0.0
    # for k = 1:size(fens.xyz, 1)
    #     Error = Error .+ abs.(approx_T[k] .- tempf(reshape(fens.xyz[k, :], (1, 2))))
    #     # Error = Error .+ abs.(Temp.values[k, 1] .- tempf(reshape(fens.xyz[k, :], (1, 2))))
    # end
    # println("Error =$Error")


    # File =  "a.vtk"
    # MeshExportModule.vtkexportmesh (File, fes.conn, [geom.values Temp.values], MeshExportModule.T3; scalars=Temp.values, scalars_name ="Temperature")

    # @test Error[1] < 1.e-5

    true
end
test()
nothing
end
