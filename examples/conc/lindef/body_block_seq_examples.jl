module body_block_seq_examples
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtools.MeshExportModule: CSV
using FinEtools.MeshTetrahedronModule: tetv
using FinEtoolsDeforLinear
using FinEtoolsDDMethods
using FinEtoolsDDMethods: mebibytes
using FinEtoolsDDMethods.CGModule: pcg_seq
using FinEtoolsDDMethods.PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData, npartitions, partition_size 
using FinEtoolsDDMethods.DDCoNCSeqModule: partition_multiply!, preconditioner!, make_partitions
using SymRCM
using Metis
using Test
using LinearAlgebra
using SparseArrays
using PlotlyLight
using Krylov
using LinearOperators
import Base: size, eltype
import LinearAlgebra: mul!
import CoNCMOR: CoNCData, transfmatrix, LegendreBasis
using Targe2
using DataDrop
using Statistics

function _execute(filename, kind, N, E, nu, 
    Nc, n1, Np, No, 
    mesh, boundary_rule, interior_rule, make_femm,
    itmax, relrestol, 
    visualize)
    CTE = 0.0
    L = 1.0
    
    fens, fes = mesh(L, L, L, 6*N, 11*N, 18*N)

    if visualize
        f = (filename == "" ? 
            "block-$(string(kind))" *
            "-N=$(N)" : 
            filename) * "-mesh"
        vtkexportmesh(f * ".vtk", fens, fes; scalars=[("label", fes.label)])
    end
    
    MR = DeforModelRed3D
    matf = MatDeforElastIso(MR, 0.0, E, nu, CTE)
    
    geom = NodalField(fens.xyz)
    u = NodalField(zeros(size(fens.xyz, 1), 3)) # displacement field

    # Renumber the nodes
    femm = FEMMBase(IntegDomain(fes, interior_rule))
    C = connectionmatrix(femm, count(fens))
    perm = symrcm(C)
    
    bfes = meshboundary(fes)
    lx0 = selectelem(fens, bfes, facing = true, direction = [-1.0, 0.0, 0.0])
    setebc!(u, connectednodes(subset(bfes, lx0)), true, 1, 0.0)
    lx0 = selectelem(fens, bfes, facing = true, direction = [+1.0, 0.0, 0.0])
    setebc!(u, connectednodes(subset(bfes, lx0)), true, 1, 0.0)
    lx0 = selectelem(fens, bfes, facing = true, direction = [0.0, -1.0, 0.0])
    setebc!(u, connectednodes(subset(bfes, lx0)), true, 2, 0.0)
    lx0 = selectelem(fens, bfes, facing = true, direction = [0.0, +1.0, 0.0])
    setebc!(u, connectednodes(subset(bfes, lx0)), true, 2, 0.0)
    lx0 = selectelem(fens, bfes, facing = true, direction = [0.0, 0.0, -1.0])
    setebc!(u, connectednodes(subset(bfes, lx0)), true, 3, 0.0)
    lx0 = selectelem(fens, bfes, facing = true, direction = [0.0, 0.0, +1.0])
    setebc!(u, connectednodes(subset(bfes, lx0)), true, 3, 0.0)
    applyebc!(u)
    numberdofs!(u, perm)
    fr = dofrange(u, DOF_KIND_FREE)
    dr = dofrange(u, DOF_KIND_DATA)
    
    @info("Kind: $(string(kind))")
    @info("Number of edges: $(N)")
    @info("Materials: $(E), $(nu)")
    
    @info("Number of fine grid partitions: $(Np)")
    @info("Number of overlaps: $(No)")
    @info("Number of elements: $(count(fes))")
    @info("Number of free dofs = $(nfreedofs(u))")

    getforce!(forceout, XYZ, tangents, feid, qpid) = 
        (forceout .= [sin(2*pi*XYZ[2]/L), sin(2*pi*XYZ[3]/L), sin(2*pi*XYZ[1]/L)])
    fi = ForceIntensity(Float64, 3, getforce!)
    femm = FEMMBase(IntegDomain(fes, interior_rule))
    F = distribloads(femm, geom, u, fi, 3)
    F_f = F[fr]

    function make_matrix(_fes)
        _femmf = make_femm(MR, IntegDomain(_fes, interior_rule), matf)
        associategeometry!(_femmf, geom)
        return stiffness(_femmf, geom, u)
    end

    t1 = time()
    cpi = CoNCPartitioningInfo(fens, fes, Np, No, u) 
    partition_list = make_partitions(cpi, fes, make_matrix, nothing)
    partition_sizes = [partition_size(_p) for _p in partition_list]
    meanps = mean(partition_sizes)
    @info "Mean fine partition size = $(meanps)"
    _b = mean([mebibytes(_p) for _p in partition_list])
    @info "Mean partition allocations: $(Int(round(_b, digits=0))) [MiB]" 
    @info "Total partition allocations: $(sum([mebibytes(_p) for _p in partition_list])) [MiB]" 
    @info "Create partitions ($(round(time() - t1, digits=3)) [s])"

    t1 = time()
    @info("Number of 1D basis functions: $(n1)")
    @info("Number of clusters (requested): $(Nc)")
    nt = (n1*(n1+1)*(n1+2)/6)
    (Nc == 0) && (Nc = Int(floor(meanps / nt / ndofs(u))))
    Nepc = count(fes) ÷ Nc
    (n1 > Nepc^(1/3)) && @error "Not enough elements per cluster"
    cpartitioning, Nc = cluster_partitioning(fens, fes, fes.label, Nepc)
    mor = CoNCData(fens, cpartitioning)
    Phi = transfmatrix(mor, LegendreBasis, n1, u)
    Phi = Phi[fr, :]
    @info("Number of clusters (actual): $(Nc)")
    @info("Size of the reduced problem: $(size(Phi, 2))")
    @info("Transformation matrix: $(mebibytes(Phi)) [MiB]")
    @info "Generate clusters ($(round(time() - t1, digits=3)) [s])"

    t1 = time()
    Kr_ff = spzeros(size(Phi, 2), size(Phi, 2))
    for partition in partition_list
        Kr_ff += (Phi' * partition.nonoverlapping_K * Phi)
    end
    Krfactor = lu(Kr_ff)
    Kr_ff = nothing
    GC.gc()
    @info "Create global factor ($(round(time() - t1, digits=3)) [s])"
    @info("Global reduced factor: $(mebibytes(Krfactor)) [MiB]")

    
    function peeksolution(iter, x, resnorm)
        @info("it $(iter): residual norm =  $(resnorm)")
    end
    
    t0 = time()
    M! = preconditioner!(Krfactor, Phi, partition_list)
    (u_f, stats) = pcg_seq(
        (q, p) -> partition_multiply!(q, partition_list, p), 
        F_f, zeros(size(F_f));
        (M!)=(q, p) -> M!(q, p),
        # peeksolution=peeksolution,
        itmax=itmax, 
        # atol=0, rtol=relrestol, normtype = KSP_NORM_NATURAL
        # atol=relrestol * norm(F_f), rtol=0, normtype = KSP_NORM_NATURAL
        atol= 0, rtol=relrestol, normtype = KSP_NORM_UNPRECONDITIONED
        )
    t1 = time()
    @info("Number of iterations:  $(stats.niter)")
    @info "Iterations ($(round(t1 - t0, digits=3)) [s])"
    stats = (niter = stats.niter, residuals = stats.residuals ./ norm(F_f))
    data = Dict(
        "nfreedofs" => nfreedofs(u),
        "Nc" => Nc,
        "n1" => n1,
        "Np" => Np,
        "No" => No,
        "size_Kr_ff" => size(Krfactor),
        "stats" => stats,
        "time" => t1 - t0,
    )
    f = (filename == "" ?
         "body_block-$(string(kind))" *
         "-N=$(N)" *
         "-Nc=$(Nc)" *
         "-n1=$(n1)" *
         "-Np=$(Np)" *
         "-No=$(No)" :
         filename)
    DataDrop.store_json(f * ".json", data)
    scattersysvec!(u, u_f)

    if visualize
        f = (filename == "" ?
         "body_block-$(string(kind))" *
         "-N=$(N)" *
         "-Nc=$(Nc)" *
         "-n1=$(n1)" *
         "-Np=$(Np)" *
         "-No=$(No)" :
         filename) * "-cg-sol"
        VTK.vtkexportmesh(f * ".vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])
    end
    
    true
end
# test(ref = 1)

function test(; filename = "", kind = "hex", N = 4, E = 1.0e5, nu = 0.3, Nc = 2, n1 = 5, Np = 8, No = 1, itmax = 2000, relrestol = 1e-6, visualize = false)
    mesh, boundary_rule, interior_rule, make_femm = if kind == "hex"
        mesh = H8block
        boundary_rule = GaussRule(2, 2)
        interior_rule = GaussRule(3, 2)
        make_femm = FEMMDeforLinearMSH8
        (mesh, boundary_rule, interior_rule, make_femm)
    else
        mesh = T10block
        boundary_rule = TriRule(6)
        interior_rule = TetRule(4)
        make_femm = FEMMDeforLinearMST10
        (mesh, boundary_rule, interior_rule, make_femm)
    end
    _execute(filename, kind, N, E, nu,
        Nc, n1, Np, No,
        mesh, boundary_rule, interior_rule, make_femm,
        itmax, relrestol,
        visualize)
end

nothing
end # module


