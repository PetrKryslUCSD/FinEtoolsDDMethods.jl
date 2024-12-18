"""
Heat conduction example described by Amuthan A. Ramabathiran
http://www.codeproject.com/Articles/579983/Finite-Element-programming-in-Julia:
Unit cube, with known temperature distribution along the boundary,
and uniform heat generation rate inside.

Solution with domain decomposition. Sequential execution.
Version: 08/19/2024
"""
module Poisson2D_seq_debug

using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsHeatDiff
using FinEtoolsDDMethods
using FinEtoolsDDMethods.CGModule: pcg
using FinEtoolsDDMethods.PartitionCoNCModule: CoNCPartitioningInfo, CoNCPartitionData, npartitions
using FinEtoolsDDMethods.DDCoNCSeqModule: DDCoNCSeqComm, PartitionedVector, aop!, TwoLevelPreConditioner, rhs
using FinEtoolsDDMethods.DDCoNCSeqModule:  vec_copyto!, vec_collect
using Metis
using Test
using LinearAlgebra
using SparseArrays
using CoNCMOR: CoNCData, transfmatrix, LegendreBasis
using DataDrop
using Statistics

function _execute_alt(filename, kind, mesher, volrule, N, Nc, n1, Np, No, itmax, relrestol, peek, visualize)
    # @info("""
    # Heat conduction example described by Amuthan A. Ramabathiran
    # http://www.codeproject.com/Articles/579983/Finite-Element-programming-in-Julia:
    # Unit cube, with known temperature distribution along the boundary,
    # and uniform heat generation rate inside.  Mesh of regular quadratic HEXAHEDRA,
    # in a grid of $(N) x $(N) x $(N) edges.
    # Version: 08/13/2024
    # """)
    A = 1.0 # dimension of the domain (length of the side of the square)
    thermal_conductivity = [i == j ? one(Float64) : zero(Float64) for i = 1:2, j = 1:2] # conductivity matrix
    Q = -6.0 # internal heat generation rate
    function getsource!(forceout, XYZ, tangents, feid, qpid)
        forceout[1] = Q #heat source
    end
    tempf(x) = (1.0 .+ x[:, 1] .^ 2 + 2.0 .* x[:, 2] .^ 2)#the exact distribution of temperature1
    t0 = time()
    fens, fes = mesher(A, A, N, N)
    @info("Mesh generation ($(round(time() - t0, digits=3)) [s])")
    t0 = time()
    geom = NodalField(fens.xyz)
    Temp = NodalField(zeros(size(fens.xyz, 1), 1))
    Tolerance = 1.0 / N / 100.0
    l1 = selectnode(fens; box = [0.0 0.0 0.0 A], inflate = Tolerance)
    l2 = selectnode(fens; box = [A   A   0.0 A], inflate = Tolerance)
    l3 = selectnode(fens; box = [0.0 A 0.0 0.0], inflate = Tolerance)
    l4 = selectnode(fens; box = [0.0 A  A  A], inflate = Tolerance)
    List = vcat(l1, l2, l3, l4, )
    setebc!(Temp, List, true, 1, tempf(geom.values[List, :])[:])
    numberdofs!(Temp)
    fr = dofrange(Temp, DOF_KIND_FREE)
    dr = dofrange(Temp, DOF_KIND_DATA)
    @info("Number of free degrees of freedom: $(nfreedofs(Temp)) ($(round(time() - t0, digits=3)) [s])")
    t0 = time()
    material = MatHeatDiff(thermal_conductivity)
    
    @info("Number of edges: $(N)")
    @info("Number of fine grid partitions: $(Np)")
    @info("Number of overlaps: $(No)")
    @info("Number of nodes: $(count(fens))")
    @info("Number of elements: $(count(fes))")
    @info("Number of free dofs = $(nfreedofs(Temp))")

    function make_matrix(fes)
        femm2 = FEMMHeatDiff(IntegDomain(fes, volrule, 1.0), material)
        return conductivity(femm2, geom, Temp)
    end
    function make_interior_load(fes)
        femm2 = FEMMHeatDiff(IntegDomain(fes, volrule, 1.0), material)
        fi = ForceIntensity(Float64[Q])
        return distribloads(femm2, geom, Temp, fi, 3)
    end

    t1 = time()
    cpi = CoNCPartitioningInfo(fens, fes, Np, No, Temp) 
    DataDrop.store_json("cpi.json", cpi) # DEBUG
    @info("Create partitioning info ($(round(time() - t1, digits=3)) [s])")
    t2 = time()
    ddcomm = DDCoNCSeqComm(nothing, cpi, fes, make_matrix, make_interior_load)
    @info("Make partitions ($(round(time() - t2, digits=3)) [s])")
    meanps = mean_partition_size(cpi)
    @info("Mean fine partition size: $(meanps)")
    @info("Create partitions ($(round(time() - t1, digits=3)) [s])")

    t1 = time()
    @info("Number of clusters (requested): $(Nc)")
    @info("Number of 1D basis functions: $(n1)")
    nt = n1*(n1+1)/2 
    (Nc == 0) && (Nc = Int(floor(meanps / nt / ndofs(Temp))))
    Nepc = count(fes) ÷ Nc
    (n1 > (Nepc/2)^(1/2)) && @error "Not enough elements per cluster"
    @info("Number of elements per cluster: $(Nepc)")
    cpartitioning, Nc = cluster_partitioning(fens, fes, fes.label, Nepc)
    @info("Number of clusters (actual): $(Nc)")
    @info "Create clusters ($(round(time() - t1, digits=3)) [s])"
        
    mor = CoNCData(fens, cpartitioning)
    Phi = transfmatrix(mor, LegendreBasis, n1, Temp)
    Phi = Phi[fr, :]
    @info("Size of the reduced problem: $(size(Phi, 2))")
    @info("Generate clusters ($(round(time() - t1, digits=3)) [s])")

    function peeksolution(iter, x, resnorm)
        peek && (@info("it $(iter): residual norm =  $(resnorm)"))
    end
    
    t1 = time()
    F_f = rhs(ddcomm)

    t1 = time()
    M! = TwoLevelPreConditioner(ddcomm, Phi)
    DataDrop.empty_hdf5_file("Kr_ff.h5") # DEBUG
    DataDrop.store_matrix("Kr_ff", M!.Kr_ff)
    @info("Create preconditioner ($(round(time() - t1, digits=3)) [s])")

    t0 = time()
    x0 = PartitionedVector(Float64, ddcomm)
    vec_copyto!(x0, 0.0)
    b = PartitionedVector(Float64, ddcomm)
    vec_copyto!(b, F_f)
    (sol, stats) = pcg(
        (q, p) -> aop!(q, p), 
        b, x0;
        (M!)=(q, p) -> M!(q, p),
        peeksolution=peeksolution,
        itmax=itmax, 
        atol= 0, rtol=relrestol, normtype = KSP_NORM_UNPRECONDITIONED
        )
    t1 = time()
    @info("Number of iterations:  $(stats.niter)")
    @info("Iterations ($(round(t1 - t0, digits=3)) [s])")
    stats = (niter = stats.niter, residuals = stats.residuals ./ norm(F_f))
    data = Dict(
        "number_nodes" => count(fens),
        "number_elements" => count(fes),
        "nfreedofs" => nfreedofs(Temp),
        "Nc" => Nc,
        "n1" => n1,
        "Np" => Np,
        "No" => No,
        "meanps" => meanps,
        "size_Kr_ff" => size(M!.Kr_ff_factor),
        "stats" => stats,
        "iteration_time" => t1 - t0,
    )
    f = (filename == "" ?
         "Poisson2D-" *
         "$(kind)-" *
         "-N=$(N)" *
         "-Nc=$(Nc)" *
         "-n1=$(n1)" *
         "-Np=$(Np)" *
         "-No=$(No)" :
         filename)
    @info("Storing data in $(f * ".json")")
    DataDrop.store_json(f * ".json", data)
    T_f = vec_collect(sol)
    scattersysvec!(Temp, T_f, DOF_KIND_FREE)
    
    if visualize
        f = (filename == "" ?
         "Poisson2D-" *
         "$(kind)-" *
         "-N=$(N)" *
         "-Nc=$(Nc)" *
         "-n1=$(n1)" *
         "-Np=$(Np)" *
         "-No=$(No)" :
         filename) * "-cg-sol"
        VTK.vtkexportmesh(f * ".vtk", fens, fes;
            scalars=[("T", deepcopy(Temp.values),)])
    end

    t0 = time()
    Error = 0.0
    for k in axes(fens.xyz, 1)
        Error = Error + abs.(Temp.values[k, 1] - tempf(reshape(fens.xyz[k, :], (1, 2)))[1])
    end
    @info("Error =$Error ($(round(time() - t0, digits=3)) [s])")

    true
end

using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--filename"
        help = "Use filename to name the output files"
        arg_type = String
        default = ""
        "--kind"
        help = "Kind of element (T3, T6, Q4, Q8)"
        arg_type = String
        default = "T3"
        "--Nc"
        help = "Number of clusters"
        arg_type = Int
        default = 2
        "--n1"
        help = "Number 1D basis functions"
        arg_type = Int
        default = 2
        "--No"
        help = "Number of overlaps"
        arg_type = Int
        default = 1
        "--Np"
        help = "Number of partitions"
        arg_type = Int
        default = 3
        "--N"
        help = "How many edges per side?"
        arg_type = Int
        default = 5
        "--itmax"
        help = "Maximum number of iterations allowed"
        arg_type = Int
        default = 200
        "--relrestol"
        help = "Relative residual tolerance"
        arg_type = Float64
        default = 1.0e-6
        "--peek"
        help = "Peek at the iterations?"
        arg_type = Bool
        default = true
        "--visualize"
        help = "Write out visualization files?"
        arg_type = Bool
        default = false
    end
    return parse_args(s)
end

p = parse_commandline()

kind = p["kind"]
if kind == "Q8"
    mesher = Q8block
    volrule = GaussRule(2, 3)
elseif kind == "Q4"
    mesher = Q4block
    volrule = GaussRule(2, 2)
elseif kind == "T6"
    mesher = T6block
    volrule = TriRule(3)
elseif kind == "T3"
    mesher = T3block
    volrule = TriRule(1)
else
    error("Unknown kind of element")
end

_execute_alt(
    p["filename"],
    kind,
    mesher,
    volrule,
    p["N"],
    p["Nc"], 
    p["n1"],
    p["Np"], 
    p["No"], 
    p["itmax"], 
    p["relrestol"],
    p["peek"],
    p["visualize"],
    )


nothing
end # module


