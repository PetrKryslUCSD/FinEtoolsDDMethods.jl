"""
Heat conduction example described by Amuthan A. Ramabathiran
http://www.codeproject.com/Articles/579983/Finite-Element-programming-in-Julia:
Unit cube, with known temperature distribution along the boundary,
and uniform heat generation rate inside.

Solution with domain decomposition. Sequential execution.
Version: 08/19/2024
"""
module fib_seq_driver

using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtools.MeshExportModule: CSV
using FinEtools.MeshTetrahedronModule: tetv
using FinEtoolsDeforLinear
using FinEtoolsDDMethods
using FinEtoolsDDMethods: mebibytes
using FinEtoolsDDMethods.CGModule: pcg_seq, vec_copyto!
using FinEtoolsDDMethods.PartitionCoNCModule: CoNCPartitioningInfo, npartitions, NONSHARED, EXTENDED
using FinEtoolsDDMethods.DDCoNCSeqModule: make_partitions, PartitionedVector, aop!, TwoLevelPreConditioner, vec_copyto!
using FinEtoolsDDMethods: set_up_timers
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

function rotate(fens)
    Q = [cos(pi/2) -sin(pi/2); sin(pi/2) cos(pi/2)]
    for i in 1:count(fens)
        fens.xyz[i, :] .=  Q * fens.xyz[i, :]
    end
    fens 
end

function translate(fens, d)
    for i in 1:count(fens)
        fens.xyz[i, :] .+= d
    end
    fens
end

function matrix_unit(a, R, nL, nH, nW, tolerance)
    meshes = Array{Tuple{FENodeSet,AbstractFESet},1}()
    fens, fes = Q4elliphole(R, R, a + R, a + R, nL, nH, nW)
    push!(meshes, (fens, fes))
    fens, fes = Q4elliphole(R, R, a + R, a + R, nL, nH, nW)
    fens = rotate(fens)
    push!(meshes, (fens, fes))
    fens, fes = Q4elliphole(R, R, a + R, a + R, nL, nH, nW)
    fens = rotate(fens)
    fens = rotate(fens)
    push!(meshes, (fens, fes))
    fens, fes = Q4elliphole(R, R, a + R, a + R, nL, nH, nW)
    fens = rotate(fens)
    fens = rotate(fens)
    fens = rotate(fens)
    push!(meshes, (fens, fes))
    fens, fesa = mergenmeshes(meshes, tolerance)
    fes = fesa[1]
    for i in 2:length(fesa)
        fes = cat(fes, fesa[i])
    end
    setlabel!(fes, 0)
    return fens, fes
end

function fiber_unit(R, nr, tolerance, label)
    fens1, fes1 = Q4circlen(R, nr)
    fens1.xyz = fens1.xyz[:, 1:2]
    meshes = Array{Tuple{FENodeSet,AbstractFESet},1}()
    _fens, _fes = deepcopy(fens1), deepcopy(fes1)
    push!(meshes, (_fens, _fes))
    _fens, _fes = deepcopy(fens1), deepcopy(fes1)
    _fens = rotate(_fens)
    push!(meshes, (_fens, _fes))
    _fens, _fes = deepcopy(fens1), deepcopy(fes1)
    _fens = rotate(_fens)
    _fens = rotate(_fens)
    push!(meshes, (_fens, _fes))
    _fens, _fes = deepcopy(fens1), deepcopy(fes1)
    _fens = rotate(_fens)
    _fens = rotate(_fens)
    _fens = rotate(_fens)
    push!(meshes, (_fens, _fes))
    fens, fesa = mergenmeshes(meshes, tolerance)
    fes = fesa[1]
    for i in 2:length(fesa)
        fes = cat(fes, fesa[i])
    end
    setlabel!(fes, label)
    return fens, fes
end

function fibers_mesh_hex(ref = 1)
    nr = 3
    a = 1.0
    R = 1.0 
    d = a + R + (nr - 1) * 2 * (a + R) + a + R
    h = 2 * R / (2 + ref)
    nR = 2 * Int(round(R / h))
    nL = Int(round(nR / 2))
    nH = nL
    nW = 2 * Int(round(nR/2))
    Lz = 2.0 * d
    nlayers = ref + Int(ceil(Lz / (3 * h)))
    tolerance = R / nR / 5

    meshes = Array{Tuple{FENodeSet,AbstractFESet},1}()
    fens, fes = matrix_unit(a, R, nL, nH, nW, tolerance)
    push!(meshes, (fens, fes))
    fens, fes = fiber_unit(R, nR, tolerance, -1)
    push!(meshes, (fens, fes))
    fens, fesa = mergenmeshes(meshes, tolerance)
    fes = fesa[1]
    for i in 2:length(fesa)
        fes = cat(fes, fesa[i])
    end
    fens = translate(fens, [(a + R), (a + R)])
    
    fens1, fes1 = deepcopy(fens), deepcopy(fes)
    fel = selectelem(fens1, fes1, label = -1)
    labels = fes.label
    
    meshes = Array{Tuple{FENodeSet,AbstractFESet},1}()
    for r in 1:nr
        for c in 1:nr
            fens, fes = deepcopy(fens1), deepcopy(fes1)
            fens = translate(fens, [(c - 1) * (a + R) * 2, (r - 1) * (a + R) * 2])
            labels[fel] .= (r - 1) * nr + c
            setlabel!(fes, deepcopy(labels))
            push!(meshes, (fens, fes))
        end
    end

    fens, fesa = mergenmeshes(meshes, tolerance)
    fes = fesa[1]
    for i in 2:length(fesa)
        fes = cat(fes, fesa[i])
    end

    fens, fes = H8extrudeQ4(
        fens,
        fes,
        nlayers,
        (X, layer) -> [X[1], X[2], layer * Lz / nlayers],
    )

    # File =  "fibers_soft_hard_hex-mesh.vtk"
    # vtkexportmesh(File, fens, fes; scalars=[("label", fes.label)])
    # @async run(`"paraview.exe" $File`)

    return fens, fes
end # fibers_mesh

function fibers_mesh_tet(ref = 1)
    nr = 3
    a = 1.0
    R = 1.0 
    h = 2 * R / (2 + ref)
    d = a + R + (nr - 1) * 2 * (a + R) + a + R
    Lz = 2.0 * d
    nlayers = ref + Int(ceil(Lz / (5 * h)))
    input = """
    curve 1 line 0 0 $(d) 0
    curve 2 line $(d) 0 $(d) $(d)
    curve 3 line $(d) $(d) 0 $(d)  
    curve 4 line 0 $(d) 0 0 
    """ 
    p = 1
    for r in 1:nr
        for c in 1:nr 
            cx = a + R + (r - 1) * 2 * (a + R)
            cy = a + R + (c - 1) * 2 * (a + R)
            input *= "\n" * "curve $(4+p) circle center $(cx) $(cy) radius $R"
            p += 1
        end
    end
    input *= "\n" * 
    """
    subregion 1  property 1 boundary 1 2 3 4 hole """
    p = 1
    for r in 1:nr
        for c in 1:nr
            input *= " $(-(4+p))"
            p += 1
        end
    end
    input *= "\n" 
    p = 1
    for r in 1:nr
        for c in 1:nr
            input *= "subregion $(p+1) property 2 boundary  $(+(4+p))\n"
            p += 1
        end
    end
    input *= "\n" * 
    """    
    m-ctl-point constant $h
    """
    
    mesh = triangulate(input)
    
    fens = FENodeSet(mesh.xy)
    fes = FESetT3(mesh.triconn)
    labels = fill(0, count(fes)) # the matrix should have a label of 0
    for i in eachindex(mesh.trigroups)
        for j in mesh.trigroups[i]
            labels[j] = i - 1 # the fibers should have labels 1, 2, 3, ...
        end
    end
    setlabel!(fes, labels)
    
    fens, fes = T4extrudeT3(
        fens,
        fes,
        nlayers,
        (X, layer) -> [X[1], X[2], layer * Lz / nlayers],
    )
    
    # X = zeros(4, 3)
    # for i in eachindex(fes)
    #     X[1, :] = fens.xyz[fes.conn[i][1], :]
    #     X[2, :] = fens.xyz[fes.conn[i][2], :]
    #     X[3, :] = fens.xyz[fes.conn[i][3], :]
    #     X[4, :] = fens.xyz[fes.conn[i][4], :]
    #     @test  tetv(X) > 0.0
    # end

    fens, fes = T4toT10(fens, fes)
    
    # File =  "fibers_mesh.vtk"
    # vtkexportmesh(File, fens, fes; scalars=[("label", fes.label)])
    # @async run(`"paraview.exe" $File`)

    return fens, fes
end # fibers_mesh

function _execute(filename, kind, ref, Em, num, Ef, nuf,
    Nc, n1, Np, No,
    mesh, boundary_rule, interior_rule, make_femm, itmax, relrestol, peek, visualize)
    comm = 0
    CTE = 0.0
    magn = 1.0
    
    function getfrcL!(forceout, XYZ, tangents, feid, qpid)
        copyto!(forceout, [0.0; magn; 0.0])
    end
    fens, fes = mesh(ref)

    if visualize
        f = (filename == "" ? 
            "fib-$(string(kind))" *
            "-ref=$(ref)" : 
            filename) * "-mesh"
        vtkexportmesh(f * ".vtk", fens, fes; scalars=[("label", fes.label)])
    end
    
    MR = DeforModelRed3D
    matf = MatDeforElastIso(MR, 0.0, Ef, nuf, CTE)
    matm = MatDeforElastIso(MR, 0.0, Em, num, CTE)

    # Material orientation matrix
    csmat = [i == j ? one(Float64) : zero(Float64) for i = 1:3, j = 1:3]

    function updatecs!(csmatout, XYZ, tangents, feid, qpid)
        copyto!(csmatout, csmat)
    end

    geom = NodalField(fens.xyz)
    u = NodalField(zeros(size(fens.xyz, 1), 3)) # displacement field

    # Renumber the nodes
    femm = FEMMBase(IntegDomain(fes, interior_rule))
    C = connectionmatrix(femm, count(fens))
    perm = symrcm(C)
    
    matrixel = selectelem(fens, fes, label = 0)
    fiberel = setdiff(1:count(fes), matrixel)
    bfes = meshboundary(subset(fes, fiberel))
    sectionL = selectelem(fens, bfes; facing = true, direction = [0.0 0.0 +1.0])
    loadfes = subset(bfes, sectionL)
    bfes = meshboundary(fes)
    section0 = selectelem(fens, bfes; facing = true, direction = [0.0 0.0 -1.0])

    lx0 = connectednodes(subset(bfes, section0))
    setebc!(u, lx0, true, 1, 0.0)
    setebc!(u, lx0, true, 2, 0.0)
    setebc!(u, lx0, true, 3, 0.0)
    applyebc!(u)
    numberdofs!(u, perm)
    fr = dofrange(u, DOF_KIND_FREE)
    dr = dofrange(u, DOF_KIND_DATA)
    
    @info("Kind: $(string(kind))")
    @info("Refinement factor: $(ref)")
    @info("Materials: $(Ef), $(nuf), $(Em), $(num)")
    
    @info("Number of fine grid partitions: $(Np)")
    @info("Number of overlaps: $(No)")
    @info("Number of elements: $(count(fes))")
    @info("Number of free dofs = $(nfreedofs(u))")

    fi = ForceIntensity(Float64, 3, getfrcL!)
    el2femm = FEMMBase(IntegDomain(loadfes, boundary_rule))
    F = distribloads(el2femm, geom, u, fi, 2)
    F_f = F[fr]

    function make_matrix(fes)
        _matrixel = selectelem(fens, fes, label = 0)
        _fiberel = setdiff(1:count(fes), _matrixel)
        _femmm = make_femm(MR, IntegDomain(subset(fes, _matrixel), interior_rule), matm)
        associategeometry!(_femmm, geom)
        _femmf = make_femm(MR, IntegDomain(subset(fes, _fiberel), interior_rule), matf)
        associategeometry!(_femmf, geom)
        return stiffness(_femmm, geom, u) + stiffness(_femmf, geom, u)
    end

    @info("Refinement level: $(ref)")
    @info("Number of fine grid partitions: $(Np)")
    @info("Number of overlaps: $(No)")
    @info("Number of nodes: $(count(fens))")
    @info("Number of elements: $(count(fes))")
    @info("Number of free dofs = $(nfreedofs(u))")

    t1 = time()
    cpi = CoNCPartitioningInfo(fens, fes, Np, No, u) 
    @info "Create partitioning info ($(round(time() - t1, digits=3)) [s])"
    t2 = time()
    partition_list  = make_partitions(cpi, fes, make_matrix, nothing)
    @info "Make partitions ($(round(time() - t2, digits=3)) [s])"
    partition_sizes = [partition_size(_p) for _p in partition_list]
    meanps = mean(partition_sizes)
    @info "Mean fine partition size: $(meanps)"
    @info "Create partitions ($(round(time() - t1, digits=3)) [s])"

    t1 = time()
    @info("Number of clusters (requested): $(Nc)")
    @info("Number of 1D basis functions: $(n1)")
    nt = n1*(n1+1)/2 
    (Nc == 0) && (Nc = Int(floor(meanps / nt / ndofs(u))))
    Nepc = count(fes) ÷ Nc
    (n1 > (Nepc/2)^(1/2)) && @error "Not enough elements per cluster"
    @info("Number of elements per cluster: $(Nepc)")
    cpartitioning, Nc = cluster_partitioning(fens, fes, fes.label, Nepc)
    @info("Number of clusters (actual): $(Nc)")
    @info "Create clusters ($(round(time() - t1, digits=3)) [s])"
        
    mor = CoNCData(fens, cpartitioning)
    Phi = transfmatrix(mor, LegendreBasis, n1, u)
    Phi = Phi[fr, :]
    @info("Size of the reduced problem: $(size(Phi, 2))")
    @info "Generate clusters ($(round(time() - t1, digits=3)) [s])"

    function peeksolution(iter, x, resnorm)
        peek && (@info("it $(iter): residual norm =  $(resnorm)"))
    end
    
    t1 = time()
    M! = TwoLevelPreConditioner(partition_list, Phi, comm)
    @info "Create preconditioner ($(round(time() - t1, digits=3)) [s])"

    t0 = time()
    x0 = PartitionedVector(Float64, partition_list, comm)
    vec_copyto!(x0, 0.0)
    b = PartitionedVector(Float64, partition_list, comm)
    vec_copyto!(b, F_f)
    (T, stats) = pcg_seq(
        (q, p) -> aop!(q, p), 
        b, x0;
        (M!)=(q, p) -> M!(q, p),
        peeksolution=peeksolution,
        itmax=itmax, 
        atol= 0, rtol=relrestol, normtype = KSP_NORM_UNPRECONDITIONED
        )
    t1 = time()
    @info("Number of iterations:  $(stats.niter)")
    @info "Iterations ($(round(t1 - t0, digits=3)) [s])"
    stats = (niter = stats.niter, residuals = stats.residuals ./ norm(F_f))
    data = Dict(
        "number_nodes" => count(fens),
        "number_elements" => count(fes),
        "nfreedofs" => nfreedofs(u),
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
         "fib-" *
         "$(kind)-" *
         "-ref=$(ref)" *
         "-Nc=$(Nc)" *
         "-n1=$(n1)" *
         "-Np=$(Np)" *
         "-No=$(No)" :
         filename)
    @info "Storing data in $(f * ".json")"
    DataDrop.store_json(f * ".json", data)
    T_f = deepcopy(F_f); T_f .= 0.0
    scattersysvec!(u, vec_copyto!(T_f, T), DOF_KIND_FREE)
    
    if visualize
        f = (filename == "" ?
         "fib-" *
         "$(kind)-" *
         "-ref=$(ref)" *
         "-Nc=$(Nc)" *
         "-n1=$(n1)" *
         "-Np=$(Np)" *
         "-No=$(No)" :
         filename) * "-cg-sol"
        VTK.vtkexportmesh(f * ".vtk", fens, fes;
            vectors=[("u", deepcopy(u.values),)])
    end

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
        "--Nc"
        help = "Number of clusters"
        arg_type = Int
        default = 2
        "--n1"
        help = "Number 1D basis functions"
        arg_type = Int
        default = 5
        "--Np"
        help = "Number fine grid partitions"
        arg_type = Int
        default = 2
        "--No"
        help = "Number of overlaps"
        arg_type = Int
        default = 1
        "--N"
        help = "Number of element edges in one direction"
        arg_type = Int
        default = 2
        "--ref"
        help = "Refinement factor"
        arg_type = Int
        default = 2
        "--itmax"
        help = "Maximum number of iterations allowed"
        arg_type = Int
        default = 2000
        "--relrestol"
        help = "Relative residual tolerance"
        arg_type = Float64
        default = 1.0e-6
        "--kind"
        help = "hex or tet"
        arg_type = String
        default = "hex"
        "--Ef"
        help = "Young's modulus of the fibres"
        arg_type = Float64
        default = 1.00e5
        "--nuf"
        help = "Poisson ratio of the fibres"
        arg_type = Float64
        default = 0.3
        "--Em"
        help = "Young's modulus of the matrix"
        arg_type = Float64
        default = 1.00e3
        "--num"
        help = "Poisson ratio of the matrix"
        arg_type = Float64
        default = 0.4999
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
mesh, boundary_rule, interior_rule, make_femm = if kind == "hex"
    mesh = fibers_mesh_hex
    boundary_rule = GaussRule(2, 2)
    interior_rule = GaussRule(3, 2)
    make_femm = FEMMDeforLinearMSH8
    (mesh, boundary_rule, interior_rule, make_femm)
else
    mesh = fibers_mesh_tet
    boundary_rule = TriRule(6)
    interior_rule = TetRule(4)
    make_femm = FEMMDeforLinearMST10
    (mesh, boundary_rule, interior_rule, make_femm)
end

_execute(
    p["filename"],
    p["kind"], 
    p["ref"], 
    p["Em"], p["num"], p["Ef"], p["nuf"],
    p["Nc"], p["n1"], p["Np"], p["No"], 
    mesh, boundary_rule, interior_rule, make_femm, 
    p["itmax"], p["relrestol"],
    p["peek"],
    p["visualize"])

nothing
end # module


