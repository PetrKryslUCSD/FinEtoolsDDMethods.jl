module fib_seq_examples

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

function _execute(prefix, kind, ref, Em, num, Ef, nuf,
    Nc, n1, Np, No,
    mesh, boundary_rule, interior_rule, make_femm, itmax, relrestol, visualize)
    CTE = 0.0
    magn = 1.0
    
    function getfrcL!(forceout, XYZ, tangents, feid, qpid)
        copyto!(forceout, [0.0; magn; 0.0])
    end
    fens, fes = mesh(ref)

    if visualize
        f = "fib-$(prefix)-$(string(kind))" *
                "-rf=$(ref)" *
                "-mesh"
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
    n1adj = n1 + 1 # adjust for a safety margin
    ntadj = 0.97 * (n1*(n1+1)*(n1+2)/6) + 0.03 * (n1adj*(n1adj+1)*(n1adj+2)/6)
    (Nc == 0) && (Nc = Int(floor(minimum(partition_sizes) / ntadj / ndofs(u))))
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
        "rf" => ref, 
        "Nc" => Nc,
        "n1" => n1,
        "Np" => Np,
        "No" => No,
        "size_Kr_ff" => size(Krfactor),
        "stats" => stats,
        "time" => t1 - t0,
    )
    f = (prefix != "" ? "$(prefix)-" : "") * "fib-" *
        "-rf=$(ref)" *
        "-Nc=$(Nc)" *
        "-n1=$(n1)"  * 
        "-Np=$(Np)"  * 
        "-No=$(No)"  
    DataDrop.store_json(f * ".json", data)
    scattersysvec!(u, u_f)

    if visualize
        f = "fib-$(string(kind))" *
            "-ref=$(ref)" *
            "-Nc=$(Nc)" *
            "-n1=$(n1)" *
            "-Np=$(Np)" *
            "-cg-sol"
        VTK.vtkexportmesh(f * ".vtk", fens, fes; vectors=[("u", deepcopy(u.values),)])
    end
    
    true
end

function test(; prefix = "soft_hard",
    kind = "hex", ref = 1, 
    Em = 1.0e3, num = 0.4999, Ef = 1.0e5, nuf = 0.3, 
    Nc = 0, n1 = 6, Np = 4, No = 1, 
    itmax = 2000, relrestol = 1e-6, visualize = false)
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
    _execute(prefix, kind, ref, Em, num, Ef, nuf,
        Nc, n1, Np, No,
        mesh, boundary_rule, interior_rule, make_femm, itmax, relrestol, visualize)
end

nothing
end # module


