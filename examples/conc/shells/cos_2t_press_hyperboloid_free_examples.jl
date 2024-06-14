"""
Pressurized hyperboloid entirely free.

Example introduced in
Hiller, J.F. and K.J. Bathe, Measuring convergence of mixed finite element discretizations: 
an application to shell structures. Computers & Structures, 2003. 81(8-11): p. 639-654.
The applied pressure in the above paper needs to be 1 MPa for the energy values 
to correspond to the two tables.

See also: watch out, confusing mix up with magnitude of the modulus and applied pressure.
@article{Lee2004,
   author = {Lee, P. S. and Bathe, K. J.},
   title = {Development of MITC isotropic triangular shell finite elements},
   journal = {Computers & Structures},
   volume = {82},
   number = {11-12},
   pages = {945-962},
   ISSN = {0045-7949},
   DOI = {10.1016/j.compstruc.2004.02.004},
   year = {2004},
   type = {Journal Article}
}
"""
module cos_2t_press_hyperboloid_free_examples

using LinearAlgebra
using SparseArrays
using FinEtools
using FinEtools.MeshModificationModule: distortblock
using FinEtoolsDeforLinear
using FinEtoolsFlexStructures.FESetShellT3Module: FESetShellT3
using FinEtoolsFlexStructures.FESetShellQ4Module: FESetShellQ4
using FinEtoolsFlexStructures.FEMMShellT3FFModule
using FinEtoolsFlexStructures.RotUtilModule: initial_Rfield, update_rotation_field!
using FinEtools.MeshExportModule.VTKWrite: vtkwrite
using FinEtools.MeshExportModule: VTK
using FinEtoolsDDParallel
using FinEtoolsDDParallel.PartitionCoNCDDModule: element_overlap
using FinEtoolsDDParallel.CGModule: pcg_seq
using CoNCMOR: CoNCData, transfmatrix, LegendreBasis
using Metis

# Parameters:
E = 2.0e11
nu = 1/3;
pressure = 1.0e3;
Length = 2.0;

function computetrac!(forceout, XYZ, tangents, feid, qpid)
    r = vec(XYZ); r[2] = 0.0
    r .= vec(r)/norm(vec(r))
    theta = atan(r[3], r[1])
    n = cross(tangents[:, 1], tangents[:, 2]) 
    n = n/norm(n)
    forceout[1:3] = n*pressure*cos(2*theta)
    forceout[4:6] .= 0.0
    # @show dot(n, forceout[1:3])
    return forceout
end

function test_no_dd(n = 18, thickness = Length/2/100, visualize = false, distortion = 0.0)
    tolerance = Length/n/100
    fens, fes = distortblock(T3block, 90/360*2*pi, Length/2, n, n, distortion, distortion);
    fens.xyz = xyz3(fens)
    for i in 1:count(fens)
        a=fens.xyz[i, 1]; y=fens.xyz[i, 2];
        R = sqrt(1 + y^2)
        fens.xyz[i, :] .= (R*sin(a), y, R*cos(a))
    end

    mater = MatDeforElastIso(DeforModelRed3D, E, nu)
    
    sfes = FESetShellT3()
    accepttodelegate(fes, sfes)
    femm = FEMMShellT3FFModule.make(IntegDomain(fes, TriRule(1), thickness), mater)
    stiffness = FEMMShellT3FFModule.stiffness
    associategeometry! = FEMMShellT3FFModule.associategeometry!

    # Construct the requisite fields, geometry and displacement
    # Initialize configuration variables
    geom0 = NodalField(fens.xyz)
    u0 = NodalField(zeros(size(fens.xyz,1), 3))
    Rfield0 = initial_Rfield(fens)
    dchi = NodalField(zeros(size(fens.xyz,1), 6))

    # Apply EBC's
    # plane of symmetry perpendicular to Z
    l1 = selectnode(fens; box = Float64[-Inf Inf -Inf Inf 0 0], inflate = tolerance)
    for i in [3,4,5]
        setebc!(dchi, l1, true, i)
    end
    # plane of symmetry perpendicular to Y
    l1 = selectnode(fens; box = Float64[-Inf Inf 0 0 -Inf Inf], inflate = tolerance)
    for i in [2,4,6]
        setebc!(dchi, l1, true, i)
    end
    # plane of symmetry perpendicular to X
    l1 = selectnode(fens; box = Float64[0 0 -Inf Inf -Inf Inf], inflate = tolerance)
    for i in [1,5,6]
        setebc!(dchi, l1, true, i)
    end
    # clamped edge perpendicular to Y
    # l1 = selectnode(fens; box = Float64[-Inf Inf L/2 L/2 -Inf Inf], inflate = tolerance)
    # for i in [1,2,3,4,5,6]
    #     setebc!(dchi, l1, true, i)
    # end
    applyebc!(dchi)
    numberdofs!(dchi);

    # Assemble the system matrix
    associategeometry!(femm, geom0)
    K = stiffness(femm, geom0, u0, Rfield0, dchi);

    # Midpoint of the free edge
    # nl = selectnode(fens; box = Float64[R R L/2 L/2 -Inf Inf], inflate = tolerance)
    lfemm = FEMMBase(IntegDomain(fes, TriRule(3)))
    
    fi = ForceIntensity(Float64, 6, computetrac!);
    F = distribloads(lfemm, geom0, dchi, fi, 2);
    
    # Solve
    fr = dofrange(dchi, DOF_KIND_FREE)
    U = K[fr, fr]\F[fr]
    scattersysvec!(dchi, U[:])
    strainenergy = 1/2 * U' * K[fr, fr] * U
    @info "Strain Energy: $(round(strainenergy, digits = 9))"

    return strainenergy
end

function test(n = 60, thickness = Length/2/100, visualize = false, distortion = 0.0)
    npartitions = 3
    nbf1max = 3
    tolerance = Length/n/100
    fens, fes = distortblock(T3block, 90/360*2*pi, Length/2, n, n, distortion, distortion);
    fens.xyz = xyz3(fens)
    for i in 1:count(fens)
        a=fens.xyz[i, 1]; y=fens.xyz[i, 2];
        R = sqrt(1 + y^2)
        fens.xyz[i, :] .= (R*sin(a), y, R*cos(a))
    end

    mater = MatDeforElastIso(DeforModelRed3D, E, nu)
    
    sfes = FESetShellT3()
    accepttodelegate(fes, sfes)
    femm = FEMMShellT3FFModule.make(IntegDomain(fes, TriRule(1), thickness), mater)
    stiffness = FEMMShellT3FFModule.stiffness
    associategeometry! = FEMMShellT3FFModule.associategeometry!

    # Construct the requisite fields, geometry and displacement
    # Initialize configuration variables
    geom0 = NodalField(fens.xyz)
    u0 = NodalField(zeros(size(fens.xyz,1), 3))
    Rfield0 = initial_Rfield(fens)
    dchi = NodalField(zeros(size(fens.xyz,1), 6))

    # Apply EBC's
    # plane of symmetry perpendicular to Z
    l1 = selectnode(fens; box = Float64[-Inf Inf -Inf Inf 0 0], inflate = tolerance)
    for i in [3,4,5]
        setebc!(dchi, l1, true, i)
    end
    # plane of symmetry perpendicular to Y
    l1 = selectnode(fens; box = Float64[-Inf Inf 0 0 -Inf Inf], inflate = tolerance)
    for i in [2,4,6]
        setebc!(dchi, l1, true, i)
    end
    # plane of symmetry perpendicular to X
    l1 = selectnode(fens; box = Float64[0 0 -Inf Inf -Inf Inf], inflate = tolerance)
    for i in [1,5,6]
        setebc!(dchi, l1, true, i)
    end
    # clamped edge perpendicular to Y
    # l1 = selectnode(fens; box = Float64[-Inf Inf L/2 L/2 -Inf Inf], inflate = tolerance)
    # for i in [1,2,3,4,5,6]
    #     setebc!(dchi, l1, true, i)
    # end
    applyebc!(dchi)
    numberdofs!(dchi);

    # Assemble the system matrix
    associategeometry!(femm, geom0)
    K = stiffness(femm, geom0, u0, Rfield0, dchi);

    # Midpoint of the free edge
    # nl = selectnode(fens; box = Float64[R R L/2 L/2 -Inf Inf], inflate = tolerance)
    lfemm = FEMMBase(IntegDomain(fes, TriRule(3)))
    
    fi = ForceIntensity(Float64, 6, computetrac!);
    F = distribloads(lfemm, geom0, dchi, fi, 2);
    
    # Solve
    fr = dofrange(dchi, DOF_KIND_FREE)
    K_ff = K[fr, fr]
    F_f = F[fr]
    U = K_ff\F_f
    scattersysvec!(dchi, U[:])
    strainenergy = 1/2 * U' * K_ff * U
    @info "Strain Energy: $(round(strainenergy, digits = 9))"

    VTK.vtkexportmesh("cos_2t_press_hyperboloid_free-direct.vtk", fens, fes; 
    vectors=[("u", deepcopy(dchi.values[:, 1:3]),)])

    C = connectionmatrix(femm, count(fens))
    g = Metis.graph(C; check_hermitian=true)
    partitioning = Metis.partition(g, npartitions; alg=:KWAY)
    # partitioning = nodepartitioning(fens, npartitions)
    partitionnumbers = unique(partitioning)
    npartitions = length(partitionnumbers)

    element_overlaps = element_overlap(fes, partitioning)
    
    mor = CoNCData(fens, partitioning)
    Phi = transfmatrix(mor, LegendreBasis, nbf1max, dchi)
    Phi = Phi[fr, :] # trim the transformation to free degrees of freedom only
    PhiT = Phi'
    
    femm = FEMMShellT3FFModule.make(IntegDomain(fes, TriRule(1), thickness), mater)
    associategeometry!(femm, geom0)
    Kr_ff = spzeros(size(Phi, 2), size(Phi, 2))
    partitions = []
    for i in partitionnumbers
        pnl = findall(x -> x == i, partitioning)
        doflist =   Int[]
        for n in pnl
            for d in axes(dchi.dofnums, 2)
                if dchi.dofnums[n, d] in fr
                    push!(doflist, dchi.dofnums[n, d])
                end
            end
        end
        cel = connectedelems(fes, pnl, count(fens))
        pfes = subset(fes, cel)
        pelement_overlaps = element_overlaps[cel]
        pmaxoverlap = maximum(pelement_overlaps)
        pK = spzeros(nalldofs(dchi), nalldofs(dchi))
        for overlap in 1:pmaxoverlap
            l = findall(x -> x == overlap, pelement_overlaps)
            femm.integdomain.fes = subset(pfes, l)
            pK += (1 / overlap) .* stiffness(femm, geom0, u0, Rfield0, dchi);
        end
        pK_ff = pK[fr, fr]
        Kr_ff += Phi' * pK_ff * Phi
        pKfactor = lu(pK[doflist, doflist])
        part = (nodelist = pnl, pK_ff = pK_ff, factor = pKfactor, doflist = doflist)
        push!(partitions, part)
    end
    Krfactor = lu(Kr_ff)
    
    function Aop!(q, p)
        q .= 0.0
        for part in partitions
            q .+= (part.pK_ff * p)
        end
        q
    end

    function M!(q, p)
        q .= Phi * (Krfactor \ (PhiT * p))
        for part in partitions
            q[part.doflist] .+= (part.factor \ p[part.doflist])
        end
        q
    end

    (U_f, stats) = pcg_seq((q, p) -> Aop!(q, p), F_f, zeros(size(F_f));
        (M!)=(q, p) -> M!(q, p),
        itmax=1000, atol=1e-6, rtol=1e-6)
    @show stats
    scattersysvec!(dchi, U_f)
    strainenergy = 1/2 * U_f' * K_ff * U_f
    @info "Strain Energy: $(round(strainenergy, digits = 9))"

    VTK.vtkexportmesh("cos_2t_press_hyperboloid_free-dd.vtk", fens, fes; 
    vectors=[("u", deepcopy(dchi.values[:, 1:3]),)])


    return strainenergy
end

test()
nothing

end # module
nothing
