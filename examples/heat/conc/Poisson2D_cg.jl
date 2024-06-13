module Poisson2D_cg
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsHeatDiff
using FinEtoolsDDParallel
using FinEtoolsDDParallel.CGModule: pcg_seq
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
    
    npartitions = 80
    nbf1max = 6

    tempf(x) = (1.0 .+ x[:, 1] .^ 2 .+ 2 * x[:, 2] .^ 2)#the exact distribution of temperature
    N = 1000 # number of subdivisions along the sides of the square domain

    fens, fes = T3block(A, A, N, N)

    geom = NodalField(fens.xyz)
    Temp = NodalField(zeros(size(fens.xyz, 1), 1))

    material = MatHeatDiff(thermal_conductivity)

    femm = FEMMHeatDiff(IntegDomain(fes, TriRule(1), 100.0), material)

    l1 = selectnode(fens; box=[0.0 0.0 0.0 A], inflate=1.0 / N / 100.0)
    l2 = selectnode(fens; box=[A A 0.0 A], inflate=1.0 / N / 100.0)
    l3 = selectnode(fens; box=[0.0 A 0.0 0.0], inflate=1.0 / N / 100.0)
    l4 = selectnode(fens; box=[0.0 A A A], inflate=1.0 / N / 100.0)
    List = vcat(l1, l2, l3, l4)
    setebc!(Temp, List, true, 1, tempf(geom.values[List, :])[:])
    applyebc!(Temp)

    #  First the usual way
    # ============================================================
    numberdofs!(Temp)
    fr = dofrange(Temp, DOF_KIND_FREE)
    dr = dofrange(Temp, DOF_KIND_DATA)

    K = conductivity(femm, geom, Temp)

    fi = ForceIntensity(Float64[Q])
    F1 = distribloads(femm, geom, Temp, fi, 3)
    F1 = F1[fr]

    K_fd = K[fr, dr]
    T_d = gathersysvec(Temp, DOF_KIND_DATA)
    F2 = - K_fd * T_d
    
    partitioning = nodepartitioning(fens, npartitions)
    mor = CoNCData(fens, partitioning)
    Phi = transfmatrix(mor, LegendreBasis, nbf1max, Temp)
    Phi = Phi[fr, :]
    transfm(m, t, tT) = (tT * m * t)
    transfv(v, t, tT) = (tT * v)
    PhiT = Phi'
    K_ff = K[fr, fr]
    Kr_ff = transfm(K_ff, Phi, PhiT)
    @show size(Kr_ff)
    Krfactor = lu(Kr_ff)
    # Ur = Phi * (Krfactor \ (PhiT * F))

    partitions = []
    for i in 1:npartitions
        pnl = findall(x -> x == i, partitioning)
        doflist = [Temp.dofnums[n] for n in pnl if Temp.dofnums[n] <= nfreedofs(Temp)]
        pK = K[doflist, doflist]
        pKfactor = lu(pK)
        part = (nodelist = pnl, factor = pKfactor, doflist = doflist)
        push!(partitions, part)
    end

    function M!(q, p)
        q .= Phi * (Krfactor \ (PhiT * p))
        # q .= 0.0
        for part in partitions
            q[part.doflist] .+= part.factor \ p[part.doflist]
        end
        q
    end

    (T_f, stats) = pcg_seq((q, p) -> mul!(q, K_ff, p), F1 + F2, zeros(size(F1)); 
        M! = (q, p) -> M!(q, p), 
        itmax=1000, atol=1e-10, rtol=1e-10)
        @show stats
        scattersysvec!(Temp, T_f)

    # fr = dofrange(Temp, DOF_KIND_FREE)
    # dr = dofrange(Temp, DOF_KIND_DATA)
    # K_ff = K[fr, fr]
    # K_fd = K[fr, dr]
    # F_f = F1[fr]
    # T_d = gathersysvec(Temp, DOF_KIND_DATA)
    # @time T_f = K_ff \ (F_f - K_fd * T_d)
    # scattersysvec!(Temp, T_f)
    
    T = deepcopy(Temp.values)
    VTK.vtkexportmesh("ref.vtk", fes.conn, [geom.values T], VTK.T3; scalars=[("Temperature", T,)])


    Error = 0.0
    for k = 1:size(fens.xyz, 1)
        Error = Error .+ abs.(Temp.values[k, 1] .- tempf(reshape(fens.xyz[k, :], (1, 2))))
    end
    println("Error =$Error")

    true
end
test()
nothing
end
