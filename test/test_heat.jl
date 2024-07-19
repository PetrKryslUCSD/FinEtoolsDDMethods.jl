module mmmmPoiss_06122017
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsHeatDiff
using FinEtoolsDDMethods
using Metis
using Test
using LinearAlgebra
using SparseArrays
using PlotlyLight
function spy_matrix(A::SparseMatrixCSC, name="")
    I, J, V = findnz(A)
    p = PlotlyLight.Plot()
    p(x=J, y=I, mode="markers")
    p.layout.title = name
    p.layout.yaxis.title = "Row"
    p.layout.yaxis.range = [size(A, 1) + 1, 0]
    p.layout.xaxis.title = "Column"
    p.layout.xaxis.range = [0, size(A, 2) + 1]
    p.layout.xaxis.side = "top"
    p.layout.margin.pad = 10
    display(p)
end
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
    DOF_KIND_INTERFACE::KIND_INT = 3
    ndoms = 3

    tempf(x) = (1.0 .+ x[:, 1] .^ 2 .+ 2 * x[:, 2] .^ 2)#the exact distribution of temperature
    N = 1000# number of subdivisions along the sides of the square domain

    fens, fes = T3block(A, A, N, N)

    geom = NodalField(fens.xyz)
    Temp = NodalField(zeros(size(fens.xyz, 1), 1))

    material = MatHeatDiff(thermal_conductivity)

    femm = FEMMHeatDiff(IntegDomain(fes, TriRule(1), 100.0), material)
    C = dualconnectionmatrix(femm, fens, nodesperelem(boundaryfe(fes)))
    g = Metis.graph(C; check_hermitian=true)
    element_partitioning = Metis.partition(g, ndoms; alg=:KWAY)
    n2p = FENodeToPartitionMap(fens, fes, element_partitioning)

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

    K = conductivity(femm, geom, Temp)

    fi = ForceIntensity(Float64[Q])
    F1 = distribloads(femm, geom, Temp, fi, 3)

    fr = dofrange(Temp, DOF_KIND_FREE)
    dr = dofrange(Temp, DOF_KIND_DATA)
    K_ff = K[fr, fr]
    K_fd = K[fr, dr]
    F_f = F1[fr]
    T_d = gathersysvec(Temp, DOF_KIND_DATA)
    T_f = K_ff \ (F_f - K_fd * T_d)
    scattersysvec!(Temp, T_f)
    ref_T = deepcopy(Temp.values)

    VTK.vtkexportmesh("ref.vtk", fes.conn, [geom.values ref_T], VTK.T3; scalars=[("Temperature", ref_T,)])


    Error = 0.0
    for k = 1:size(fens.xyz, 1)
        Error = Error .+ abs.(Temp.values[k, 1] .- tempf(reshape(fens.xyz[k, :], (1, 2))))
    end
    println("Error =$Error")

    # Now the parallel way
    # ============================================================
    Temp.values .= 0.0
    for i in eachindex(n2p.map)
        if length(n2p.map[i]) > 1
            Temp.kind[i, :] .= DOF_KIND_INTERFACE
        end
    end
    setebc!(Temp, List, true, 1, tempf(geom.values[List, :])[:])
    applyebc!(Temp)

    numberdofs!(Temp, 1:count(fens), [DOF_KIND_FREE, DOF_KIND_INTERFACE, DOF_KIND_DATA])

    K = conductivity(femm, geom, Temp)

    fi = ForceIntensity(Float64[Q])
    F1 = distribloads(femm, geom, Temp, fi, 3)

    fr = dofrange(Temp, DOF_KIND_FREE)
    ir = dofrange(Temp, DOF_KIND_INTERFACE)
    dr = dofrange(Temp, DOF_KIND_DATA)
    K_ff = K[fr, fr]
    K_fi = K[fr, ir]
    K_fd = K[fr, dr]
    K_ii = K[ir, ir]
    K_if = K[ir, fr]
    K_id = K[ir, dr]
    F_f = F1[fr]
    F_i = F1[ir]
    T_f = gathersysvec(Temp, DOF_KIND_FREE)
    T_i = gathersysvec(Temp, DOF_KIND_INTERFACE)
    T_d = gathersysvec(Temp, DOF_KIND_DATA)

    # spy_matrix(K_ff, "K_ff")
    # spy_matrix(K_ii, "K_ii")

    approx_T = deepcopy(Temp.values)
    # for iter in 1:100
    #     T_f .= K_ff \ (F_f - K_fi * T_i - K_fd * T_d)
    #     T_i .= K_ii \ (F_i - K_if * T_f - K_id * T_d)
    #     scattersysvec!(Temp, T_f, DOF_KIND_FREE)
    #     scattersysvec!(Temp, T_i, DOF_KIND_INTERFACE)
    #     scattersysvec!(Temp, T_d, DOF_KIND_DATA)
    #     approx_T .= Temp.values
    #     @show norm(ref_T - approx_T) / norm(ref_T)
    #     # VTK.vtkexportmesh("approx-$(iter).vtk", fes.conn, [geom.values approx_T], VTK.T3; scalars=[("Temperature", approx_T,)])
    # end
    @show size(K_ii), size(K)
    S = K_ii - K_if * (K_ff \ Matrix(K_fi))
    
    T_i = S \ (F_i - K_id * T_d - K_if * (K_ff \ (F_f - K_fd * T_d)))
    T_f = K_ff \ (F_f - K_fd * T_d - K_fi * T_i)
    scattersysvec!(Temp, T_f, DOF_KIND_FREE)
    scattersysvec!(Temp, T_i, DOF_KIND_INTERFACE)
    scattersysvec!(Temp, T_d, DOF_KIND_DATA)
    approx_T .= Temp.values

    Error = 0.0
    for k = 1:size(fens.xyz, 1)
        Error = Error .+ abs.(approx_T[k] .- tempf(reshape(fens.xyz[k, :], (1, 2))))
        # Error = Error .+ abs.(Temp.values[k, 1] .- tempf(reshape(fens.xyz[k, :], (1, 2))))
    end
    Error /= count(fens)
    println("Error =$Error")


    # File =  "a.vtk"
    # MeshExportModule.vtkexportmesh (File, fes.conn, [geom.values Temp.values], MeshExportModule.T3; scalars=Temp.values, scalars_name ="Temperature")

    @test Error[1] < 1.e-5

    true
end
test()
nothing
end
