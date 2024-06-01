module Poisson2D_cg
using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsHeatDiff
using FinEtoolsDDParallel
using Metis
using Test
using LinearAlgebra
using SparseArrays
using PlotlyLight
using Krylov
using LinearOperators
import Base: size, eltype
import LinearAlgebra: mul!
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
struct SLinearOperator{T}
    tempf::Vector{T}
    tempi::Vector{T}
    K_ii::SparseMatrixCSC{T}
    K_fi::SparseMatrixCSC{T}
    K_if::SparseMatrixCSC{T}
    K_ff_factor
    function SLinearOperator(K_ii::SparseMatrixCSC{T}, K_fi::SparseMatrixCSC{T}, K_if::SparseMatrixCSC{T}, K_ff::SparseMatrixCSC{T}) where T
        new{T}(zeros(T, size(K_ff, 1)), zeros(T, size(K_ii, 1)), K_ii, K_fi, K_if, cholesky(K_ff))
    end
end
function mul!(y, Sop::SLinearOperator{T}, v) where {T}
    mul!(Sop.tempf, Sop.K_fi, v)
    mul!(Sop.tempi, Sop.K_if, (Sop.K_ff_factor \ Sop.tempf))
    mul!(y, Sop.K_ii, v) 
    y .-= Sop.tempi
    y
end
size(Sop::SLinearOperator) = size(Sop.K_ii)
eltype(Sop::SLinearOperator) = eltype(Sop.K_ii)
function mul!(y::Vector{Float64}, F::SparseArrays.CHOLMOD.Factor{Float64, Int64}, v::Vector{Float64})
    y .= F \ v
end

function _cg(A, b, x0, maxiter)
    x = deepcopy(x0)
    g = A * x - b
    d = similar(x)
    Ad = similar(x)
    @. d = -g
    for iter in 1:maxiter
        mul!(Ad, A, d)
        rho = dot(d, Ad)
        alpha = -dot(g, d) / rho
        @. x += alpha * d
        mul!(g, A, x)
        @. g -= b
        beta = dot(g, Ad) / rho
        @. d *= beta
        @. d -= g
    end
    return x
end

function _cg_smith(A, b, x0, maxiter)
    x = deepcopy(x0)
    p = b - A * x
    r = deepcopy(p)
    # rn = deepcopy(p)
    q = similar(p)
    for iter in 1:maxiter
        mul!(q, A, p)
        rho = dot(r, r)
        alpha = rho / dot(p, q)
        @. x += alpha * p
        # @. rn = r - alpha * q
        @. r -= alpha * q
        # beta = dot(rn, rn) / rho
        beta = dot(r, r) / rho
        @. p = r + beta * p
        # r, rn = rn, r
    end
    return x
end

struct SPreConditioner{MATRIX, T, FACTOR}
    S::MATRIX
    temp::Vector{T}
    factor::FACTOR
end
function mul!(y, Pre::PTYPE, v) where {PTYPE<:SPreConditioner}
    mul!(Pre.temp, Pre.S, v)
    # y .= Pre.factor \ Pre.temp
    ldiv!(y, Pre.factor, Pre.temp) # lu factor supports this
    y
end

function _cg_op(Aop!, b, x0, maxiter)
    x = deepcopy(x0)
    p = similar(x)
    r = similar(x)
    q = similar(x)
    Aop!(q, x) # p = b - A * x
    @. p = b - q
    @. r = p
    for iter in 1:maxiter
        Aop!(q, p) # mul!(q, A, p)
        rho = dot(r, r)
        alpha = rho / dot(p, q)
        @. x += alpha * p
        @. r -= alpha * q
        beta = dot(r, r) / rho
        @. p = r + beta * p
    end
    return x
end

# function _cg(A, b, x0, maxiter)
#     x = deepcopy(x0)
#     g = x' * A - b'
#     d = -g'
#     for iter in 1:maxiter
#         Ad = A * d
#         rho = (d' * Ad)
#         alpha = (-g * d) / rho
#         x = x + alpha * d
#         g = x' * A - b'
#         beta = (g * Ad) / rho
#         d = beta * d - g'
#     end
#     return x
# end


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
    N = 450 # number of subdivisions along the sides of the square domain

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
    @time T_f = K_ff \ (F_f - K_fd * T_d)
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

    T_if = [K_ff K_fi; K_if K_ii] \ [F_f - K_fd * T_d; F_i - K_id * T_d]
    T_i_ref = T_if[ir]

    # spy_matrix(K_ff, "K_ff")
    # spy_matrix(K_ii, "K_ii")

    approx_T = deepcopy(Temp.values)
  
    @show size(K_ii), size(K)
    
    Sop = SLinearOperator(K_ii, K_fi, K_if, K_ff)
    b = (F_i - K_id * T_d - K_if * (K_ff \ (F_f - K_fd * T_d)))
    x0 = zeros(size(b)) .+ 1.0
    @info "Preconditioned CG: Krylov"
    P = cholesky(Sop.K_ii)
    # @time (T_i, stats) = cg(Sop, b, x0; M=P)
    # @time (T_i, stats) = cg(Sop, b)
    # show(stats)
    # @show norm(T_i - T_i_ref)
    @info "CG without preconditioner: Krylov"
    @info "Make S"
    S = K_ii - K_if * (K_ff \ Matrix(K_fi))
    @time (T_i, stats) = cg(S, b)
    show(stats)
    @show norm(T_i - T_i_ref)
    # @info "CG with no preconditioner: _cg"
    # x = _cg(S, b, zeros(size(b)), 100);
    # @time x = _cg(S, b, zeros(size(b)), stats.niter)
    # @show norm(x - T_i)
    
    @info "CG with no preconditioner: _cg_smith"
    @time x = _cg_smith(S, b, x0, stats.niter)
    @show norm(x - T_i_ref)
    @info "CG with no preconditioner: _cg_op"
    @time x = _cg_op((q, p) -> mul!(q, S, p), b, x0, stats.niter)
    @show norm(x - T_i_ref)
    @info "CG with preconditioner: _cg_op"
    Pre = SPreConditioner(S, x0, lu(K_ii))
    @time x = _cg_op((q, p) -> mul!(q, Pre, p), Pre.factor \ b, x0, stats.niter)
    # @time x = _cg_op((q, p) -> mul!(q, S, p), b, zeros(size(b)), stats.niter)
    @show norm(x - T_i_ref)

    @info "Recovery of the solution in the interior"
    # T_i = S \ (F_i - K_id * T_d - K_if * (K_ff \ (F_f - K_fd * T_d)))
    @time T_f = K_ff \ (F_f - K_fd * T_d - K_fi * T_i)
    scattersysvec!(Temp, T_f, DOF_KIND_FREE)
    scattersysvec!(Temp, T_i, DOF_KIND_INTERFACE)
    scattersysvec!(Temp, T_d, DOF_KIND_DATA)
    approx_T .= Temp.values

    VTK.vtkexportmesh("approx.vtk", fes.conn, [geom.values approx_T], VTK.T3; scalars=[("Temperature", approx_T,)])


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
