module CGModule

using LinearAlgebra
using SparseArrays
using MPI

"""
    pcg_seq(Aop!, b, x0; M! =(q, p) -> (q), itmax=0, atol=√eps(eltype(b)), rtol=√eps(eltype(b)))

Solves a linear system `Ax = b` using the Preconditioned Conjugate Gradient
method.

The multiplication with the matrix `A` is accomplished with an operator `Aop!`.
The solution with the preconditioner `M` is accomplished with an operator `M!`.
This is a sequential version of the `pcg` function.

# Arguments
- `Aop!`: Function that applies the linear operator `A` to a vector `x` and
  stores the result in `y`. Example: `(q, p) -> mul!(q, S, p)`.
- `b`: Right-hand side vector.
- `x0`: Initial guess for the solution.
- `M!`: Preconditioner function that applies the preconditioner `M` to a vector
  `q` and stores the result in `p`. Defaults to `(q, p) -> (q)` (identity).
- `itmax`: Maximum number of iterations. Defaults to `0`, which means the method
  will iterate until convergence.
- `atol`: Absolute tolerance for convergence. Defaults to `√eps(eltype(b))`.
- `rtol`: Relative tolerance for convergence. Defaults to `√eps(eltype(b))`.

# Returns
- (`x`, `stats`): Tuple of solution vector and solution statistics.

"""
function pcg_seq(Aop!, b, x0; 
    M! =(q, p) -> (q .= p), 
    itmax=0, 
    atol=√eps(eltype(b)), 
    rtol=√eps(eltype(b)), 
    peeksolution = (iter, x, resnorm) -> nothing
)
    itmax = (itmax > 0 ? itmax : length(b))
    x = deepcopy(x0)
    p = similar(x)
    r = similar(x)
    z = similar(x)
    Ap = similar(x)
    Aop!(Ap, x) 
    @. r = b - Ap
    M!(z, r)
    @. p = z
    rho = dot(z, r)
    tol = atol + rtol * sqrt(rho)
    resnorm = Inf
    residuals = typeof(tol)[]
    stats = (niter=itmax, resnorm=resnorm, residuals=residuals)
    iter = 1
    while iter < itmax
        Aop!(Ap, p)
        rho = dot(z, r)
        alpha = rho / dot(p, Ap)
        @. x += alpha * p
        @. r -= alpha * Ap
        M!(z, r)
        beta = dot(z, r) / rho
        @. p = z + beta * p
        resnorm = sqrt(rho)
        push!(residuals, resnorm)
        peeksolution(iter, x, resnorm)
        if resnorm < tol
            break
        end
        iter += 1
    end
    stats = (niter=iter, resnorm=resnorm, residuals=residuals)
    return (x, stats)
end

function pcg_mpi(Aop!, b, x0; M! =(q, p) -> (q .= p), itmax=0, atol=√eps(eltype(b)), rtol=√eps(eltype(b)))
    itmax = (itmax > 0 ? itmax : length(b))
    tol = atol
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    x = deepcopy(x0)
    p = similar(x)
    r = similar(x)
    z = similar(x)
    Ap = similar(x); Ap .= 0.0
    MPI.Bcast!(x, comm; root=0)
    Aop!(Ap, x)
    MPI.Reduce!(Ap, MPI.SUM, comm; root=0)
    if rank == 0
        @. r = b - Ap
        M!(z, r)
        @. p = z
        rho = dot(z, r)
        tol += rtol * sqrt(rho)
    end
    tol = MPI.Bcast(tol, 0, comm)
    resnorm = Inf
    stats = (niter=itmax, resnorm=resnorm)
    iter = 1
    while iter < itmax
        MPI.Bcast!(p, comm; root=0)
        Aop!(Ap, p)
        MPI.Reduce!(Ap, MPI.SUM, comm; root=0)
        if rank == 0
            rho = dot(z, r)
            alpha = rho / dot(p, Ap)
            @. x += alpha * p
            @. r -= alpha * Ap
            M!(z, r)
            beta = dot(z, r) / rho
            @. p = z + beta * p
            resnorm = sqrt(rho)
        end
        resnorm = MPI.Bcast(resnorm, 0, comm)
        if resnorm < tol
            break
        end
        iter += 1
    end
    MPI.Bcast!(x, comm; root=0)
    stats = (niter=iter, resnorm=resnorm)
    return (x, stats)
end

end # module