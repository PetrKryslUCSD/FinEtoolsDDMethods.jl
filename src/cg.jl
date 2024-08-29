module CGModule

using LinearAlgebra
using SparseArrays
using MPI

const KSP_NORM_UNPRECONDITIONED = 0
const KSP_NORM_NATURAL = 1

"""
    pcg_seq(Aop!, b, x0; 
        M! =(q, p) -> (q .= p), 
        itmax=0, 
        atol=√eps(eltype(b)), 
        rtol=√eps(eltype(b)), 
        peeksolution = (iter, x, resnorm) -> nothing,
        normtype = KSP_NORM_NATURAL
    )

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
  `p` and stores the result in `q`. Defaults to `(q, p) -> (q .= p)` (identity).
- `itmax`: Maximum number of iterations. Defaults to `0`, which means the method
  will iterate until convergence.
- `atol`: Absolute tolerance for convergence. Defaults to `√eps(eltype(b))`.
- `rtol`: Relative tolerance for convergence. Defaults to `√eps(eltype(b))`.
- `normtype`: Type of norm to use for convergence. Defaults to `KSP_NORM_NATURAL`.
- `peeksolution`: Function that is called at each iteration. It receives the
  iteration number, the current solution vector and the residual norm. Defaults
  to `(iter, x, resnorm) -> nothing`.

# Returns
- (`x`, `stats`): Tuple of solution vector and solution statistics.
"""
function pcg_seq(Aop!, b, x0; 
    M! =(q, p) -> (q .= p), 
    itmax=0, 
    atol=√eps(eltype(b)), 
    rtol=√eps(eltype(b)), 
    peeksolution = (iter, x, resnorm) -> nothing,
    normtype = KSP_NORM_NATURAL
    )
    itmax = (itmax > 0 ? itmax : length(b))
    (normtype == KSP_NORM_UNPRECONDITIONED || normtype == KSP_NORM_NATURAL) || throw(ArgumentError("Invalid normtype"))
    x = deepcopy(x0); p = similar(x); r = similar(x); z = similar(x); 
    Ap = z # Alias for legibility
    Aop!(Ap, x) 
    @. r = b - Ap
    M!(z, r)
    @. p = z
    rhoold = dot(z, r)
    if normtype == KSP_NORM_UNPRECONDITIONED
        tol = atol + rtol * sqrt(dot(r, r))
    else
        tol = atol + rtol * sqrt(rhoold)
    end
    resnorm = Inf
    residuals = typeof(tol)[]
    stats = (niter=itmax, resnorm=resnorm, residuals=residuals)
    iter = 1
    while iter < itmax
        Aop!(Ap, p)
        alpha = rhoold / dot(p, Ap)
        @. x += alpha * p
        @. r -= alpha * Ap
        M!(z, r)
        rho = dot(z, r)
        beta = rho / rhoold;   rhoold = rho
        @. p = z + beta * p
        if normtype == KSP_NORM_UNPRECONDITIONED
            resnorm = sqrt(dot(r, r))
        else
            resnorm = sqrt(rho) 
        end
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

# Smith, Gropp  1996
# There is something weird about this: it takes quite a few more iterations!
function pcg_seq_sg(Aop!, b, x0;
    (M!)=(q, p) -> (q .= p),
    itmax=0,
    atol=√eps(eltype(b)),
    rtol=√eps(eltype(b)),
    peeksolution=(iter, x, resnorm) -> nothing
    )
    itmax = (itmax > 0 ? itmax : length(b))
    x = deepcopy(x0); p = similar(x); r = similar(x); z = similar(x); 
    Ap = z # Alias for legibility
    tol = zero(typeof(atol))
    resnorm = Inf
    residuals = typeof(tol)[]
    stats = (niter=itmax, resnorm=resnorm, residuals=residuals)
    betaold = one(typeof(atol))
    Aop!(Ap, x) 
    @. r = b - Ap
    M!(z, r)
    @. p = z
    betaold = dot(z, r)
    iter = 1
    while iter < itmax
        beta = dot(z, r)
        c = beta / betaold; betaold = beta
        @. p = z + c * p
        Aop!(Ap, p)
        @show a = beta / dot(p, Ap)
        @. x += a * p
        @. r -= a * Ap
        resnorm = sqrt(dot(r, r))
        push!(residuals, resnorm)
        peeksolution(iter, x, resnorm)
        tol == 0 && (tol = atol + rtol * sqrt(beta))
        if resnorm < tol
            break
        end
        M!(z, r)
        iter += 1
    end
    stats = (niter=iter, resnorm=resnorm, residuals=residuals)
    return (x, stats)
end

"""
    pcg_mpi_2level_Schwarz(
        comm, 
        rank,
        Aop!,
        b,
        x0,
        (MG!)=(q, p) -> (q .= p),
        (ML!)=(q, p) -> (q .= p),
        itmax=0,
        atol=√eps(eltype(b)),
        rtol=√eps(eltype(b))
    )

Solves a linear system `Ax = b` using the Preconditioned Conjugate Gradient
method on MPI.

The communicator and the rank of the process are passed as arguments `comm` and
`rank`. The multiplication with the matrix `A` is accomplished with an operator
`Aop!`. 

The preconditioning on level 1 (using local solves on each subdomain partition)
is accomplished with an operator `ML!`. The preconditioning on level 2 (using
global solve based on CoNC clustering) is accomplished with an operator `MG!`. 

This is a MPI parallel version of the `pcg` function.

# Arguments
- `Aop!`: Function that applies the linear operator `A` to a vector `x` and
  stores the result in `y`. Example: `(q, p) -> mul!(q, S, p)`.
- `b`: Right-hand side vector.
- `x0`: Initial guess for the solution.
- `ML!`, `MG!`: Preconditioner functions that apply the preconditioner to a
  vector `p` and store the result in `q`. Defaults to `(q, p) -> (q .= p)`
  (identity). Local (level 1) and global (level 2) versions, respectively.
- `itmax`: Maximum number of iterations. Defaults to `0`, which means the method
  will iterate until convergence.
- `atol`: Absolute tolerance for convergence. Defaults to `√eps(eltype(b))`.
- `rtol`: Relative tolerance for convergence. Defaults to `√eps(eltype(b))`.
- `normtype`: Type of norm to use for convergence. Defaults to `KSP_NORM_NATURAL`.
- `peeksolution`: Function that is called at each iteration. It receives the
  iteration number, the current solution vector and the residual norm. Defaults
  to `(iter, x, resnorm) -> nothing`.

"""
function pcg_mpi_2level_Schwarz(
    comm, 
    rank,
    Aop!,
    b,
    x0,
    (MG!)=(q, p) -> (q .= p),
    (ML!)=(q, p) -> (q .= p);
    itmax=0,
    atol=√eps(eltype(b)),
    rtol=√eps(eltype(b)),
    normtype = KSP_NORM_NATURAL,
    peeksolution = (iter, x, resnorm) -> nothing
    )
    itmax = (itmax > 0 ? itmax : length(b))
    (normtype == KSP_NORM_UNPRECONDITIONED || normtype == KSP_NORM_NATURAL) || throw(ArgumentError("Invalid normtype"))
    x = deepcopy(x0); p = similar(x); r = similar(x); zg = similar(x); zl = similar(x)
    z = zg # Alias for legibility
    Ap = z # Alias for legibility
    MPI.Bcast!(x, comm; root=0) # Broadcast the initial guess
    Aop!(Ap, x) # If partition, compute contribution to the A*p
    MPI.Reduce!(Ap, MPI.SUM, comm; root=0) # Reduce the A*p
    if rank == 0
        @. r = b - Ap # Compute the residual
        MG!(zg, r) # If root, apply the global preconditioner
    end
    MPI.Bcast!(r, comm; root=0) # Broadcast the residual
    ML!(zl, r) # Apply the local preconditioner, if partition
    MPI.Reduce!(zl, MPI.SUM, comm; root=0) # Reduce the local preconditioner
    tol = zero(typeof(atol))
    if rank == 0
        @. z = zl + zg # Combine the local and global preconditioners
        @. p = z
        rhoold = dot(z, r)
        if normtype == KSP_NORM_UNPRECONDITIONED
            tol = atol + rtol * sqrt(dot(r, r))
        else
            tol = atol + rtol * sqrt(rhoold)
        end
    end
    tol = MPI.Bcast(tol, 0, comm) # Broadcast the tolerance
    resnorm = Inf
    residuals = typeof(tol)[]
    stats = (niter=itmax, resnorm=resnorm)
    iter = 1
    while iter < itmax
        MPI.Bcast!(p, comm; root=0) # Broadcast the search direction
        Aop!(Ap, p) # If partition, compute contribution to the A*p
        MPI.Reduce!(Ap, MPI.SUM, comm; root=0) # Reduce the A*p
        if rank == 0
            alpha = rhoold / dot(p, Ap)
            @. x += alpha * p
            @. r -= alpha * Ap
            MG!(zg, r) # If root, apply the global preconditioner
        end
        MPI.Bcast!(r, comm; root=0) # Broadcast the residual
        ML!(zl, r) # Apply the local preconditioner, if partition
        MPI.Reduce!(zl, MPI.SUM, comm; root=0) # Reduce the local preconditioner
        if rank == 0
            @. z = zl + zg # Combine the local and global preconditioners
            rho = dot(z, r)
            beta = rho / rhoold;   rhoold = rho
            @. p = z + beta * p
            if normtype == KSP_NORM_UNPRECONDITIONED
                resnorm = sqrt(dot(r, r))
            else
                resnorm = sqrt(rho) 
            end
        end
        resnorm = MPI.Bcast(resnorm, 0, comm) # Broadcast the residual norm
        push!(residuals, resnorm)
        rank == 0 && peeksolution(iter, x, resnorm)
        if resnorm < tol
            break
        end
        iter += 1
    end
    MPI.Bcast!(x, comm; root=0) # Broadcast the solution
    iter = MPI.Bcast(iter, 0, comm) # Broadcast the number of iterations
    resnorm = MPI.Bcast(resnorm, 0, comm) # Broadcast the residual norm
    stats = (niter=iter, resnorm=resnorm)
    return (x, stats)
end

end # module