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
        resnorm = sqrt(dot(r, r))
        tol = atol + rtol * resnorm
    else
        resnorm = sqrt(rhoold)
        tol = atol + rtol * resnorm
    end
    residuals = typeof(tol)[]
    peeksolution(0, x, resnorm)
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

"""
    pcg_mpi_2level_Schwarz_alt(
        comm, 
        rank,
        Aop!,
        b,
        x0,
        (M!)=(q, p) -> (q .= p);
        itmax=0,
        atol=√eps(eltype(b)),
        rtol=√eps(eltype(b)),
        normtype = KSP_NORM_NATURAL,
        peeksolution = (iter, x, resnorm) -> nothing
        )

Solves a linear system `Ax = b` using the Preconditioned Conjugate Gradient
method on MPI.

The communicator and the rank of the process are passed as arguments `comm` and
`rank`. The multiplication with the matrix `A` is accomplished with an operator
`Aop!`. 

The preconditioning on level 1 (using local solves on each subdomain partition)
and the  preconditioning on level 2 (using global solve based on CoNC
clustering) is accomplished with an operator `M!`. 

This is a MPI parallel version of the `pcg` function.

# Arguments
- `Aop!`: Function that applies the linear operator `A` to a vector `x` and
  stores the result in `y`. Example: `(q, p) -> mul!(q, S, p)`.
- `b`: Right-hand side vector.
- `x0`: Initial guess for the solution.
- `M!`: Preconditioner function that applies the 2-level preconditioner to a
  vector `p` and stores the result in `q`. Defaults to `(q, p) -> (q .= p)`
  (identity). 
- `itmax`: Maximum number of iterations. Defaults to `0`, which means the method
  will iterate until convergence.
- `atol`: Absolute tolerance for convergence. Defaults to `√eps(eltype(b))`.
- `rtol`: Relative tolerance for convergence. Defaults to `√eps(eltype(b))`.
- `normtype`: Type of norm to use for convergence. Defaults to
  `KSP_NORM_NATURAL`.
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
    alpha = zero(typeof(atol))
    beta = zero(typeof(atol))
    z = zg # Alias for legibility
    Ap = z # Alias for legibility
    MPI.Bcast!(x, comm; root=0) # Broadcast the initial guess
    Aop!(Ap, x) # If partition, compute contribution to the A*p
    MPI.Reduce!(Ap, MPI.SUM, comm; root=0) # Reduce the A*p
    if rank == 0
        @. r = b - Ap # Compute the residual
    end
    req = MPI.Ibcast!(r, comm; root=0) # Broadcast the residual
    if rank == 0
        MG!(zg, r) # If root, apply the global preconditioner
    end
    MPI.Wait(req) # Wait for the residual to be broadcasted
    ML!(zl, r) # Apply the local preconditioner, if partition
    MPI.Reduce!(zl, MPI.SUM, comm; root=0) # Reduce the local preconditioner
    tol = zero(typeof(atol))
    resnorm = Ref(Inf)
    if rank == 0
        @. z = zl + zg # Combine the local and global preconditioners
        @. p = z
        rhoold = dot(z, r)
        if normtype == KSP_NORM_UNPRECONDITIONED
            resnorm[] = sqrt(dot(r, r))
            tol = atol + rtol * resnorm[]
        else
            resnorm[] = sqrt(rhoold)
            tol = atol + rtol * resnorm[]
        end
    end
    tol = MPI.Bcast(tol, 0, comm) # Broadcast the tolerance
    residuals = typeof(tol)[]
    rank == 0 && peeksolution(0, x, resnorm[])
    iter = 1
    t198 = 0.0
    t201 = 0.0
    t204 = 0.0
    t209 = 0.0
    t215 = 0.0
    t221 = 0.0
    t224 = 0.0
    t227 = 0.0
    t239 = 0.0
    while iter < itmax
        tstart = MPI.Wtime()
        MPI.Bcast!(p, comm; root=0) # Broadcast the search direction
        tend = MPI.Wtime(); t198 += tend - tstart; tstart = tend
        Aop!(Ap, p) # If partition, compute contribution to the A*p
        tend = MPI.Wtime(); t201 += tend - tstart; tstart = tend
        MPI.Reduce!(Ap, MPI.SUM, comm; root=0) # Reduce the A*p
        tend = MPI.Wtime(); t204 += tend - tstart; tstart = tend
        if rank == 0
            alpha = rhoold / dot(p, Ap)
            @. r -= alpha * Ap # Update the residual
        end
        tend = MPI.Wtime(); t209 += tend - tstart; tstart = tend
        req = MPI.Ibcast!(r, comm; root=0) # Broadcast the residual
        if rank == 0
            MG!(zg, r) # If root, apply the global preconditioner
            @. x += alpha * p # Update the solution
        end
        MPI.Wait(req) # Wait for the broadcast to finish
        tend = MPI.Wtime(); t215 += tend - tstart; tstart = tend
        ML!(zl, r) # Apply the local preconditioner, if partition
        tend = MPI.Wtime(); t221 += tend - tstart; tstart = tend
        MPI.Reduce!(zl, MPI.SUM, comm; root=0) # Reduce the local preconditioner
        tend = MPI.Wtime(); t224 += tend - tstart; tstart = tend
        if rank == 0
            @. z = zl + zg # Combine the local and global preconditioners
            rho = dot(z, r)
            beta = rho / rhoold;   rhoold = rho
            if normtype == KSP_NORM_UNPRECONDITIONED
                resnorm[] = sqrt(dot(r, r))
            else
                resnorm[] = sqrt(rho) 
            end
        end
        tend = MPI.Wtime(); t227 += tend - tstart; tstart = tend
        req = MPI.Ibcast!(resnorm, comm; root=0) # Broadcast the residual norm 
        if rank == 0
            @. p = z + beta * p # Update the search direction
        end
        MPI.Wait(req) # Wait for the broadcast to finish
        tend = MPI.Wtime(); t239 += tend - tstart; tstart = tend
        push!(residuals, resnorm[])
        rank == 0 && peeksolution(iter, x, resnorm[])
        if resnorm[] < tol
            @info """Rank $rank 
                    broadcast p          : $(t198) 
                    A op                 : $(t201) 
                    reduce Ap            : $(t204) 
                    update r             : $(t209) 
                    compute zg + update x: $(t215) 
                    compute zl           : $(t221) 
                    reduce zl            : $(t224) 
                    update z, z*r        : $(t227) 
                    bcast resnorm, upda p: $(t239) 
                    """
            break
        end
        iter += 1
    end
    MPI.Bcast!(x, comm; root=0) # Broadcast the solution
    iter = MPI.Bcast(iter, 0, comm) # Broadcast the number of iterations
    resnorm[] = MPI.Bcast(resnorm[], 0, comm) # Broadcast the residual norm
    stats = (niter=iter, resnorm=resnorm[], residuals=residuals)
    return (x, stats)
end


function pcg_mpi_2level_Schwarz_alt(
    comm, 
    rank,
    Aop!,
    b,
    x0,
    (M!)=(q, p) -> (q .= p);
    itmax=0,
    atol=√eps(eltype(b)),
    rtol=√eps(eltype(b)),
    normtype = KSP_NORM_NATURAL,
    peeksolution = (iter, x, resnorm) -> nothing
    )
    itmax = (itmax > 0 ? itmax : length(b))
    (normtype == KSP_NORM_UNPRECONDITIONED || normtype == KSP_NORM_NATURAL) || throw(ArgumentError("Invalid normtype"))
    x = deepcopy(x0); p = similar(x); r = similar(x); z = similar(x);
    alpha = zero(typeof(atol))
    beta = zero(typeof(atol))
    Ap = z # Alias for legibility
    Aop!(Ap, x) # If partition, compute contribution to the A*p
    if rank == 0
        @. r = b - Ap # Compute the residual
    end
    M!(z, r) # Apply the 2-level preconditioner
    tol = zero(typeof(atol))
    resnorm = Ref(Inf)
    if rank == 0
        @. p = z
        rhoold = dot(z, r)
        if normtype == KSP_NORM_UNPRECONDITIONED
            resnorm[] = sqrt(dot(r, r))
            tol = atol + rtol * resnorm[]
        else
            resnorm[] = sqrt(rhoold)
            tol = atol + rtol * resnorm[]
        end
    end
    tol = MPI.Bcast(tol, 0, comm) # Broadcast the tolerance
    residuals = typeof(tol)[]
    rank == 0 && peeksolution(0, x, resnorm[])
    iter = 1
    t201 = 0.0
    t209 = 0.0
    t215 = 0.0
    t227 = 0.0
    t239 = 0.0
    while iter < itmax
        tstart = MPI.Wtime()
        Aop!(Ap, p) # Compute A*p
        tend = MPI.Wtime(); t201 += tend - tstart; tstart = tend
        if rank == 0
            alpha = rhoold / dot(p, Ap)
            @. r -= alpha * Ap # Update the residual
        end
        tend = MPI.Wtime(); t209 += tend - tstart; tstart = tend
        M!(z, r) # Apply the 2-level preconditioner
        if rank == 0
            @. x += alpha * p # Update the solution
        end
        tend = MPI.Wtime(); t215 += tend - tstart; tstart = tend
        if rank == 0
            rho = dot(z, r)
            beta = rho / rhoold;   rhoold = rho
            if normtype == KSP_NORM_UNPRECONDITIONED
                resnorm[] = sqrt(dot(r, r))
            else
                resnorm[] = sqrt(rho) 
            end
        end
        tend = MPI.Wtime(); t227 += tend - tstart; tstart = tend
        req = MPI.Ibcast!(resnorm, comm; root=0) # Broadcast the residual norm 
        if rank == 0
            @. p = z + beta * p # Update the search direction
        end
        MPI.Wait(req) # Wait for the broadcast to finish
        tend = MPI.Wtime(); t239 += tend - tstart; tstart = tend
        push!(residuals, resnorm[])
        rank == 0 && peeksolution(iter, x, resnorm[])
        if resnorm[] < tol
            @info """Rank $rank 
                    A op                 : $(t201) 
                    update r             : $(t209) 
                    compute Z + update x: $(t215) 
                    update z, z*r        : $(t227) 
                    bcast resnorm, upda p: $(t239) 
                    """
            break
        end
        iter += 1
    end
    MPI.Bcast!(x, comm; root=0) # Broadcast the solution
    iter = MPI.Bcast(iter, 0, comm) # Broadcast the number of iterations
    resnorm[] = MPI.Bcast(resnorm[], 0, comm) # Broadcast the residual norm
    stats = (niter=iter, resnorm=resnorm[], residuals=residuals)
    return (x, stats)
end


end # module