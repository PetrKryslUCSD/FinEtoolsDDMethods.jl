module CGModule

using LinearAlgebra
using SparseArrays
using MPI
using ..FinEtoolsDDMethods: set_up_timers, update_timer!

const KSP_NORM_UNPRECONDITIONED = 0
const KSP_NORM_NATURAL = 1

function vec_copyto!(y, x)
    @. y = x 
end

# Computes y = x + a y.
function vec_aypx!(y, a, x)
    @. y = x + a * y
end

# Computes y += a x
function vec_ypax!(y, a, x)
    @. y = a * x + y
end

function vec_dot(x, y)
    return dot(x, y)
end

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
    x = deepcopy(x0); p = deepcopy(x); r = deepcopy(x); z = deepcopy(x); 
    Ap = z # Alias for legibility
    Aop!(Ap, x) 
    vec_copyto!(r, b); vec_ypax!(r, -1.0, Ap) # @. r = b - Ap
    M!(z, r)
    vec_copyto!(p, z); # @. p = z
    rhoold = vec_dot(z, r)
    if normtype == KSP_NORM_UNPRECONDITIONED
        resnorm = sqrt(vec_dot(r, r))
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
        alpha = rhoold / vec_dot(p, Ap)
        vec_ypax!(x, +alpha, p) # @. x += alpha * p
        vec_ypax!(r, -alpha, Ap) # @. r -= alpha * Ap
        M!(z, r)
        rho = vec_dot(z, r)
        beta = rho / rhoold;   rhoold = rho
        vec_aypx!(p, +beta, z) # @. p = z + beta * p
        if normtype == KSP_NORM_UNPRECONDITIONED
            resnorm = sqrt(vec_dot(r, r))
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
    pcg_mpi_2level_Schwarz(
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

This is an MPI parallel version of the `pcg` function.

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
    rhoold = zero(typeof(atol))
    Ap = z # Alias for legibility
    Aop!(Ap, x) # Compute A*p
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
    timers = set_up_timers("1_aop", "2_pre", "3_total")  
    while iter < itmax
        tstart = MPI.Wtime()
        Aop!(Ap, p) # Compute A*p
        update_timer!(timers, "1_aop", MPI.Wtime() - tstart)
        if rank == 0
            alpha = rhoold / dot(p, Ap)
            @. r -= alpha * Ap # Update the residual
        end
        tstart = MPI.Wtime()
        M!(z, r) # Apply the 2-level preconditioner
        update_timer!(timers, "2_pre", MPI.Wtime() - tstart)
        if rank == 0
            @. x += alpha * p # Update the solution
        end
        if rank == 0
            rho = dot(z, r)
            beta = rho / rhoold;   rhoold = rho
            if normtype == KSP_NORM_UNPRECONDITIONED
                resnorm[] = sqrt(dot(r, r))
            else
                resnorm[] = sqrt(rho) 
            end
        end
        req = MPI.Ibcast!(resnorm, comm; root=0) # Broadcast the residual norm 
        if rank == 0
            @. p = z + beta * p # Update the search direction
        end
        MPI.Wait(req) # Wait for the broadcast to finish
        push!(residuals, resnorm[])
        rank == 0 && peeksolution(iter, x, resnorm[])
        if resnorm[] < tol
            break
        end
        update_timer!(timers, "3_total", MPI.Wtime() - tstart)
        iter += 1
    end
    MPI.Bcast!(x, comm; root=0) # Broadcast the solution
    iter = MPI.Bcast(iter, 0, comm) # Broadcast the number of iterations
    resnorm[] = MPI.Bcast(resnorm[], 0, comm) # Broadcast the residual norm
    stats = (niter = iter, resnorm = resnorm[], residuals = residuals, timers = timers)
    return (x, stats)
end


end # module