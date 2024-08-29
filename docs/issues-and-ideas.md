-- Why does this CG version give different results than the original?

```
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
```