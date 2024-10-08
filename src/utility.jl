using  OrderedCollections

function meminfo_julia(where = "")
    toint(n) = Int(ceil(n))
    @info """Memory: $(where)
    GC total:  $(toint(Base.gc_total_bytes(Base.gc_num())/2^20)) [MiB]
    GC live:   $(toint(Base.gc_live_bytes()/2^20)) [MiB]
    JIT:       $(toint(Base.jit_total_bytes()/2^20)) [MiB]
    Max. RSS:  $(toint(Sys.maxrss()/2^20)) [MiB]
    """
end

function allbytes(a)
    T = typeof(a)
    return if isbitstype(T)
        sizeof(a)
    else
        return if fieldcount(T) == 0
            if length(a) > 0
                sum(allbytes(a[i]) for i in eachindex(a))
            else
                sizeof(a)
            end
        else
            sum(allbytes(getfield(a, fieldname(T, i))) for i in 1:fieldcount(T))
        end
    end
end

function mebibytes(a)
    b = allbytes(a)
    Int(round(b/2^20, digits=0))
end    

function set_up_timers(names...)
    LittleDict([(name, 0.0) for name in names])
end

function update_timer!(timers, n, t)
    timers[n] += t
end

function reset_timers!(timers)
    for (k, v) in timers
        timers[k] = 0.0
    end
end

# @show allbytes(1)
# @show allbytes([1, 2, 3])
# @show allbytes([1, 2, [3, 2]])
# @show allbytes([1, 2, [3, 2.0+1im]])

# using SparseArrays
# a = sparse([1, 3, 4, 5, 2, 1], [4, 3, 2, 1, 5, 5], ones(6), 5, 5)
# # using About
# # about(a)
# T = typeof(a)
# for i in 1:fieldcount(T)
#     @show fieldname(T, i), fieldtype(T, i)
# end
# @show allbytes(a)

# using About
# about(1)
# about([1, 2, 3])
# about([1, 2, [3, 2]])
# about([1, 2, [3, 2.0+1im]])