"""
    Run as
mpiexecjl -n 3 julia --project=. mwe.jl
"""

module mwe

using MPI
using LinearAlgebra
using Statistics

function work(N = 250)
    M = rand(N, N)
    factor = lu(M)
    return factor
end

function _execute(N)
    t0 = time()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    rank == 0 && (@info "$(MPI.Get_library_version())")

    BLAS_THREADS = parse(Int, """$(get(ENV, "BLAS_THREADS", 1))""")
    rank == 0 && (@info "BLAS_THREADS = $(BLAS_THREADS)")
    BLAS.set_num_threads(BLAS_THREADS)

    nfpartitions = nprocs - 1

    rank == 0 && (@info "Number of processes: $nprocs")
    rank == 0 && (@info "Number of partitions (number of processes - 1): $nfpartitions")

    t1 = time()
    if rank == 0
        @info "DEBUG rank=$rank starts work at $(time() - t0)"
        junk = work(N)
        @info "DEBUG rank=$rank ($(round(time() - t1, digits=3)) [s])"
    end
    t1 = time()
    partition = nothing
    if rank > 0
        @info "DEBUG rank=$rank starts work at $(time() - t0)"
        partition = work(N + 20 * rank)
        @info "DEBUG rank=$rank ($(round(time() - t1, digits=3)) [s])"
    end
    MPI.Barrier(comm)
    
    MPI.Finalize()

    true
end

function test(;N = 10000)
    _execute(N)
end

test()

nothing
end # module
 