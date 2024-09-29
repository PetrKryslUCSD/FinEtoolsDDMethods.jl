module mpi_experiment_1

using MPI
using LinearAlgebra

function _execute(N)
    
    to = time()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    rank == 0 && (@info "$(MPI.Get_library_version())")

    BLAS_THREADS = parse(Int, """$(get(ENV, "BLAS_THREADS", 1))""")
    rank == 0 && (@info "BLAS_THREADS = $(BLAS_THREADS)")
    BLAS.set_num_threads(BLAS_THREADS)

    Np = nprocs - 1

    rank == 0 && (@info "Number of processes: $nprocs")
    rank == 0 && (@info "Number of partitions: $Np")
    rank == 0 && (@info "Size of the vector: $N")

    p = rand(N)

    tstart = time()
    MPI.Bcast!(p, comm; root=0) 
    tend = time()
    rank == 0 && (@info "Time to broadcast: $(tend - tstart) [s]")

    MPI.Finalize()
    rank == 0 && (@info("Total time: $(round(time() - to, digits=3)) [s]"))
    
    true
end

using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--N"
        help = "Size of the vector"
        arg_type = Int
        default = 2000000
        
    end
    return parse_args(s)
end

p = parse_commandline()

_execute(
    p["N"], 
    )


nothing
end # module


