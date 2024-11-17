# examples/03-reduce.jl
# This example shows how to use custom datatypes and reduction operators
# It computes the variance in parallel in a numerically stable way

using MPI, Statistics

MPI.Init()
const comm = MPI.COMM_WORLD
const root = 0

rank = MPI.Comm_rank(comm)

X = fill(rank, 7)

# Perform a sum reduction
X = MPI.Allreduce(X, MPI.SUM, comm)

if MPI.Comm_rank(comm) == root
    println("The sum of the arrays is: ", X)
end