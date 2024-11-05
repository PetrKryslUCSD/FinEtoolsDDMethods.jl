#!/bin/sh

arg0=$(basename "$0" .sh)
blnk=$(echo "$arg0" | sed 's/./ /g')

usage_info()
{
    echo "Usage:"
    echo "for Np in 16 32 64 ; do "
    echo "   bash $arg0.sh --Np \$Np > do-\$Np.sh; "
    echo "   sbatch do-\$Np.sh; "
    echo "done"
}

usage()
{
    exec 1>2   # Send standard output to standard error
    usage_info
    exit 1
}

error()
{
    echo "$arg0: $*" >&2
    exit 1
}

help()
{
    usage_info
    echo
    echo "  {--filename} f                 -- Use filename to name the output files"
    echo "  {--kind} hex or tet            -- Relative residual tolerance"
    echo "  {--Nc} integer                 -- Number of clusters"
    echo "  {--n1} integer                 -- Number 1D basis functions"
    echo "  {--Np} integer                 -- Number fine grid partitions"
    echo "  {--No} integer                 -- Number of overlaps"
    echo "  {--N} integer                  -- Number of element edges in one direction"
    echo "  {--ref} integer                -- Refinement factor"
    echo "  {--itmax} integer              -- Maximum number of iterations allowed"
    echo "  {--relrestol} float            -- Relative residual tolerance"
    echo "  {--Ef} float                   -- Young's modulus of the fibres"
    echo "  {--nuf} float                  -- Poisson ratio of the fibres"
    echo "  {--Em} float                   -- Young's modulus of the matrix"
    echo "  {--num} float                  -- Poisson ratio of the matrix"
    echo "  {--peek} true or false         -- Peek at the iterations?"
    echo "  {--visualize} true or false    -- Write out visualization files?"
    echo "  {-h|--help}                    -- Print this help message and exit"
#   echo "  {-V|--version}                 -- Print version information and exit"
    exit 0
}

flags()
{
    export FILENAME=""
    export NC=100
    export N1=6
    export NP=2
    export NO=5
    export N=2
    export REF=6
    export ITMAX=2000
    export RELRESTOL=1.0e-6
    export KIND="hex"
    export EF=1.00e5
    export NUF=0.3
    export EM=1.00e3
    export NUM=0.4999
    export PEEK=true
    export VISUALIZE=false

    while test $# -gt 0
    do
        case "$1" in
        (--filename)
            shift
            [ $# = 0 ] && error "No file name specified"
            export FILENAME="$1"
            shift;;
        (--kind)
            shift
            [ $# = 0 ] && error "No kind specified"
            export KIND="$1"
            shift;;
        (--Nc)
            shift
            [ $# = 0 ] && error "No value specified"
            export NC="$1"
            shift;;
        (--n1)
            shift
            [ $# = 0 ] && error "No value specified"
            export N1="$1"
            shift;;
        (--Np)
            shift
            [ $# = 0 ] && error "No value specified"
            export NP="$1"
            shift;;
        (--No)
            shift
            [ $# = 0 ] && error "No value specified"
            export NO="$1"
            shift;;
        (--ref)
            shift
            [ $# = 0 ] && error "No value specified"
            export REF="$1"
            shift;;
        (--itmax)
            shift
            [ $# = 0 ] && error "No value specified"
            export ITMAX="$1"
            shift;;
        (--relrestol)
            shift
            [ $# = 0 ] && error "No value specified"
            export RELRESTOL="$1"
            shift;;
        (--Ef)
            shift
            [ $# = 0 ] && error "No value specified"
            export EF="$1"
            shift;;
        (--nuf)
            shift
            [ $# = 0 ] && error "No value specified"
            export NUF="$1"
            shift;;
        (--Em)
            shift
            [ $# = 0 ] && error "No value specified"
            export EM="$1"
            shift;;
        (--num)
            shift
            [ $# = 0 ] && error "No value specified"
            export NUM="$1"
            shift;;
        (--peek)
            shift
            [ $# = 0 ] && error "No value specified"
            export PEEK="$1"
            shift;;
        (--visualize)
            shift
            [ $# = 0 ] && error "No value specified"
            export VISUALIZE="$1"
            shift;;
        (-h|--help)
            help;;
#       (-V|--version)
#           version_info;;
        (*) usage;;
        esac
    done
}

flags "$@"

if [ -z "$FILENAME" ] ; then
    FILENAME="fib-const-$KIND-ref=$REF-Nc=$NC-Np=$NP.json"
fi

QUEUE=short
if [ $NP -gt 16 ] ; then
        QUEUE=medium
fi
if [ $NP -gt 32 ] ; then
        QUEUE=large
fi

cat <<EOF
#!/usr/bin/env bash

#SBATCH --job-name=job_fib
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=$NP
#SBATCH --time=01:45:00
#SBATCH -p $QUEUE
#SBATCH --output=$FILENAME.out

# Load OpenMPI and Julia
export JULIA_DEPOT_PATH="~/a64fx/depot"
export PATH=\$PATH:"~/a64fx/depot/bin"
module load julia
module load gcc/13.1.0
module load slurm
module load openmpi/gcc8/4.1.2

# Automatically set the number of Julia threads depending on number of Slurm threads
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export BLAS_THREADS=2

cd FinEtoolsDDMethods.jl/examples
mpiexecjl julia --project=. conc/lindef/fib_mpi_driver.jl \
--filename $FILENAME \
--kind $KIND \
--Nc $NC \
--n1 $N1 \
--No $NO \
--ref $REF \
--itmax  $ITMAX \
--relrestol  $RELRESTOL \
--Ef  $EF \
--nuf  $NUF \
--Em  $EM \
--num  $NUM \
--peek  $PEEK \
--visualize  $VISUALIZE 
EOF



