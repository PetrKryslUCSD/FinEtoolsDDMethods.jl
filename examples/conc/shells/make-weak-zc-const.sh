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
    echo "  {--Nc} integer                 -- Number of clusters"
    echo "  {--n1} integer                 -- Number 1D basis functions"
    echo "  {--Np} integer                 -- Number fine grid partitions"
    echo "  {--No} integer                 -- Number of overlaps"
    echo "  {--Nepp} integer               -- Number of elements per partition "
    echo "  {--ref} integer                -- Refinement factor"
    echo "  {--itmax} integer              -- Maximum number of iterations allowed"
    echo "  {--relrestol} float            -- Relative residual tolerance"
    echo "  {--peek} true or false         -- Peek at the iterations?"
    echo "  {--visualize} true or false    -- Write out visualization files?"
    echo "  {-h|--help}                    -- Print this help message and exit"
#   echo "  {-V|--version}                 -- Print version information and exit"
    exit 0
}

flags()
{
    export FILENAME=""
    export NC=0 # let the software compute this
    export N1=6
    export NP=2
    export NO=5
    export NEPP=5000
    export REF=0
    export ITMAX=2000
    export RELRESTOL=1.0e-6
    export PEEK=true
    export VISUALIZE=false
    export NTPN=1

    while test $# -gt 0
    do
        case "$1" in
        (--filename)
            shift
            [ $# = 0 ] && error "No file name specified"
            export FILENAME="$1"
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
        (--Nepp)
            shift
            [ $# = 0 ] && error "No value specified"
            export NEPP="$1"
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
        (--Ntpn)
            shift
            [ $# = 0 ] && error "No value specified"
            export NTPN="$1"
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
    FILENAME="zc-weak-const-Ntpn=$NTPN-Nepp=$NEPP-Nc=$NC-Np=$NP.json"
fi

QUEUE=short
if [ $NP -gt $((16*NTPN)) ] ; then
        QUEUE=medium
fi
if [ $NP -gt $((32*NTPN)) ] ; then
        QUEUE=large
fi
if [ $NP -gt $((64*NTPN)) ] ; then
        QUEUE=all-nodes
fi

cat <<EOF
#!/usr/bin/env bash

#SBATCH --job-name=job_zc
#SBATCH --ntasks=$NP
#SBATCH --ntasks-per-node=$NTPN
#SBATCH --time=01:45:00
#SBATCH -p $QUEUE
#SBATCH --output=$(basename "$FILENAME" .json).out

export JULIA_DEPOT_PATH="~/a64fx/depot"
export PATH=$PATH:"~/a64fx/depot/bin"
module load julia
module load gcc/13.1.0
module load slurm
module load mpich/gcc12.2/4.1.1

export JULIA_NUM_THREADS=1
export BLAS_THREADS=2

cd ~/a64fx/FinEtoolsDDMethods.jl/examples
mpiexecjl -n $NP julia --project=. conc/shells/zc_mpi_driver.jl \
--filename $FILENAME \
--Np $NP \
--Nc $NC \
--n1 $N1 \
--No $NO \
--Nepp $NEPP \
--ref $REF \
--itmax  $ITMAX \
--relrestol  $RELRESTOL \
--peek  $PEEK \
--visualize  $VISUALIZE 
EOF



