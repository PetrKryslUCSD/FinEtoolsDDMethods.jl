prefix="zc-weak-grow"
n1=6
No=1
Nep=10000
Np=4
Nt=$((n1*(n1-1)*(n1-2)/6))
for ref in 3 4 5 6  ; do
    # echo "ref = $ref"
    N=$(((192*4**(ref-1))))
    Nep=$((N/Np))
    # echo "Nep = $Nep"
    meanNsub=$((Nep/2*6))
    Nc=$((meanNsub/6/Nt))
    echo "Np = $Np, Nc = $Nc"
    julia conc/shells/zc.jl --prefix "$prefix" --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No; 
done
