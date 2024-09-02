prefix="zc-weak-sqrt"
n1=6
No=1
Nep=1000
for ref in $(seq 4 8) ; do
    Np=$(((192*4**(ref-1)) / Nep))
    Nc=`echo "sqrt($Np)" | bc`
    echo "Np = $Np, Nc = $Nc"
    julia conc/shells/zc.jl --prefix "$prefix" --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No; 
done
