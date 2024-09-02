prefix="zc-weak-grow"
n1=6
Nt=$((n1*(n1+1)*(n1+2)/6)) # three dimensional body
Nt=$((n1*(n1+1)/2)) # shell

for No in 1 3 5 ; do
    for ref in  5 6 7 ; do
        for Np in 4 8 16  ; do
            # echo "ref = $ref"
            N=$(((192*4**(ref-1))))
            Nep=$((N/Np))
            # echo "Nep = $Nep"
            meanNsub=$((Nep/2*6))
            Nc=$((meanNsub/Nt))
            echo "Np = $Np, Nc = $Nc"
            julia conc/shells/zc.jl --prefix "$prefix" --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No; 
        done
    done
done
