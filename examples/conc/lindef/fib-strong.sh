prefix="fib-strong-cni"
kind="hex"
n1=6
# Nt=$((n1*(n1+1)*(n1+2)/6)) # three dimensional body
# Nt=$((n1*(n1+1)/2)) # shell
ref=6
Nc=0
Ef=100000.0
nuf=0.3
Em=1000.0
num=0.4999
for No in 1 3 5 ; do
        for Np in 4 8 16 32 64 ; do
            julia --project=. conc/lindef/fib.jl --prefix "$prefix" --ref $ref --kind $kind \
                --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No \
                --Ef $Ef --nuf $nuf --Em $Em --num $num; 
        done
    done
done
