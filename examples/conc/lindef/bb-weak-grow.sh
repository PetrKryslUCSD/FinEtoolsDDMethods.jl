prefix="bb-weak-grow"
n1=6
# Nt=$((n1*(n1+1)*(n1+2)/6)) # three dimensional body
# Nt=$((n1*(n1+1)/2)) # shell

for No in 1 3 5 ; do
    for N in $(seq 3 10) ; do
        for Np in 4 8 16  ; do
            Nc=0
            echo "Np = $Np, Nc = $Nc"
            julia conc/lindef/bb.jl --prefix "$prefix" --N $N --Nc $Nc --n1 $n1 --Np $Np --No $No; 
        done
    done
done
