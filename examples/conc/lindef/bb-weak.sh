Nc=10
n1=6
No=1
Nep=1000
for N in $(seq 1 10) ; do
    Np=$(((5*9*19*N)**3 / Nep))
    echo "Np = $Np"
    julia conc/lindef/bb.jl --Nc $Nc --n1 $n1 --N $N --Np $Np --No $No; 
done
