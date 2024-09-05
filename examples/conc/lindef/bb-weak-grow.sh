prefix="bb-weak-grow"
n1=6
Nc=0
No=5
n1=6
nt=$(((n1*(n1+1)*(n1+2)/6))) # three dimensional body

for N in $(seq 2 9) ; do
    Ne=$((7*12*21*N**3)) # (6840/2)
    Ne2=$((7*12*21*2**3))
    Np=$((Ne/Ne2*2)) # (6840/2)
    Nc=$((Ne/Np*3/nt/3)) # (6840/2)
    filename="${prefix}-N=$N.json"
    echo "N=$N Nc=$Nc Np=$Np $filename "
    julia conc/lindef/bb.jl --filename "$filename" \
                    --N $N --Nc $Nc --n1 $n1 --Np $Np --No $No; 
done