prefix="bb-weak-grow"
n1=6
Nc=0

for No in 1 3 5 ; do
    for N in $(seq 3 10) ; do
        for Np in 4 8 16  ; do
            filename="${prefix}-N=$N-Np=$Np-No=$No.json"
            echo $filename
            julia conc/lindef/bb.jl --filename "$filename" \
                --N $N --Nc $Nc --n1 $n1 --Np $Np --No $No; 
        done
    done
done
