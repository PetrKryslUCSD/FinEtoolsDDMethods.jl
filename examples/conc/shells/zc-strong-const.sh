prefix="zc-strong-const"
n1=6
Nc=100
ref=7

for No in 1 3 5; do
    for Np in 4 8 16 32 64 128 ; do
        filename="${prefix}-ref=$ref-Nc=$Nc-Np=$Np-No=$No.json"
        julia conc/shells/zc.jl --filename "$filename" \
            --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No; 
    done
done

