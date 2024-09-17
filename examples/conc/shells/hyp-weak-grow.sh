prefix="hyp-weak-grow"
n1=6
No=5

for as in 10 100 ; do
    for ref in $(seq 4 20) ; do
        Np=$((ref**2/2))
        Nc=$((2*Np))
        filename="${prefix}-ref=$ref-as=$as-Np=$Np-No=$No.json"
        julia conc/shells/hyp.jl --filename "$filename" \
        --aspect $as \
        --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No;
    done
done
