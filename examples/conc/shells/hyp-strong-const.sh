prefix="hyp-strong-const"
n1=6
ref=7

for Nc in 100 500 ; do
    for as in 10000 10 100 1000 ; do
        for No in 1 3 5; do
            for Np in 4 8 16 32 64 128 ; do
                filename="${prefix}-ref=$ref-as=$as-Nc=$Nc-Np=$Np-No=$No.json"
                julia conc/shells/hyp.jl --filename "$filename" \
                    --aspect $as \
                    --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No; 
            done
        done
    done
done
