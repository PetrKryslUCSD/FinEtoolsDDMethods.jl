script=barrel
for rf in 1; do
    for Np in 8 16 32; do
        for n1 in 6; do
            for Nc in 400; do
                for No in 1 3 5; do
                   julia conc/shells/${script}.jl --nelperpart $ne --nbf1max $n1 --ref $rf --nfpartitions $nf --overlap $ov  --stabilize false; 
                done
            done 
        done
    done
done

