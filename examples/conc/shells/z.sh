script=z
for as in 10; do
    for rf in  8; do
        for nf in 4 8 16 32 64 128; do
            for n1 in 4 5 6; do
                for ne in 5000 20000; do
                    for ov in 1 3 5; do
                        julia conc/shells/${script}.jl --nelperpart $ne --nbf1max $n1 --ref $rf --nfpartitions $nf --overlap $ov --aspect $as; 
                    done
                done 
            done
        done
    done
done
