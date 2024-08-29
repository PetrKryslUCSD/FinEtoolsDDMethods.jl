script=z
for as in 10; do
    for rf in  8; do
        for nf in 32 64 128 256 512; do
            for n1 in 4 5 6; do
                for ne in  20000 200000 ; do
                    for ov in 5 ; do
                        julia -t 8 conc/shells/${script}.jl --nelperpart $ne --nbf1max $n1 --ref $rf --nfpartitions $nf --overlap $ov --aspect $as; 
                    done
                done 
            done
        done
    done
done
