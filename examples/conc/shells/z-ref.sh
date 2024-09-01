script=z
for as in 10; do
    for rf in 4 5 6 ; do
        for nf in 4 ; do
            for n1 in 6; do
                for ne in 1000 ; do
#                 for ne in $(($rf*1000)); do
                    for ov in 5; do
                        julia conc/shells/${script}.jl --nelperpart $ne --nbf1max $n1 --ref $rf --nfpartitions $nf --overlap $ov --aspect $as; 
                    done
                done 
            done
        done
    done
done
