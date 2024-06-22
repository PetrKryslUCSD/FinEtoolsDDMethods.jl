for rf in 4 6; do
    for nf in 4 6; do
        for n1 in $(seq 2 5); do
            for ne in $(seq 800 200 1000); do
                julia conc/lindef/cc.jl --nelperpart $ne --nbf1max $n1 --ref $rf --nfpartitions $nf; 
            done 
        done
    done
done
