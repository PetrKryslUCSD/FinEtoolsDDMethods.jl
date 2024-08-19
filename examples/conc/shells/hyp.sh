script=hyp
for as in 100 200 500 1000 2000 5000 10000; do
    for rf in  8; do
        for nf in 4 8 16; do
            for n1 in 6; do
                for ne in $(seq 100 200 400); do
                    for ov in 1 3 5; do
                        jsonfile=cos_2t_press_hyperboloid_free-${script}-as=$as-rf=$rf-ne=$ne-n1=$n1-nf=$nf.json
                        if [ -f ${jsonfile} ]; then
                            echo "${jsonfile} exists" 
                        else
                            julia conc/shells/${script}.jl --nelperpart $ne --nbf1max $n1 --ref $rf --nfpartitions $nf --overlap $ov --aspect $as; 
                        fi
                    done
                done 
            done
        done
    done
done
