script=hyp
for as in 100 1000 10000; do
    for rf in  8; do
        for nf in 4 8 16 32 64 128 ; do
            for n1 in 6; do
                for ne in 300; do
                    for ov in 5; do
                        jsonfile=cos_2t_p_hyp_free-${script}-as=$as-rf=$rf-ne=$ne-n1=$n1-nf=$nf.json
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
