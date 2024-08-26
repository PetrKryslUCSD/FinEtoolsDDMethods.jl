script=css
for rf in 6 8; do
    for nf in 4 8 16; do
        for n1 in $(seq 2 2 6); do
            for ne in $(seq 400 800 2000); do
                for ov in 1 3 5; do
                    jsonfile=fibers-${script}-hex-rf=$rf-ne=$ne-n1=$n1-nf=$nf-ov=$ov.json
                    if [ -f ${jsonfile} ]; then
                        echo "${jsonfile} exists" 
                    else
                        julia conc/lindef/${script}.jl --nelperpart $ne --nbf1max $n1 --ref $rf --nfpartitions $nf --overlap $ov; 
                    fi
                done
            done 
        done
    done
done
