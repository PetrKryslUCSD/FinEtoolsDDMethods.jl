script=cc
for rf in 4 6; do
    for nf in 4 8 16; do
        for n1 in $(seq 2 5); do
            for ne in $(seq 400 400 2000); do
                jsonfile=fibers-${script}-hex-rf=$rf-ne=$ne-n1=$n1-nf=$nf.json
                if [ -f ${jsonfile} ]; then
                    echo "${jsonfile} exists" 
                else
                    julia conc/lindef/${script}.jl --nelperpart $ne --nbf1max $n1 --ref $rf --nfpartitions $nf; 
                fi
            done 
        done
    done
done
