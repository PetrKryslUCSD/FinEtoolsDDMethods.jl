script=cc
for rf in 6 ; do
    for nf in 4 8 16; do
        for n1 in 6; do
            for ne in 400 ; do
                for ov in  5; do
                    jsonfile=fibers-${script}-hex-rf=$rf-ne=$ne-n1=$n1-nf=$nf-ov=$ov.json
                    if [ -f ${jsonfile} ]; then
                        echo "${jsonfile} exists" 
                    else
                        julia conc/lindef/${script}.jl --kind tet --nelperpart $ne --nbf1max $n1 --ref $rf --nfpartitions $nf --overlap $ov; 
                    fi
                done
            done 
        done
    done
done
