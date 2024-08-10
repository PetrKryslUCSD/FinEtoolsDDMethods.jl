script=cni
for rf in 6 8 10; do
    for nf in 32; do
        for n1 in 6; do
            for ne in 2000; do
                for ov in 5; do
                    jsonfile=fibers-${script}-hex-rf=$rf-ne=$ne-n1=$n1-nf=$nf.json
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
