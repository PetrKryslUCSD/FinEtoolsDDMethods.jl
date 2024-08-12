script=cfcond
E=1.0e3
nu=0.499999
for rf in 4; do
    for nf in 4 ; do
        for n1 in 4 5; do
            for ne in 1000; do
                for ov in 1 ; do
                    jsonfile=fibers-${script}-hex-rf=$rf-ne=$ne-n1=$n1-nf=$nf.json
                    if [ -f ${jsonfile} ]; then
                        echo "${jsonfile} exists"
                    else
                        julia conc/lindef/${script}.jl --nelperpart $ne \
                             --nbf1max $n1 --ref $rf --nfpartitions $nf --overlap $ov \
                             --E $E --nu $nu;
                    fi
                done
            done
        done
    done
done
