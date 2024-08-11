script=cfcond
Em=1.0
Ef=1.0
num=0.3
nuf=0.3
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
                             --Em $Em --Ef $Ef --num $num --nuf $nuf; 
                    fi
                done
            done 
        done
    done
done
