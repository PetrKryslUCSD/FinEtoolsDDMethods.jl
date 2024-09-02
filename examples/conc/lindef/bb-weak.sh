script=bb
for rf in 2 4 8 16 ; do
    for nf in $((rf*rf*rf)); do
        for n1 in 6; do
            for ne in 4000; do
                for ov in 5; do
                    jsonfile=${script}-hex-rf=$rf-ne=$ne-n1=$n1-nf=$nf-ov=$ov.json
                    if [ -f ${jsonfile} ]; then
                        echo "${jsonfile} exists" 
                    else
                        #echo "${jsonfile} " 
                        julia conc/lindef/${script}.jl --Nepc $ne --nbf1max $n1 --ref $rf --nfpartitions $nf --overlap $ov; 
                    fi
                done
            done 
        done
    done
done
