script=barrel
for rf in 1; do
    for nf in 8 16 32; do
        for n1 in 4 5 6; do
            for ne in $(seq 200 400 1000); do
                for ov in 1 3 5; do
                    jsonfile=barrel_overlapped-rf=$rf-ne=$ne-n1=$n1-nf=$nf-ov=$ov.json
                    if [ -f ${jsonfile} ]; then
                        echo "${jsonfile} exists" 
                    else
                        julia conc/shells/${script}.jl --nelperpart $ne --nbf1max $n1 --ref $rf --nfpartitions $nf --overlap $ov  --stabilize false; 
                    fi
                done
            done 
        done
    done
done

