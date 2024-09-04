ref=1
n1=6

for Np in 8 16 32; do
        for Nc in 0 400; do
            for No in 1 3 5; do
                for st in false true; do
                    filename="barrel-st=$st-Nc=$Nc-Np=$Np-No=$No.json"
                    julia conc/shells/barrel.jl  \
                        --filename "$filename" \
                        --stabilize $st \
                        --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No; 
                done
            done
        done 
    done
done


