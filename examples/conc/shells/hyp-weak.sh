prefix="hyp-weak"
n1=6
Nc=0
No=5
as=10

for ref in $(seq 4 16) ; do
    Np=$((ref**2/2))
    filename="${prefix}-ref=$ref-as=$as-Np=$Np-No=$No.json"
    julia conc/shells/hyp.jl --filename "$filename" \
    --aspect $as \
    --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No; 
done
