prefix="hyp-weak"
n1=6
Nc=0
No=5
as=10

for ref in 4 8 16 ; do
    Np=$((2*ref**2))
    filename="${prefix}-ref=$ref-as=$as-Np=$Np-No=$No.json"
    julia conc/shells/hyp.jl --filename "$filename" \
    --aspect $as \
    --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No; 
done
