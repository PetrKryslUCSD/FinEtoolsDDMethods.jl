prefix="bb-weak-grow"
n1=6
Nc=0
No=5
# for No in 5 ; do
#     for N in 2 4 8 ; do
#         for Np in 4 8 16  ; do
#             filename="${prefix}-N=$N-Np=$Np-No=$No.json"
#             echo $filename
#             julia conc/lindef/bb.jl --filename "$filename" \
#                 --N $N --Nc $Nc --n1 $n1 --Np $Np --No $No; 
#         done
#     done
# done
for N in $(seq 2 9) ; do
    Np=$((5*9*19*N**3/3420)) # (6840/2)
    filename="${prefix}-N=$N.json"
    echo "$N $Np $filename "
    julia conc/lindef/bb.jl --filename "$filename" \
                    --N $N --Nc $Nc --n1 $n1 --Np $Np --No $No; 
done