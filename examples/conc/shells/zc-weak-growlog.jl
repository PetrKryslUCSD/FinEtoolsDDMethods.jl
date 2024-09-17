prefix="zc-weak-growlog"
n1=6
Nc=0
No=5

for ref in 4:8
    Np = Int(round(ref^2 / 2))
    filename = "$(prefix)-ref=$ref-Np=$Np-No=$No.json"
    echo$filename
    run(`
    julia --project=. conc/shells/zc.jl --filename "$filename" --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No
    `)
end
