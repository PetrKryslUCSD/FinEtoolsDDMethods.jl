prefix="zc-weak-growlog"
n1=6
No=5

for ref in 4:8
    Np = Int(round(ref^2 / 2))
    Nc = Int(ceil(100 * log(Np / 4)))
    filename = "$(prefix)-ref=$ref-Np=$Np-No=$No.json"
    run(`
    julia --project=. conc/shells/zc.jl --filename "$filename" --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No
    `)
end
