prefix="zc-weak-growlin"
n1=6
No=5

for ref in 4:8
    Np = Int(round(4 ^ ref / 32))
    Nc = Int(ceil(Np * 2))
    filename = "$(prefix)-ref=$ref-Np=$Np-No=$No.json"
    run(`
    julia --project=. conc/shells/zc.jl --filename "$filename" --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No
    `)
end
