prefix="zc-strong-match"
n1 = 6
No = 5
nt = n1 * (n1 + 1) / 2 * 6
ref = 100
Nc = 0

for Np in 8:8:64
    filename = "$(prefix)-ref=$ref-Np=$Np-No=$No-Nc=$Nc.json"
    run(`
    julia --project=. conc/shells/zc.jl --filename "$filename" --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No
    `)
end
