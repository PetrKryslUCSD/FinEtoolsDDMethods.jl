prefix="zc-strong-const"
n1 = 6
No = 5
nt = n1 * (n1 + 1) / 2 * 6
ref = 100
Nc = 250

for Np in 8:8:64
    filename = "$(prefix)-ref=$ref-Np=$Np-No=$No-Nc=$Nc.json"
    run(`
    julia --project=. conc/shells/zc_seq_driver.jl --filename "$filename" --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No
    `)
end
