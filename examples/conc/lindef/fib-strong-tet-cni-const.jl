kind = "tet"
prefix = "fib-strong-$(kind)-cni-const"
n1 = 6
No = 5
nt = n1 * (n1 + 1) * (n1 + 2) / 6 * 3
ref = 4
Nc = 100
Ef = 100000.0
nuf = 0.3
Em = 1000.0
num = 0.4999

for Np in [64, 128, 256, 512, ]
    filename = "$(prefix)-ref=$ref-Np=$Np-No=$No-Nc=$Nc.json"
    run(`
    julia --project=. conc/lindef/fib.jl --filename "$filename" --kind $kind --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No --Ef $Ef --nuf $nuf --Em $Em --num $num
    `)
end
