kind = "hex"
prefix = "fib-strong-$(kind)-cni-grow"
n1 = 6
No = 5
nt = n1 * (n1 + 1) * (n1 + 2) / 6 * 3
ref = 6
Ef = 100000.0
nuf = 0.3
Em = 1000.0
num = 0.4999
number_of_clusters(Np) = Int(ceil(Np / 2))

for Np in [64, 128, 256, 512, ]
    filename = "$(prefix)-ref=$ref-Np=$Np-No=$No-Nc=$Nc.json"
    Nc = number_of_clusters(Np)
    run(`
    julia --project=. conc/lindef/fib.jl --filename "$filename" --kind $kind --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No --Ef $Ef --nuf $nuf --Em $Em --num $num
    `)
end
