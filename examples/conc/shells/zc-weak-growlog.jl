prefix="zc-weak-growlog"
n1 = 6
No = 5
nt = n1 * (n1 + 1) / 2 * 6
Nepp = 5000
number_of_clusters(Np) = Int(ceil(10 * log(Np / 2)))

for ref in 20:10:140
    Ne = Int(round(48 * ref^2))
    Np = Int(round(Ne / Nepp))
    Nc = number_of_clusters(Np)
    filename = "$(prefix)-ref=$ref-Np=$Np-No=$No.json"
    run(`
    julia --project=. conc/shells/zc.jl --filename "$filename" --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No
    `)
end
