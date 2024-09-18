prefix="zc-weak-growlin"
n1 = 6
No = 5
nt = n1 * (n1 + 1) / 2 * 6
number_of_clusters(Np) = Int(ceil(Np / nt) * 2)

for ref in 10:20
    Np = Int(round(48 * ref^2 / 5000))
    Nc = number_of_clusters(Np)
    filename = "$(prefix)-ref=$ref-Np=$Np-No=$No.json"
    run(`
    julia --project=. conc/shells/zc.jl --filename "$filename" --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No
    `)
end