prefix="zc-weak-growlin"
n1 = 6
No = 5
nt = n1 * (n1 + 1) / 2 * 6
Nepp = 5000
number_of_clusters(Np) = Int(ceil(Np / 2))

for Nepp in [5000, 1000, 200]
    for ref in 20:10:140
        @show Ne = Int(round(48 * ref^2))
        @show Np = Int(round(Ne / Nepp))
        @show Nc = number_of_clusters(Np)
        filename = "$(prefix)-ref=$ref-Np=$Np-No=$No-Nepp=$Nepp.json"
        n = sqrt(Ne / Np) / 2
        @show nd = (2 * n^2 + 4 * (5 - 1) * 2 * n) * 6
        run(`
        julia --project=. conc/shells/zc.jl --filename "$filename" --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No
        `)
    end
end
