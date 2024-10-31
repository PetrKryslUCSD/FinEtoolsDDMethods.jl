prefix="zc-weak-const"
n1 = 6
No = 5
nt = n1 * (n1 + 1) / 2 * 6
number_of_clusters(Np) = Int(252)

for Nepp in [5000, 1000, 200]
    for ref in 20:10:140
        Ne = Int(round(48 * ref^2))
        Np = Int(round(Ne / Nepp))
        Nc = number_of_clusters(Np)
        filename = "$(prefix)-ref=$ref-Np=$Np-No=$No-Nepp=$Nepp.json"
        run(`
        julia --project=. conc/shells/zc_seq_driver.jl --filename "$filename" --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No
        `)
    end
end
