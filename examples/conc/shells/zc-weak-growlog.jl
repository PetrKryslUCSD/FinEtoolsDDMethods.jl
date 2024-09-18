prefix="zc-weak-growlog"
n1 = 6
No = 5
nt = n1 * (n1 + 1) / 2 * 6
number_of_clusters(Np) = Int(ceil(10 * log10(Np / 4)))

for ref in 4:8
    @show Np = Int(round(4 ^ ref / 32))
    @show Nc = number_of_clusters(Np)
    # filename = "$(prefix)-ref=$ref-Np=$Np-No=$No.json"
    # run(`
    # julia --project=. conc/shells/zc.jl --filename "$filename" --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No
    # `)
end
