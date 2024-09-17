prefix="hyp-weak-growlin"
n1=6
No=5

for as in [10 100]
    for ref in 4:20
        Np = Int(round(ref^2 / 2))
        Nc = Int(ceil(Np * 2))
        filename="$(prefix)-ref=$ref-as=$as-Np=$Np-No=$No.json"
        run(`
        julia --project=. conc/shells/hyp.jl --filename "$filename" --aspect $as --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No
        `)
    end
end
