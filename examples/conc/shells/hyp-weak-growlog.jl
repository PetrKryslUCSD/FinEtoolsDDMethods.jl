prefix="hyp-weak-grow"
n1=6
No=5

for as in [10 100]
    for ref in 4:5
        Np = Int(round(ref^2 / 2))
        Nc = Int(ceil(log(10*Np)))
        @info "$Np $(log(10*Np)) $Nc"
        filename="$(prefix)-ref=$ref-as=$as-Np=$Np-No=$No.json"
        # "bash julia --project=. hyp.jl --filename "$filename" --aspect $as --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No"
        run(`
        julia --project=. conc/shells/hyp.jl --filename "$filename" --aspect $as --ref $ref --Nc $Nc --n1 $n1 --Np $Np --No $No
        `)
    end
end
