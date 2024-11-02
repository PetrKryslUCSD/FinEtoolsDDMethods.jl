-- Why does this CG version give different results than the original?

```
# Smith, Gropp  1996
# There is something weird about this: it takes quite a few more iterations!
function pcg_seq_sg(Aop!, b, x0;
    (M!)=(q, p) -> (q .= p),
    itmax=0,
    atol=√eps(eltype(b)),
    rtol=√eps(eltype(b)),
    peeksolution=(iter, x, resnorm) -> nothing
    )
    itmax = (itmax > 0 ? itmax : length(b))
    x = deepcopy(x0); p = similar(x); r = similar(x); z = similar(x); 
    Ap = z # Alias for legibility
    tol = zero(typeof(atol))
    resnorm = Inf
    residuals = typeof(tol)[]
    stats = (niter=itmax, resnorm=resnorm, residuals=residuals)
    betaold = one(typeof(atol))
    Aop!(Ap, x) 
    @. r = b - Ap
    M!(z, r)
    @. p = z
    betaold = dot(z, r)
    iter = 1
    while iter < itmax
        beta = dot(z, r)
        c = beta / betaold; betaold = beta
        @. p = z + c * p
        Aop!(Ap, p)
        @show a = beta / dot(p, Ap)
        @. x += a * p
        @. r -= a * Ap
        resnorm = sqrt(dot(r, r))
        push!(residuals, resnorm)
        peeksolution(iter, x, resnorm)
        tol == 0 && (tol = atol + rtol * sqrt(beta))
        if resnorm < tol
            break
        end
        M!(z, r)
        iter += 1
    end
    stats = (niter=iter, resnorm=resnorm, residuals=residuals)
    return (x, stats)
end
```

-- Measurement for zc on Ookami:

[pkrysl@login2 a64fx]$ tail -f job_zc.log
┌ Info: MPICH Version:      4.2.2
│ MPICH Release date: Wed Jul  3 09:16:22 AM CDT 2024
│ MPICH ABI:          16:2:4
│ MPICH Device:       ch3:nemesis
│ MPICH configure:    --prefix=/workspace/destdir --build=x86_64-linux-musl --host=aarch64-linux-gnu --disable-dependency-tracking --disable-doc --enable-fast=all,O3 --enable-static=no --with-device=ch3 --with-hwloc=/workspace/destdir
│ MPICH CC:           cc    -DNDEBUG -DNVALGRIND -O3
│ MPICH CXX:          c++   -DNDEBUG -DNVALGRIND -O3
│ MPICH F77:          gfortran   -O3
│ MPICH FC:           gfortran   -O3
└ MPICH features:
[ Info: BLAS_THREADS = 2
[ Info: Number of processes: 17
[ Info: Number of partitions: 16
[ Info: Refinement factor: 100
[ Info: Number of fine grid partitions: 16
[ Info: Number of overlaps: 1
[ Info: Number of elements: 480000
[ Info: Number of free dofs = 1445703
[ Info: Create partitioning info (14.745 [s])
[ Info: Mean partition size: 94022
[ Info: Number of clusters (requested): 100
[ Info: Number of 1D basis functions: 6
[ Info: Number of elements per cluster: 4800
[ Info: Number of clusters (actual): 102
[ Info: Size of the reduced problem: 12852
[ Info: Generate clusters (132.499 [s])
[ Info: Create partitions and clusters (141.64 [s])
[ Info: Create global factor (13.652 [s])
┌ Info: Rank 0
│ broadcast p          : 8.359413126999925   <----------
│ A op                 : 0.010473324999793476
│ reduce Ap            : 15.18611645700048   <----------
│ update r             : 0.165569330999233
│ compute zg + update x: 25.9383279890003        <----------
│ compute zl           : 0.010416767000378968
│ reduce zl            : 19.206073070000684   <----------
│ update z, z*r        : 0.30862214499916263
└ bcast resnorm, upda p: 0.11648019400126941
┌ Info: Rank 3
│ broadcast p          : 8.451745567999296   <----------
│ A op                 : 0.28443736800022634
│ reduce Ap            : 12.317726576000268   <----------
│ update r             : 3.361099970788928e-5
│ compute zg + update x: 28.70205673100122   <----------
│ compute zl           : 4.304917771999499    <----------
│ reduce zl            : 12.460360866000201   <----------
│ update z, z*r        : 2.710099965952395e-5
└ bcast resnorm, upda p: 2.7775708919991757
┌ Info: Rank 2
│ broadcast p          : 8.451490443999546
│ A op                 : 0.3059160139994219
│ reduce Ap            : 14.521160037000072
│ update r             : 2.4619999749120325e-5
│ compute zg + update x: 26.47890883500054
│ compute zl           : 4.948404821999702
│ reduce zl            : 13.891613564000181
│ update z, z*r        : 2.507999988665688e-5
└ bcast resnorm, upda p: 0.7013506180005606
┌ Info: Rank 16
│ broadcast p          : 8.467566131000467
│ A op                 : 0.28322105200004444
│ reduce Ap            : 11.294734402999893
│ update r             : 2.938000011454278e-5
│ compute zg + update x: 29.721708211999612
│ compute zl           : 4.029492041999674
│ reduce zl            : 11.567765707000717
│ update z, z*r        : 2.4459999622195028e-5
└ bcast resnorm, upda p: 3.9342935060003583
┌ Info: Rank 12
│ broadcast p          : 8.42845623199969
│ A op                 : 0.29636884399974406
│ reduce Ap            : 11.166308822000246
│ update r             : 2.4212000653278665e-5
│ compute zg + update x: 29.827398560998972
│ compute zl           : 4.510415577000003
│ reduce zl            : 11.042511926999623
│ update z, z*r        : 2.432000019325642e-5
└ bcast resnorm, upda p: 4.027326298000162
┌ Info: Rank 1
│ broadcast p          : 8.452004283000178
│ A op                 : 0.28702923699938765
│ reduce Ap            : 3.51097196699925
│ update r             : 2.1250000145300874e-5
│ compute zg + update x: 37.50659476300143
│ compute zl           : 4.241695161999132
│ reduce zl            : 3.5152083230007065
│ update z, z*r        : 1.7740000657795463e-5
└ bcast resnorm, upda p: 11.785393448999002
┌ Info: Rank 10
│ broadcast p          : 8.39366340500078
│ A op                 : 0.29173354299973653
│ reduce Ap            : 10.772513891999324
│ update r             : 2.803999996103812e-5
│ compute zg + update x: 30.21561660999987
│ compute zl           : 4.558787576000441
│ reduce zl            : 10.53298569800063
│ update z, z*r        : 2.8320999490460963e-5
└ bcast resnorm, upda p: 4.533455858999332
┌ Info: Rank 6
│ broadcast p          : 8.514723764001019
│ A op                 : 0.27336156799879063
│ reduce Ap            : 11.309494974000472
│ update r             : 2.6780000325743458e-5
│ compute zg + update x: 29.55283356000018
│ compute zl           : 3.8661544710003
│ reduce zl            : 11.893959697999662
│ update z, z*r        : 3.634099880400754e-5
└ bcast resnorm, upda p: 3.888294048000944
┌ Info: Rank 7
│ broadcast p          : 8.394113196000035
│ A op                 : 0.2871877170005064
│ reduce Ap            : 11.494190316999493
│ update r             : 3.5289999232190894e-5
│ compute zg + update x: 29.480840330000547
│ compute zl           : 4.4886128440004995
│ reduce zl            : 11.496646873998543
│ update z, z*r        : 2.8510001584436395e-5
└ bcast resnorm, upda p: 3.657243000998733
┌ Info: Rank 14
│ broadcast p          : 8.45581944300011
│ A op                 : 0.3553533470003458
│ reduce Ap            : 10.902298709000206
│ update r             : 2.5230000346709858e-5
│ compute zg + update x: 30.042683373998898
│ compute zl           : 5.2544969759997
│ reduce zl            : 10.020274711000638
│ update z, z*r        : 2.6160000061281607e-5
└ bcast resnorm, upda p: 4.2677146880002965
┌ Info: Rank 8
│ broadcast p          : 8.401846918000729
│ A op                 : 0.31304289699937726
│ reduce Ap            : 11.709068893000449
│ update r             : 3.396100032659888e-5
│ compute zg + update x: 29.246754467999608
│ compute zl           : 4.957545548000098
│ reduce zl            : 11.118104873999755
│ update z, z*r        : 2.5650999987192336e-5
└ bcast resnorm, upda p: 3.552426128999741
┌ Info: Rank 4
│ broadcast p          : 8.449401064000085
│ A op                 : 0.31405232700103625
│ reduce Ap            : 12.61787359699997
│ update r             : 2.4619999294372974e-5
│ compute zg + update x: 28.36654504199987
│ compute zl           : 5.40323115900037
│ reduce zl            : 11.549885446999951
│ update z, z*r        : 2.5090999997701147e-5
└ bcast resnorm, upda p: 2.5978318670001954
┌ Info: Rank 13
│ broadcast p          : 8.459874372000058
│ A op                 : 0.2952233339997292
│ reduce Ap            : 10.67731494800114
│ update r             : 2.8451999014578178e-5
│ compute zg + update x: 30.32488003400067
│ compute zl           : 4.285044771000003
│ reduce zl            : 10.876703741000028
│ update z, z*r        : 4.01999993755453e-5
└ bcast resnorm, upda p: 4.379742716999772
┌ Info: Rank 15
│ broadcast p          : 8.461142273999712
│ A op                 : 0.2749288190002517
│ reduce Ap            : 11.04065923799908
│ update r             : 2.6310000521334587e-5
│ compute zg + update x: 29.985127745000227
│ compute zl           : 3.7129389129995616
│ reduce zl            : 11.7657088440003
│ update z, z*r        : 2.636999965943687e-5
└ bcast resnorm, upda p: 4.05830885499995
┌ Info: Rank 9
│ broadcast p          : 8.400377196000363
│ A op                 : 0.28679769799919086
│ reduce Ap            : 10.492663494001135
│ update r             : 2.622999886625621e-5
│ compute zg + update x: 30.48994489100005
│ compute zl           : 4.46202209900116
│ reduce zl            : 10.531536759999426
│ update z, z*r        : 2.3570000166728278e-5
└ bcast resnorm, upda p: 4.635485562000667
┌ Info: Rank 11
│ broadcast p          : 8.40756735200057
│ A op                 : 0.2920467159999589
│ reduce Ap            : 10.829544455000587
│ update r             : 2.650099850143306e-5
│ compute zg + update x: 30.161460914001054
│ compute zl           : 4.310158131999742
│ reduce zl            : 10.993201146000047
│ update z, z*r        : 2.9919999860794633e-5
└ bcast resnorm, upda p: 4.304020672999968
┌ Info: Rank 5
│ broadcast p          : 8.487063083000066
│ A op                 : 0.28236597100021754
│ reduce Ap            : 11.040800140000101
│ update r             : 2.718999917306064e-5
│ compute zg + update x: 29.989978913001096
│ compute zl           : 4.294101121998892
│ reduce zl            : 11.197428629000115
│ update z, z*r        : 2.6951000336339348e-5
└ bcast resnorm, upda p: 4.007080283999585
[ Info: Number of iterations:  39
[ Info: Iterations (78.8 [s])
[ Info: Storing data in zc--ref=100-Nc=102-n1=6-Np=16-No=1.json
[ Info: Total time: 277.922 [s]

-- Combined local and global pre conditioners into a single routine so that communication and computation can be overlapped

-- The resolved vector needs always to be zeroed out. Even for rank != 0. Why?

-- The global pre conditioner should use in-place matrix multiplications. Done.

-- Extrae MPI tracing on Ookami.

Use spack to install   `extrae`.

Set up the environment.
```
. /lustre/home/pkrysl/a64fx/spack/share/spack/setup-env.sh
spack load extrae
```


Find `libmpitrace.so`.

Install Extrae.jl
```
using Extrae
Extrae.use_system_binary(library_names=["/lustre/home/pkrysl/a64fx/spack/opt/spack/linux-rocky8-thunderx2/gcc-13.1.0/extrae-4.1.2-evpjedfg336em5at377f2g64yzrfc6np/lib/libmpitrace"], export_prefs=true)
using Pkg; Pkg.build("Extrae");  
```

Augment LocalPreferences.toml with a preload of the MPI tracing library:
```
[MPIPreferences]
__clear__ = ["libmpi", "abi", "mpiexec", "cclibs", "preloads_env_switch"]
_format = "1.0"
binary = "MPICH_jll"
preloads = ["/lustre/home/pkrysl/a64fx/spack/opt/spack/linux-rocky8-thunderx2/gcc-13.1.0/extrae-4.1.2-evpjedfg336em5at377f2g64yzrfc6np/lib/libmpitrace"]
```

Set up `extrae.xml`:
```
<?xml version='1.0'?>

<trace enabled="yes"
  home="@sed_MYPREFIXDIR@"
  initial-mode="detail"
  type="paraver"
>
  <mpi enabled="yes">
    <counters enabled="yes" />
  </mpi>

  <pthread enabled="no">
    <locks enabled="no" />
    <counters enabled="yes" />
  </pthread>

  <openmp enabled="no"></openmp>

  <callers enabled="yes">
    <mpi enabled="no">1-3</mpi>
    <sampling enabled="no">1-5</sampling>
  </callers>

  <user-functions enabled="no"
    list="/home/bsc41/bsc41273/user-functions.dat"
    exclude-automatic-functions="no">
    <counters enabled="yes" />
  </user-functions>

  <counters enabled="yes">
    <cpu enabled="yes" starting-set-distribution="1">
      <set enabled="yes" domain="all" changeat-globalops="5">
        PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_L1_DCM,PAPI_BR_CN,PAPI_BR_MSP,PAPI_VEC_INS
        <sampling enabled="no" period="100000000">PAPI_TOT_CYC</sampling>
      </set>
      <set enabled="no" domain="user" changeat-globalops="5">
        PAPI_TOT_INS,PAPI_FP_INS,PAPI_TOT_CYC
      </set>
    </cpu>
    <network enabled="no" />
    <resource-usage enabled="no" />
  </counters>

  <storage enabled="no">
    <trace-prefix enabled="yes">TRACE</trace-prefix>
    <size enabled="no">5</size>
    <temporal-directory enabled="yes">/scratch</temporal-directory>
    <final-directory enabled="yes">/gpfs/scratch/bsc41/bsc41273</final-directory>
  </storage>

  <buffer enabled="yes">
    <size enabled="yes">150000</size>
    <circular enabled="no" />
  </buffer>

  <trace-control enabled="no">
    <file enabled="no" frequency="5M">/gpfs/scratch/bsc41/bsc41273/control</file>
    <global-ops enabled="no">10</global-ops>
    <remote-control enabled="yes">
      <mrnet enabled="yes" target="150" analysis="spectral" start-after="30">
        <clustering max_tasks="26" max_points="8000"/>
        <spectral min_seen="1" max_periods="0" num_iters="3" signals="DurBurst,InMPI"/>
      </mrnet>
    </remote-control>
  </trace-control> 

  <others enabled="no">
    <minimum-time enabled="no">10M</minimum-time>
    <finalize-on-signal enabled="yes" 
      SIGUSR1="no" SIGUSR2="no" SIGINT="yes"
      SIGQUIT="yes" SIGTERM="yes" SIGXCPU="yes"
      SIGFPE="yes" SIGSEGV="yes" SIGABRT="yes"
    />
    <flush-sampling-buffer-at-instrumentation-point enabled="yes" />
  </others>

  <bursts enabled="no">
    <threshold enabled="yes">500u</threshold>
    <mpi-statistics enabled="yes" />
  </bursts>

  <sampling enabled="yes" type="default" period="1m" variability="200u"/>

  <opencl enabled="no" />

  <openacc enabled="no" />

  <cuda enabled="no" />

  <!-- Beware: High overhead! -->
  <cpu-events enabled="no" frequency="0" emit-always="no" poi="openmp"/>

  <merge enabled="yes" 
    synchronization="default"
    binary="julia"
    tree-fan-out="16"
    max-memory="512"
    joint-states="yes"
    keep-mpits="yes"
    sort-addresses="yes"
    overwrite="yes"
  >
    julia-trace.prv 
  </merge>

</trace>

```

Augment the batch file:
```
#!/usr/bin/env bash

#SBATCH --job-name=job_zc
#SBATCH --output=job_zc_Np=3.log
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=3
#SBATCH --time=00:15:00
#SBATCH -p short

# specify message size threshold for using the UCX Rendevous Protocol
#export UCX_RNDV_THRESH=65536

# use high-performance rc transports where possible
#export UCX_TLS=rc

# Load OpenMPI and Julia
export JULIA_DEPOT_PATH="~/a64fx/depot"
export JULIA_DEBUG=Extrae
module load julia
module load gcc/13.1.0
module load slurm
module load openmpi/gcc8/4.1.2

# Automatically set the number of Julia threads depending on number of Slurm threads
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export BLAS_THREADS=2
export EXTRAE_ON=1
export EXTRAE_SKIP_AUTO_LIBRARY_INITIALIZE=1
export EXTRAE_CONFIG_FILE=/lustre/home/pkrysl/a64fx/extrae.xml

cd FinEtoolsDDMethods.jl/examples
/lustre/home/pkrysl/a64fx/depot/bin/mpiexecjl julia --project=. conc/shells/zc_mpi_driver.jl --n1 6 --Nc 100 --ref 100
```

Instrument the Script File:

```
using Extrae
...
Extrae.init()
...
Extrae.finish()
```

-- Why does the first call take so much longer?
partition_mult!(q, cpi, comm, rank, partition, timers, p) (for MPI)
I put in a check that resets the timers erasing the recording of the first call, 
so that the average during iteration matches the iteration timer.
Answer: It could be compilation.

-- Alternative data distribution strategy:
```julia
initial residual is distributed to partitions
initial solution is distributed to partitions
while iter < itmax
    Aop!(Ap, p) # p is held by partitions
    alpha = rhoold / dot(p, Ap) # computed on partitions, plus reduction
    @. r -= alpha * Ap # update on partitions
    M!(z, r) # ???
    @. x += alpha * p # update on partitions
    if rank == 0
        rho = dot(z, r) # ???
        beta = rho / rhoold;   rhoold = rho
        if normtype == KSP_NORM_UNPRECONDITIONED
            resnorm[] = sqrt(dot(r, r))
        else
            resnorm[] = sqrt(rho) 
        end
    end
    @. p = z + beta * p # update on partitions
    push!(residuals, resnorm[])
    rank == 0 && peeksolution(iter, x, resnorm[])
    if resnorm[] < tol
        break
    end
    iter += 1
end
```
So the key is the preconditioner. The partition could 
transform the residual, all-broadcast to the other partitions.
All partitions could solve the global problem concurrently.
The vector z could then also be held only locally.

-- How should the global solver be structured to avoid all-reduce?
  Perhaps if I cannot avoid it, it might be overlapped with computation?

-- I think I need to allow for rank to be zero-based. Done.

"terminal.integrated.defaultProfile.windows": "Git Bash"

-- Could one do the global solve only every few iterations?

-- Is there any penalty involved in waiting for Isend? Could we use only a "free" instead? Done: free is used for the sends. 

-- Could this pattern be beneficial?
```julia
while true
  other = MPI.Waitany(requests)
  if other === nothing
    break
  end
  n = length(ldofs_other[other])
  bs.xt[ldofs_other[other]] .= bs.recv[other][1:n]
end
```
Done. Replaced all waitalls.