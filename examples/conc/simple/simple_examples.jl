module simple_examples

using FinEtools
using FinEtools.MeshExportModule: VTK
using FinEtoolsDeforLinear
using FinEtoolsFlexStructures.FESetShellT3Module: FESetShellT3
using FinEtoolsFlexStructures.FESetShellQ4Module: FESetShellQ4
using FinEtoolsFlexStructures.FEMMShellT3FFModule
using FinEtoolsFlexStructures.RotUtilModule: initial_Rfield, update_rotation_field!
using FinEtoolsDDMethods
using FinEtoolsDDMethods.CGModule: pcg
using FinEtoolsDDMethods.PartitionCoNCModule: CoNCPartitioningInfo, npartitions
using FinEtoolsDDMethods.DDCoNCMPIModule: DDCoNCMPIComm, PartitionedVector, aop!, TwoLevelPreConditioner, rhs
using FinEtoolsDDMethods.DDCoNCMPIModule: vec_collect, vec_copyto!
using FinEtoolsDDMethods.CoNCUtilitiesModule: patch_coordinates
using SymRCM
using Metis
using Test
using LinearAlgebra
using SparseArrays
using PlotlyLight
using Krylov
using LinearOperators
import Base: size, eltype
import LinearAlgebra: mul!
import CoNCMOR: CoNCData, transfmatrix, LegendreBasis
using Targe2
using DataDrop
using Statistics
using ShellStructureTopo: create_partitions

function _execute(filename, ref, Nc, n1, Np, No, itmax, relrestol, peek, visualize)
    # Parameters:
    E = 210e9
    nu = 0.3
    L = 10.0
    CTE = 0.0
    thickness = 0.1
    tolerance = thickness / 1000
    
    fens, fes = Q4quadrilateral([-1 -1; 1 1], 3, 3) 
    
    MR = DeforModelRed3D
    mater = MatDeforElastIso(MR, 0.0, E, nu, CTE)
    

    # Construct the requisite fields, geometry and displacement
    # Initialize configuration variables
    geom0 = NodalField(fens.xyz)
    u0 = NodalField(zeros(size(fens.xyz,1), 3))
    Rfield0 = initial_Rfield(fens)
    dchi = NodalField(zeros(size(fens.xyz,1), 1))

    numberdofs!(dchi);

    fr = dofrange(dchi, DOF_KIND_FREE)
    dr = dofrange(dchi, DOF_KIND_DATA)
    
    t1 = time()
    cpi = CoNCPartitioningInfo(fens, fes, Np, No, dchi, [1, 1, 1, 2, 1, 1, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2]) 
    for el in eachindex(cpi.list_of_entity_lists)
        @show cpi.list_of_entity_lists[el].own.nodes
        @show cpi.list_of_entity_lists[el].own.elements
        @show cpi.list_of_entity_lists[el].own.receive_nodes
        @show cpi.list_of_entity_lists[el].own.send_nodes

        @show cpi.list_of_entity_lists[el].extended
        @show cpi.list_of_entity_lists[el].extended
    end
    
    
    true
end

using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--filename"
        help = "Use filename to name the output files"
        arg_type = String
        default = ""
        "--Nc"
        help = "Number of clusters"
        arg_type = Int
        default = 2
        "--n1"
        help = "Number 1D basis functions"
        arg_type = Int
        default = 5
        "--No"
        help = "Number of overlaps"
        arg_type = Int
        default = 5
        "--Np"
        help = "Number of partitions"
        arg_type = Int
        default = 7
        "--ref"
        help = "Refinement factor"
        arg_type = Int
        default = 7
        "--Nepp"
        help = "Number of elements per partition"
        arg_type = Int
        default = 0
        "--itmax"
        help = "Maximum number of iterations allowed"
        arg_type = Int
        default = 200
        "--relrestol"
        help = "Relative residual tolerance"
        arg_type = Float64
        default = 1.0e-6
        "--peek"
        help = "Peek at the iterations?"
        arg_type = Bool
        default = true
        "--visualize"
        help = "Write out visualization files?"
        arg_type = Bool
        default = false
    end
    return parse_args(s)
end

p = parse_commandline()

ref = p["ref"]
Nepp = p["Nepp"]    
Np = p["Np"]
if Nepp == 0 && ref == 0
    error("Either ref or Nepp must be specified")
end
if ref == 0
    ref = Int(ceil(sqrt((Np * Nepp / 48))))
end

_execute(
    p["filename"],
    ref,
    p["Nc"], 
    p["n1"],
    Np,
    p["No"], 
    p["itmax"], 
    p["relrestol"],
    p["peek"],
    p["visualize"]
    )


nothing
end # module


