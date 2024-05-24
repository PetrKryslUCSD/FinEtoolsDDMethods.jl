"""
    make_partition_mesh(fens, fes, element_partitioning, partition)

Create a partition mesh.

The partition mesh consists of elements of belonging to the partition (i.e.
`element_partitioning[i] == partition`). It selects from the given finite element nodes
(`fens`) those that are connected by the elements of the partition.

# Arguments
- `fens`: An array of finite element nodes.
- `fes`: An array of finite element structures.
- `element_partitioning`: The element partitioning information.
- `partition`: The partition number.

# Returns
A partition mesh.

- `fens`: An array of finite element nodes.
- `fes`: An array of finite element structures.
- `global_node_numbers`: The global node numbers of the nodes within the mesh.

"""
function make_partition_mesh(fens, fes, element_partitioning, partition)
    pfes = subset(fes, findall(y -> y == partition, element_partitioning))
    connected = findunconnnodes(fens, pfes)
    fens, new_numbering = compactnodes(fens, connected)
    pfes = renumberconn!(pfes, new_numbering)
    global_node_numbers = [i for i in eachindex(new_numbering) if new_numbering[i] != 0]
    return fens, pfes, global_node_numbers
end