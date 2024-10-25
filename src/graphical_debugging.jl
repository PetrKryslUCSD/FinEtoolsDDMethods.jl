using FinEtools

function visualize_topology(filename, fens, fes, list_of_entity_lists)
    for p in eachindex(list_of_entity_lists)
        el = list_of_entity_lists[p]
        vtkexportmesh("$(filename)-$(p)-non-shared-elements.vtk", fens, subset(fes, el.nonshared.elements))
        sfes = FESetP1(reshape(el.nonshared.nodes, length(el.nonshared.nodes), 1))
        vtkexportmesh("$(filename)-$(p)-non-shared-nodes.vtk", fens, sfes)
        vtkexportmesh("$(filename)-$(p)-extended-elements.vtk", fens, subset(fes, elextended.elements))
        sfes = FESetP1(reshape(elextended.nodes, length(elextended.nodes), 1))
        vtkexportmesh("$(filename)-$(p)-extended-nodes.vtk", fens, sfes)
        receive_nodes = el.nonshared.receive_nodes
        for i in eachindex(receive_nodes)
            if length(receive_nodes[i]) > 0
                sfes = FESetP1(reshape(receive_nodes[i], length(receive_nodes[i]), 1))
                vtkexportmesh("$(filename)-$(p)-nonshared-receive-$(i).vtk", fens, sfes)
            end
        end
        send_nodes = el.nonshared.send_nodes
        for i in eachindex(send_nodes)
            if length(send_nodes[i]) > 0
                sfes = FESetP1(reshape(send_nodes[i], length(send_nodes[i]), 1))
                vtkexportmesh("$(filename)-$(p)-nonshared-send-$(i).vtk", fens, sfes)
            end
        end
        receive_nodes = elextended.receive_nodes
        for i in eachindex(receive_nodes)
            if length(receive_nodes[i]) > 0
                sfes = FESetP1(reshape(receive_nodes[i], length(receive_nodes[i]), 1))
                vtkexportmesh("$(filename)-$(p)-extended-receive-$(i).vtk", fens, sfes)
            end
        end
        send_nodes = elextended.send_nodes
        for i in eachindex(send_nodes)
            if length(send_nodes[i]) > 0
                sfes = FESetP1(reshape(send_nodes[i], length(send_nodes[i]), 1))
                vtkexportmesh("$(filename)-$(p)-extended-send-$(i).vtk", fens, sfes)
            end
        end
    end
end
