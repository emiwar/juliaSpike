@generated function update(a::NamedTuple{anames, atypes},
                           b::NamedTuple{bnames, btypes}) where {anames, atypes,
                                                                 bnames, btypes}
    vals = Any[ :(getfield($(Base.sym_in(n, bnames) ? :b : :a), $(QuoteNode(n)))) for n in anames ]
    :(NamedTuple{$anames, $atypes}(($(vals...),)))
end


function addneurons!(network::Network, neurontype::Symbol, numNeurons)
    if !(neurontype in keys(network.neurongroups))
        network.neurongroups[neurontype] = CpuNeuronGroup(NeuronType(neurontype))
    end
    addneuronstogroup!(network.neurongroups[neurontype], numNeurons)
end

function addneuronstogroup!(neurongroup::NeuronGroup, numNeurons)
    sym = symName(neurontype(neurongroup))
    [(sym, addneurontogroup!(neurongroup)) for i=1:numNeurons]
end

function connectrule!(network::Network, synapsetype::Symbol, pre, post, rule; kwargs...)
    synapses = Tuple{Tuple{Symbol, Symbol, Symbol}, Tuple{Int32, Int32}}[]
    for i in eachindex(pre), j in eachindex(post)
        if rule(pre[i], post[j])
            push!(synapses, ((synapsetype, pre[i][1], post[j][1]), (pre[i][2], post[j][2])))
            #connect!(network, pre[i], post[j], synapsetype; kwargs...))
        end
    end
    addsynapses!(network, synapses)
end

function addsynapses!(network::Network, synapses)
    perGroup = Dict{NTuple{3, Symbol}, NTuple{2, Vector{Int32}}}()
    for (key, (pre, post)) in synapses
        if !(key in keys(perGroup))
            perGroup[key] = (Int32[], Int32[])
        end
        push!(perGroup[key][1], pre)
        push!(perGroup[key][2], post)
    end
    for key in keys(perGroup)
        if !(key in keys(network.synapsegroups))
            st, pret, postt = key
            network.synapsegroups[key] = CpuSynapseGroup(SynapseType(st), NeuronType(pret), NeuronType(postt))
        end
        pre, post = perGroup[key]
        addsynapsestogroup!(network.synapsegroups[key], pre, post)
    end
end

function addsynapsestogroup!(synapsegroup::SynapseGroup, pre, post)
    for i in eachindex(pre)
        addsynapsetogroup!(synapsegroup, pre[i], post[i])
    end
end

function connectrand!(network::Network, pre, post, synapsetype::Symbol, prob; kwargs...)
    connectrule!(network, synapsetype, pre, post, (i,j)->rand()<0.5; kwargs...)
end