struct NeuronType{S} end
NeuronType(S::Symbol) = NeuronType{S}()

struct SynapseType{S} end
SynapseType(S::Symbol) = SynapseType{S}()

needs_prespike(SynapseType) = true
needs_prestatus(SynapseType) = false
needs_poststatus(SynapseType) = false

abstract type NeuronGroup end
abstract type SynapseGroup end

struct CpuNeuronGroup{neurontype <: NeuronType, VarType, ParamType} <: NeuronGroup
    vars::Vector{VarType}
    params::Vector{ParamType}
    inputCurrent::Vector{Float32}
    spiked::Vector{Bool}
end
function CpuNeuronGroup(neurontype::NeuronType)
    vars = variables(neurontype)
    params = parameters(neurontype)
    CpuNeuronGroup{typeof(neurontype), typeof(vars), typeof(params)}(typeof(vars)[], typeof(params)[], Float32[], Bool[])
end

mutable struct CpuSynapseGroup{synapsetype<:SynapseType, PreType <: NeuronType,
                            PostType <: NeuronType, VarType, ParamType} <: SynapseGroup
    vars::Vector{VarType}
    params::Vector{ParamType}
    pre::Vector{Int32}
    post::Vector{Int32}
    sorted::Bool
end
function CpuSynapseGroup(synapsetype::SynapseType, pretype::NeuronType, posttype::NeuronType)
    vars = variables(synapsetype)
    params = parameters(synapsetype)
    CpuSynapseGroup{typeof(synapsetype), typeof(pretype), typeof(posttype),
                   typeof(vars), typeof(params)}(typeof(vars)[], typeof(params)[], Int32[], Int32[], true)
end

struct GpuNeuronGroup{neurontype<:NeuronType, VarType, ParamType} <: NeuronGroup
    vars::CuVector{VarType}
    params::CuVector{ParamType}
    inputCurrent::CuVector{Float32}
    spiked::CuVector{Bool}
end
struct DeviceNeuronGroup{neurontype<:NeuronType, VT, PT, IT, ST} <: NeuronGroup
    vars::VT
    params::PT
    inputCurrent::IT
    spiked::ST
end
function Adapt.adapt(to, x::GpuNeuronGroup)
    vars = adapt(to, x.vars)
    params = adapt(to, x.params)
    inputCurrent = adapt(to, x.inputCurrent)
    spiked = adapt(to, x.spiked)
    DeviceNeuronGroup{neurontype(x), typeof(vars), typeof(params), typeof(inputCurrent), typeof(spiked)}(vars, params, inputCurrent, spiked)
end

mutable struct GpuSynapseGroup{synapsetype<:SynapseType, PreType <: NeuronType,
                            PostType <: NeuronType, VarType, ParamType} <: SynapseGroup
    vars::CuVector{VarType}
    params::CuVector{ParamType}
    pre::CuVector{Int32}
    post::CuVector{Int32}
    sorted::Bool
end

struct DeviceSynapseGroup{synapsetype<:SynapseType, PreType <: NeuronType,
                          PostType <: NeuronType, VT, PT, preT, postT} <: SynapseGroup
    vars::VT
    params::PT
    pre::preT
    post::postT
end
function Adapt.adapt(to, x::GpuSynapseGroup)
    vars = adapt(to, x.vars)
    params = adapt(to, x.params)
    pre = adapt(to, x.pre)
    post = adapt(to, x.post)
    DeviceSynapseGroup{synapsetype(x), pretype(x), posttype(x), typeof(vars), typeof(params), typeof(pre), typeof(post)}(vars, params, pre, post)
end

struct Network
    neurongroups::Dict{Symbol, NeuronGroup}
    synapsegroups::Dict{NTuple{3, Symbol}, SynapseGroup}
    globalparams::NamedTuple
end
Network(;kwargs...) = Network(Dict{Symbol, NeuronGroup}(),
                              Dict{NTuple{3, Symbol}, SynapseGroup}(), (;kwargs...))

symName(::NeuronType{S}) where {S} = S
symName(::SynapseType{S}) where {S} = S
symName(::Type{NeuronType{S}}) where {S} = S
symName(::Type{SynapseType{S}}) where {S} = S
neurontype(::CpuNeuronGroup{NT}) where {NT} = NT
neurontype(::GpuNeuronGroup{NT}) where {NT} = NT
neurontype(::DeviceNeuronGroup{NT}) where {NT} = NT
synapsetype(::CpuSynapseGroup{ST}) where {ST} = ST
synapsetype(::GpuSynapseGroup{ST}) where {ST} = ST
synapsetype(::DeviceSynapseGroup{ST}) where {ST} = ST
pretype(::CpuSynapseGroup{ST, preT}) where {ST, preT} = preT
pretype(::GpuSynapseGroup{ST, preT}) where {ST, preT} = preT
posttype(::CpuSynapseGroup{ST, preT, postT}) where {ST, preT, postT} = postT
posttype(::GpuSynapseGroup{ST, preT, postT}) where {ST, preT, postT} = postT
preName(::CpuSynapseGroup{ST, preT}) where {ST, preT} = symName(preT)
preName(::GpuSynapseGroup{ST, preT}) where {ST, preT} = symName(preT)
postName(::CpuSynapseGroup{ST, preT, postT}) where {ST, preT, postT} = symName(postT)
postName(::GpuSynapseGroup{ST, preT, postT}) where {ST, preT, postT} = symName(postT)

function todevice(network::Network)
    device_network = Network(;network.globalparams...)
    for (k, v) in pairs(network.neurongroups)
        device_network.neurongroups[k] = GpuNeuronGroup{neurontype(v), eltype(v.vars), eltype(v.params)}(
                                            CuArray(v.vars), CuArray(v.params),
                                            CuArray(v.inputCurrent), CuArray(v.spiked))
    end
    for (k, v) in pairs(network.synapsegroups)
        order = sortperm(v.post)
        device_network.synapsegroups[k]= GpuSynapseGroup{synapsetype(v), pretype(v), posttype(v),
                                                         eltype(v.vars), eltype(v.params)}(
                                            CuArray(v.vars[order]), CuArray(v.params[order]),
                                            CuArray(v.pre[order]), CuArray(v.post[order]), true)
    end
    return device_network
end

function addneuron!(network::Network, neurontype::Symbol; kwargs...)
    if !(neurontype in keys(network.neurongroups))
        network.neurongroups[neurontype] = CpuNeuronGroup(NeuronType(neurontype))
    end
    neuronId = addneurontogroup!(network.neurongroups[neurontype]; kwargs...)
    return (neurontype, Int32(length(network.neurongroups[neurontype].vars)))
end

function addneurontogroup!(neurongroup::NeuronGroup)#; kwargs...)
    v = variables(neurontype(neurongroup)())#update(variables(neurontype(neurongroup)()), NamedTuple())#(;kwargs...))
    p = parameters(neurontype(neurongroup)())#update(parameters(neurontype(neurongroup)()), NamedTuple())
    push!(neurongroup.vars, v)
    push!(neurongroup.params, p)
    push!(neurongroup.inputCurrent, 0.0f0)
    push!(neurongroup.spiked, false)
    return Int32(length(neurongroup.vars))
end

function connect!(network::Network, pre, post, synapsetype::Symbol; kwargs...)
    preType, preId = pre
    postType, postId = post
    groupkey = (synapsetype, preType, postType)
    
    if !(synapsetype in keys(network.synapsegroups))
        network.synapsegroups[groupkey] = CpuSynapseGroup(SynapseType(synapsetype),
                                                          NeuronType(preType),
                                                          NeuronType(postType))
    end
    synId = addsynapsetogroup!(network.synapsegroups[groupkey], preId, postId; kwargs...)
    return (groupkey, synId)
end

function addsynapsetogroup!(synapsegroup::SynapseGroup, preId, postId; kwargs...)
    v = update(variables(synapsetype(synapsegroup)()), (;kwargs...))
    p = update(parameters(synapsetype(synapsegroup)()), (;kwargs...))
    push!(synapsegroup.vars, v)
    push!(synapsegroup.params, p)
    push!(synapsegroup.pre, preId)
    push!(synapsegroup.post, postId)
    synapsegroup.sorted = false
    return Int32(length(synapsegroup.vars))
end

function step!(network::Network)
    for neurongroup in values(network.neurongroups)
        fill!(neurongroup.inputCurrent, 0.0f0)
    end
    CUDA.synchronize()
    for synapsegroup in values(network.synapsegroups)
        stepSynapses!(synapsegroup, network.neurongroups[preName(synapsegroup)],
                      network.neurongroups[postName(synapsegroup)], network.globalparams)
    end
    CUDA.synchronize()
    for neurongroup in values(network.neurongroups)
        fill!(neurongroup.spiked, false)
        stepNeurons!(neurongroup, network.globalparams)
    end
    CUDA.synchronize()
end

function stepNeurons!(neurongroup::GpuNeuronGroup, globalparams)
    nBlocks = UInt32(ceil(length(neurongroup.vars) / 256))
    @cuda blocks=nBlocks threads=256 neuronKernel!(neurongroup, globalparams)
end

function neuronKernel!(neurongroup, globalparams)
    id = blockDim().x * (blockIdx().x-1) + threadIdx().x
    if id <= length(neurongroup.vars)
        stepNeuron!(neurongroup, globalparams, id)
    end
    return nothing
end

function stepNeuron!(neurongroup, globalparams, id)
    @inbounds vars = neurongroup.vars[id]
    @inbounds params = neurongroup.params[id]
    @inbounds I = neurongroup.inputCurrent[id]
    consts = constants(neurontype(neurongroup)())
    stepArgs = merge(vars, params, consts, (;I=I), globalparams)
    newVars, spiked = step(neurontype(neurongroup)(); stepArgs...)
    @inbounds neurongroup.vars[id] = newVars
    @inbounds neurongroup.spiked[id] = spiked
end

function stepSynapses!(synapsegroup::GpuSynapseGroup,
                       pregroup::GpuNeuronGroup,
                       postgroup::GpuNeuronGroup,
                       globalparams)
    synapsegroup.sorted || error("Synapses must be sorted on the GPU.")
    nBlocks = UInt32(ceil(length(synapsegroup.vars) / 256))
    @cuda blocks=nBlocks threads=256 synapseKernel!(synapsegroup, pregroup, postgroup, globalparams)
end

function synapseKernel!(synapsegroup, pregroup, postgroup, globalparams)
    id = blockDim().x * (blockIdx().x-1) + threadIdx().x
    if id > length(synapsegroup.vars)
        return nothing
    end
    
    #Load data
    @inbounds vars = synapsegroup.vars[id]
    @inbounds params = synapsegroup.params[id]
    consts = constants(synapsetype(synapsegroup)())
    @inbounds pre = synapsegroup.pre[id]
    @inbounds post = synapsegroup.post[id]
    @inbounds prespiked = pregroup.spiked[pre]
    stepargs = merge(vars, params, consts, (;prespiked=prespiked), globalparams)

    #Simulating synapse
    newvars = step(synapsetype(synapsegroup)(); stepargs...)
    @inbounds synapsegroup.vars[id] = newvars
    #0.9s
        
    #Gather currents
    curr = current(synapsetype(synapsegroup)(); merge(stepargs, newvars)...)
    
    #2.3s
    if blockIdx().x == gridDim().x
        maxtid = length(synapsegroup.vars) - blockDim().x * (blockIdx().x-1)
    else
        maxtid = 256
    end
    firstingroup, curr_contribution = gatherCurrents(curr, post, maxtid)
    if firstingroup
        CUDA.atomic_add!(pointer(postgroup.inputCurrent, post), curr_contribution)
    end
    
    return nothing
end

function gatherCurrents(curr, post, maxtid)
    tid = threadIdx().x
    
    post_thread = @cuStaticSharedMem(Int32, 256)
    curr_thread = @cuStaticSharedMem(Float32, 256)

    @inbounds post_thread[tid] = post
    @inbounds curr_thread[tid] = curr
    #0.9s
    offset = Int32(1)
    while offset < 256
        sync_threads()
        other = tid + offset
        if other <= maxtid
            samePost = @inbounds post_thread[other] == post
            @inbounds curr_thread[tid] += curr_thread[other]*samePost
        end
        offset *= 2
    end
    firstingroup = tid == 1 || @inbounds post_thread[tid - 1] != post
    curr_contribution = @inbounds curr_thread[tid]
    return firstingroup, curr_contribution
end