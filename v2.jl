struct SimulatorVar{T}
    name::Symbol
    default::T
    read::Bool
    write::Bool
    target::Symbol
end

function localname(var::SimulatorVar)
    if var.target in (:pre, :post)
        return Symbol(var.target, :_, var.name)
    else
        return var.name
    end
end

function source(var::SimulatorVar)
    name = var.name
    if variable.target == :local
        return :(localVars.$name[idx])
    elseif variable.target == :pre
        return :(preVars.$name[pre_idx])
    elseif variable.target == :post
        return :(postVars.$name[post_idx])
    elseif variable.target == :global
        return :(globalVars.$name)
    end
end

struct NeuronFeature
    stepAction::Expr
    variables::Vector{SimulatorVar}
    order::Float64
end

struct Synapse
    onSpike::Expr
    #stepAction::Union{Nothing, Expr}
    variables::Vector{SimulatorVar}
    preFeatures::Vector{NeuronFeature}
    postFeatures::Vector{NeuronFeature}
    order::Float64
end

const gpuhead = quote
    idx = blockDim().x*(blockIdx().x-1) + threadIdx().x
    if idx > length(getfield(localVars, 1))
        return nothing
    end
end

function neuronKernelBody(features, target)
    sorted = sort(featutures, by=f->f.order)
    variables = Set{Symbol}()
    body = Expr(:block)
    if target == :gpu
        push!(body.args, gpuhead)
    end
    
    readVariables = Set{Symbol}()
    for feature in sorted, variable in feature.variables
        ln = localname(variable)
        if !(ln in readVariables)
            if variable.read
                push!(body.args, :($(ln) = $(source(variable))))
            else
                push!(body.args, :($(ln) = $(variable.default)))
            end
            push!(readVariables, ln)
        end
    end
    
    for feature in sorted
        push!(body.args, feature.stepAction)
    end
    
    writtenVariables = Set{Symbol}()
    for feature in sorted, variable in feature.variables
        if variable.write
            ln = localname(variable)
            if !(ln in writtenVariables)
                push!(body.args, :($(source(variable)) = $(ln)))
                push!(writtenVariables, ln)
            end
        end
    end
    return body
end

#function gpuNeuronKernel(features)
#    eval(:((localVars, globalVars)->$(neuronKernelBody(features, :gpu))))
#end

function synapseKernelBody(synapse::Synapse, target)
    body = Expr(:block)
    if target == :gpu
        push!(body.args, gpuhead)
    end
    #push!(body.args, :(pre_idx = localVars.pre_idx[idx]))
    push!(body.args, :(post_idx = localVars.post_idx[idx]))
    for variable in synapse.variables
        push!(body.args, :($(localname(variable)) = $(source(variable))))
    end
    push!(body.args, synapse.onSpike)
    for variable in synapse.variables
        if variable.write
            push!(body.args, :($(source(variable)) = $(localname(variable))))
        end
    end
end

#function gpuSynapseKernel(synapse)
#    eval(:((localVars, preVars, postVars, globalVars)->
#            $(synapseKernelBody(synapse, :gpu))))
#end

struct NeuronGroup{LV, FCN} where {LV <: NamedTuple, FCN <: Function}
    localVars::LV
    updateFcn::FCN
end

function NeuronGroup(n::Integer, features::AbstractArray{NeuronFeature}, target::Symbol)
    checkInconsistentVariables(features)
    vardict = Dict{Symbol, AbstractArray}()
    for feature in features, variable in feature.variables
        if target == :gpu
            vardict[variable.name] = CUDA.fill(variable.default, (n,))
        end
    end
    eval(:(fcn(localVars, globalVars)->$(neuronKernelBody(features, target))))
    NeuronGroup((;vardict...), fcn)
end

function checkInconsitentVariables(features::AbstractArray{NeuronFeature})
    vardict = Dict{Symbol, SimulatorVar}()
    for feature in features, variable in feature.variables
        if haskey(vardict, variable.name)
            if vardict[variable.name] != variable
                error("Inconsistent definition of variable $(variable.name).")
            end
        end
    end
end

struct SynapseGroup{LV, FCN} where {LV <: NamedTuple, PreV <: NamedTuple,
                                    PostV <: NamedTuple, FCN <: Function}
    localVars::LV
    preVars::PreV
    postVars::PostV
    updateFcn::FCN
end

#struct NeuronGroup#{ConstType} where {ConstType <: NamedTuple}
#    constants::ConstType
#    parameters::Vector{ParType}
#    variables::Vector{VarType}
#    connections
#end

#struct SynapseGroup{synapsetype, pretype, posttype}
    
    
############
struct NeuronType{S} end
NeuronType(S::Symbol) = NeuronType{S}()
constants(::NeuronType{:LIF}) = (tau=20.0f0, V_res=0.0f0, V_thres=1.0f0)
parameters(::NeuronType{:LIF}) = NamedTuple()
variables(::NeuronType{:LIF}) = (V=0.0f0,)
@generated function update(a::NamedTuple{anames, atypes},
                           b::NamedTuple{bnames, btypes}) where {anames, atypes,
                                                                 bnames, btypes}
    vals = Any[ :(getfield($(Base.sym_in(n, bnames) ? :b : :a), $(QuoteNode(n)))) for n in anames ]
    :(NamedTuple{$anames, $atypes}(($(vals...),)))
end

struct Network
    neurons::Vector{Tuple{Symbol, NamedTuple}}
    synapses::Vector{Tuple{Symbol, Int64, Int64, NamedTuple}}
end
Network() = Network(Vector{Tuple{Symbol, NamedTuple}}[],
                    Vector{Tuple{Symbol, Int64, Int64, NamedTuple}}[])

struct Simulation{VarList, ParmList, SynList}
    variables::VarList
    parameters::ParmList
    synapses::SynList
end

struct SynapseGroup{synapsetype, pretype, posttype}
    

function Simulation(network::Network)
    
    neurontypes = Tuple(unique(getindex.(network.neurons, 1)))
    vars = (;(nt=>typeof(variables(NeuronType(nt)))[] for nt in neurontypes)...)
    params = (;(nt=>typeof(parameters(NeuronType(nt)))[] for nt in neurontypes)...)
    for (nt, kwargs) in network.neurons
        push!(vars[nt], update(variables(NeuronType(nt)), kwargs))
        push!(params[nt], update(parameters(NeuronType(nt)), kwargs))
    end
    
    targettypes = (;(nt=>Tuple{Symbol, Symbol}[] for nt in neurontypes)...)
    
    #vars = NamedTuple{neurontypes, vartypes}()
    #parms = NamedTuple{neurontypes, paramtypes}()
    #vars
    Simulation(vars, params, ())
end

net = Network()
for _=1:10000
    push!(net.neurons, (:LIF, (;V=0.0)))
end
sim = Simulation(net)
sim.variables.LIF
    
