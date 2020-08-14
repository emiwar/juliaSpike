struct BuiltPopulation{LV <: NamedTuple, FCN <: Function}
    localVars::LV
    updateFcn::FCN
end

length(pop::BuiltPopulation) = Base.length(pop.localVars[1])

struct IndexMap{I <: Integer}
    synToPre::CUDA.CuVector{I}
    synToPost::CUDA.CuVector{I}
    preToSyn::CuVectorVector{I, I}
    postToSyn::CuVectorVector{I, I}
end

struct BuiltConnection{
                    LV <: NamedTuple,
                    PreV <: NamedTuple,
                    PostV <: NamedTuple,
                    PreFCN <: Function,
                    PostFCN <: Function,
                    IM <: IndexMap}
    localVars::LV
    preVars::PreV
    postVars::PostV
    onPre::PreFCN
    onPost::PostFCN
    indexMap::IM
end

struct Network
    populations::Vector{BuiltPopulation}
    connections::Vector{BuiltConnection}
    globalVars::NamedTuple
end

function Network(populations::Vector{Population},
                 connections::Vector{Connection},
                 target::Symbol=:gpu; kwargs...)
    globalVars = buildGlobalVars(populations, connections, kwargs)
    builtPopulations = build.(populations, target)
    popDict = IdDict{Population, BuiltPopulation}(zip(populations, builtPopulations))
    builtConnections = [build(c, target, popDict) for c in connections]
    Network(builtPopulations, builtConnections, globalVars)
end

function localname(var::SimulatorVar)
    if var.target in (:pre, :post)
        return Symbol(var.target, :_, var.name)
    else
        return var.name
    end
end

function source(variable::SimulatorVar)
    name = variable.name
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

neuronKernelHead(::Val{:gpu}) = quote
    idx = CUDA.blockDim().x*(CUDA.blockIdx().x-1) + CUDA.threadIdx().x
    if idx > Base.length(getfield(localVars, 1))
        return nothing
    end
end

function neuronKernelBody(features, target)
    sorted = sort(features, by=f->f.order)
    variables = Set{Symbol}()
    body = Expr(:block)
    push!(body.args, neuronKernelHead(Val(target)))
    
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
    push!(body.args, :(return nothing))
    return body
end

synapseKernelHead(::Val{:gpu}) = quote
    kernelidx = CUDA.blockDim().x*(CUDA.blockIdx().x-1) + CUDA.threadIdx().x
    #@CUDA.cuprintln("Synapse: kernelidx = ", kernelidx, " length=", Base.length(idxs))
    if kernelidx > Base.length(idxs)
        return nothing
    end
    idx = idxs[kernelidx]
    localVars = vars.localVars
    preVars = vars.preVars
    postVars = vars.postVars
    globalVars = vars.globalVars
end

function synapseKernelBody(synapse::Synapse, site::Symbol, target::Symbol)
    body = Expr(:block)
    
    push!(body.args, synapseKernelHead(Val(target)))
    if site==:pre
        push!(body.args, :(post_idx = synToPost[idx]))
    elseif site==:post
        push!(body.args, :(pre_idx = synToPre[idx]))
    end
    
    for variable in synapse.variables
        if variable.read
            push!(body.args, :($(localname(variable)) = $(source(variable))))
        end
    end
    if site == :pre
        push!(body.args, synapse.onPre)
    elseif site == :post
        push!(body.args, synapse.onPost)
    end
    for variable in synapse.variables
        if variable.write
            push!(body.args, :($(source(variable)) = $(localname(variable))))
        end
    end
    push!(body.args, :(return nothing))
    return body
end

function build(population::Population, target::Symbol)
    vardict = Dict{Symbol, SimulatorVar}()
    localvars = Dict{Symbol, AbstractArray}()
    for feature in population.features, variable in feature.variables
        existingvalue = get(vardict, variable.name, nothing)
        if existingvalue != nothing
            if existingvalue != variable
                error("Inconsistent definition of variable $(variable.name).")
            end
        else
            vardict[variable.name] = variable
            if variable.target == :local && target == :gpu
                val = get(population.values, variable.name, variable.default)
                if val isa AbstractArray
                    @assert length(val) == population.size
                    localvars[variable.name] = CUDA.cu(val)
                else
                    localvars[variable.name] = CUDA.fill(val, (population.size,))
                end
            end
        end
    end
    eval(:(fcn(localVars, globalVars)=$(neuronKernelBody(population.features, target))))
    BuiltPopulation((;localvars...), fcn)
end

function build(connection::Connection, target::Symbol,
               popdict::IdDict{Population, BuiltPopulation})
    @assert target == :gpu #TODO: make this more agnostic
    #Indicies
    presize = length(popdict[connection.prePop])
    postsize = length(popdict[connection.postPop])
    synToPre = Int32[]
    synToPost = Int32[]
    preToSyn = [Int32[] for i=1:presize]
    postToSyn = [Int32[] for i=1:postsize]
    synId = 0
    for (preId, postId) in connection.prePostItr
        synId += 1
        push!(synToPre, preId)
        push!(synToPost, postId)
        push!(preToSyn[preId], synId)
        push!(postToSyn[postId], synId)
    end
    indexMap = IndexMap(CUDA.cu(synToPre), CUDA.cu(synToPost),
                        CuVectorVector(preToSyn), CuVectorVector(postToSyn))
    connectionsize = synId
    
    #Variables
    vardict = Dict{Symbol, SimulatorVar}()
    localvars = Dict{Symbol, AbstractArray}()
    prevars = Dict{Symbol, AbstractArray}()
    postvars = Dict{Symbol, AbstractArray}()
    for variable in connection.synapse.variables
        if haskey(vardict, variable.name)
            if vardict[variable.name] != variable
                error("Inconsistent definition of variable $(variable.name).")
            end
        else
            vardict[variable.name] = variable

            if variable.target == :local
                val = get(connection.values, variable.name, variable.default)
                if val isa AbstractArray
                    @assert length(val) == connectionsize
                    localvars[variable.name] = CUDA.cu(val)
                else
                    localvars[variable.name] = CUDA.fill(val, (connectionsize,))
                end
            elseif variable.target == :pre
                prepop = popdict[connection.prePop]
                prevars[variable.name] = prepop.localVars[variable.name]
            elseif variable.target == :post
                postpop = popdict[connection.postPop]
                postvars[variable.name] = postpop.localVars[variable.name]
            end
        end
    end
    
    #Body
    prebody = synapseKernelBody(connection.synapse, :pre, target)
    postbody = synapseKernelBody(connection.synapse, :post, target)
    eval(:(onPost(post_idx, idxs, synToPre, vars)=$postbody))

    eval(:(onPre(pre_idx, idxs, synToPost, vars)=$prebody))
    
    #Build
    BuiltConnection((;localvars...),(;prevars...),(;postvars...), onPre, onPost, indexMap)
end

function buildGlobalVars(populations, connections, presets)
    globalVars = Dict{Symbol, Any}()
    for population in populations,
            feature in population.features,
            variable in feature.variables
        if variable.target == :global
            globalVars[variable.name] = variable.default
        end
    end
    for connection in connections,
            variable in connection.synapse.variables
        if variable.target == :global
            globalVars[variable.name] = variable.default
        end
    end
    for (k,v) in presets
        globalVars[k] = v
    end
    return (; globalVars...)
end

#TODO: synapses that write pre/post variables should lock the variable! 