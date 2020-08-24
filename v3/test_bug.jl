import CUDA
host_arr = [1, 2, 3, 4]
dev_arr = CUDA.cu(host_arr)

function parent_func(arr)
    @CUDA.cuprintln("Length (parent): ", length(arr))
    @CUDA.cuda blocks=1 threads=1 dynamic=true child_func(arr[3], 1, 2, 3, 4)
    return nothing
end

function child_func(arr, a, b, c, d)
    @CUDA.cuprintln("Length (child): ", length(arr))
    return nothing
end

@CUDA.cuda blocks=1 threads=1 parent_func(dev_arr)

function manyArgs(a, b, c, d, e, f, g)
    return nothing
end
function caller()
    @CUDA.cuda blocks=1 threads=1 dynamic=true manyArgs(1,2,3,4,5,6,7)
    return nothing
end
@CUDA.cuda blocks=1 threads=1 caller()

include("utils.jl")
host_vv = [[1,2,3], [4,5], [6,7,8,9,10]]
dev_vv = CuVectorVector(host_vv)
@CUDA.cuda blocks=1 threads=1 parent_func(dev_vv)
