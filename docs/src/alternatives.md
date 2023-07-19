# Alternatives

`EinExprs` is deeply inspired by [`opt_einsum`](https://github.com/dgasmith/opt_einsum) and [`cotengra`](https://github.com/jcmgray/cotengra).
Actually most of the contraction path search and slicing algorithms have been imported from there.
If you happen to be working in Python, you should definetely check out these libraries.

Although we believe there is no similar project in the Julia world, there are some overlapping libraries that may suit you if `EinExprs` doesn't fit your case.

- [`TensorOperations`](https://github.com/Jutho/TensorOperations.jl)
- [`OMEinsum`](https://github.com/under-Peter/OMEinsum.jl) and [`OMEinsumContractionOrders`](https://github.com/TensorBFS/OMEinsumContractionOrders.jl)
