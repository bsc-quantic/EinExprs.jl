# Alternatives

`EinExprs` is deeply inspired by [`opt_einsum`](https://github.com/dgasmith/opt_einsum) and [`cotengra`](https://github.com/jcmgray/cotengra).
Actually most of the contraction path search and slicing algorithms have been imported from there.
If you happen to be working in Python, you should definetely check out these libraries.

!!! info "Differences with `opt_einsum`"
    Although the differences are minimal, any user coming from `opt_einsum` should be aware that:
    - The `"optimal"` contraction path solver in `opt_einsum` is known as `Exhaustive` in `EinExprs`.
    - When counting FLOPs, `opt_einsum` gives a value $\times 2$ higher than the `EinExprs.flops` counter.[^1]

[^1]: We are not sure of the reason behind this mismatch or which package gives the correct answer, but since the factor remains constant, it should not affect for comparing contraction paths during the minimization step.

Although we believe there is no similar project in the Julia world, there are some overlapping libraries that may suit you if `EinExprs` doesn't fit your case.

- [`TensorOperations`](https://github.com/Jutho/TensorOperations.jl)
- [`OMEinsum`](https://github.com/under-Peter/OMEinsum.jl) and [`OMEinsumContractionOrders`](https://github.com/TensorBFS/OMEinsumContractionOrders.jl)
