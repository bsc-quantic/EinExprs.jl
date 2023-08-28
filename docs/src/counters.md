# Resource counting

As explained before, `EinExpr`s are symbolic expressions representing Einstein summation equations (i.e. tensor summation, permutation, contraction, ...) so no tensor operation is actually performed on them.
Many times, information about the execution cost is needed to optimize the contraction path.
In general, this is a hard task but thanks to Einsteam summation notation only representing linear algebra operations, and these operations are easy to estimate, we can count the resource requirements of any contraction path.

Currently there are 3 resource counters:

```@docs
flops
removedsize
EinExprs.removedrank
```

!!! tip
    These methods only count the resources spent of the contraction **on the root of the tree**.
    In order to count the resources of the whole tree, use `mapreduce` with [`Branches`](@ref):
    
    ```julia
    mapreduce(flops, +, Branches(path))
    ```
