@testset "Greedy" begin
    tensors = [
        EinExpr([:j, :b, :i, :h], Dict(i => 2 for i in [:j, :b, :i, :h])),
        EinExpr([:a, :c, :e, :f], Dict(i => 2 for i in [:a, :c, :e, :f])),
        EinExpr([:j], Dict(i => 2 for i in [:j])),
        EinExpr([:e, :a, :g], Dict(i => 2 for i in [:e, :a, :g])),
        EinExpr([:f, :b], Dict(i => 2 for i in [:f, :b])),
        EinExpr([:i, :h, :d], Dict(i => 2 for i in [:i, :h, :d])),
        EinExpr([:d, :g, :c], Dict(i => 2 for i in [:d, :g, :c])),
    ]

    path = einexpr(Greedy(), EinExpr(Symbol[], tensors))

    @test path isa EinExpr

    @test mapreduce(flops, +, Branches(path)) == 100

    @test all(splat(issetequal), zip(contractorder(path), [[:i, :h], [:j], [:a, :e], [:g, :c], [:f], [:b, :d]]))
end
