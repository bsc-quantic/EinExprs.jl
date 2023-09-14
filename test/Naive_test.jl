@testset "Naive" begin
    tensors = [
        EinExpr([:j, :b, :i, :h], Dict(i => 2 for i in [:j, :b, :i, :h])),
        EinExpr([:a, :c, :e, :f], Dict(i => 2 for i in [:a, :c, :e, :f])),
        EinExpr([:j], Dict(i => 2 for i in [:j])),
        EinExpr([:e, :a, :g], Dict(i => 2 for i in [:e, :a, :g])),
        EinExpr([:f, :b], Dict(i => 2 for i in [:f, :b])),
        EinExpr([:i, :h, :d], Dict(i => 2 for i in [:i, :h, :d])),
        EinExpr([:d, :g, :c], Dict(i => 2 for i in [:d, :g, :c])),
    ]

    path = einexpr(EinExprs.Naive(), EinExpr(Symbol[], tensors))

    @test path isa EinExpr
    @test foldl((a, b) -> sum([a, b]), tensors) == path

    # TODO traverse through the tree and check everything is ok
    @test mapreduce(flops, +, Branches(path)) == 872

    # FIXME non-determinist behaviour on order
    @test all(
        splat(issetequal),
        zip(map(suminds, Branches(path)), [Symbol[], [:j], [:a, :e], [:f, :b], [:i, :h], [:d, :g, :c]]),
    )

    @testset "hyperedges" begin
        a = EinExpr([:i, :β, :j], Dict(i => 2 for i in [:i, :β, :j]))
        b = EinExpr([:k, :β], Dict(i => 2 for i in [:k, :β]))
        c = EinExpr([:β, :l, :m], Dict(i => 2 for i in [:β, :l, :m]))

        path = einexpr(EinExprs.Naive(), sum([a, b, c], skip = [:β]))
        @test all(∋(:β) ∘ head, branches(path))

        path = einexpr(EinExprs.Naive(), sum([a, b, c], skip = Symbol[]))
        @test all(∋(:β) ∘ head, branches(path)[1:end-1])
        @test all(!∋(:β) ∘ head, branches(path)[end:end])
    end
end