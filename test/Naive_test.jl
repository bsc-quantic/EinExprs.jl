@testset "Naive" begin
    sizedict = Dict(i => 2 for i in [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j])
    tensors = [
        EinExpr([:j, :b, :i, :h]),
        EinExpr([:a, :c, :e, :f]),
        EinExpr([:j]),
        EinExpr([:e, :a, :g]),
        EinExpr([:f, :b]),
        EinExpr([:i, :h, :d]),
        EinExpr([:d, :g, :c]),
    ]

    path = einexpr(EinExprs.Naive(), EinExpr(Symbol[], tensors), sizedict)

    @test path isa EinExpr
    @test foldl((a, b) -> sum([a, b]), tensors) == path

    # TODO traverse through the tree and check everything is ok
    @test mapreduce(flops(sizedict), +, Branches(path)) == 872

    # FIXME non-determinist behaviour on order
    @test all(
        splat(issetequal),
        zip(map(suminds, Branches(path)), [Symbol[], [:j], [:a, :e], [:f, :b], [:i, :h], [:d, :g, :c]]),
    )

    @testset "hyperedges" begin
        sizedict = Dict(i => 2 for i in [:i, :j, :k, :l, :m, :β])
        a = EinExpr([:i, :β, :j])
        b = EinExpr([:k, :β])
        c = EinExpr([:β, :l, :m])

        path = einexpr(EinExprs.Naive(), sum([a, b, c], skip = [:β]), sizedict)
        @test all(∋(:β) ∘ head, branches(path))

        path = einexpr(EinExprs.Naive(), sum([a, b, c], skip = Symbol[]), sizedict)
        @test all(∋(:β) ∘ head, branches(path)[1:end-1])
        @test all(!∋(:β) ∘ head, branches(path)[end:end])
    end
end
