@testset "Slicing" begin
    sizes = Dict(
        :o => 3,
        :b => 7,
        :p => 6,
        :n => 7,
        :j => 9,
        :k => 8,
        :d => 4,
        :e => 2,
        :c => 2,
        :h => 5,
        :i => 5,
        :l => 10,
        :m => 7,
        :q => 5,
        :a => 3,
        :f => 7,
        :g => 3,
    )

    expr = EinExpr(
        [:p, :j],
        [
            EinExpr(
                [:k, :j],
                [
                    EinExpr(
                        [:n, :a, :d, :h],
                        [
                            EinExpr(
                                [:c, :n, :a, :e, :d],
                                [
                                    EinExpr(
                                        [:o, :c, :n, :a],
                                        [
                                            EinExpr(
                                                [:g, :o, :c, :l],
                                                [
                                                    EinExpr(
                                                        [:f, :g, :o, :i, :c],
                                                        [
                                                            EinExpr(
                                                                [:m, :f, :g],
                                                                [
                                                                    EinExpr(
                                                                        [:m, :f, :q],
                                                                        Dict(i => sizes[i] for i in [:m, :f, :q]),
                                                                    ),
                                                                    EinExpr(
                                                                        [:g, :q],
                                                                        Dict(i => sizes[i] for i in [:g, :q]),
                                                                    ),
                                                                ],
                                                            ),
                                                            EinExpr(
                                                                [:o, :i, :m, :c],
                                                                Dict(i => sizes[i] for i in [:o, :i, :m, :c]),
                                                            ),
                                                        ],
                                                    ),
                                                    EinExpr([:f, :l, :i], Dict(i => sizes[i] for i in [:f, :l, :i])),
                                                ],
                                            ),
                                            EinExpr([:g, :n, :l, :a], Dict(i => sizes[i] for i in [:g, :n, :l, :a])),
                                        ],
                                    ),
                                    EinExpr(
                                        [:e, :d, :o],
                                        [
                                            EinExpr([:b, :e], Dict(i => sizes[i] for i in [:b, :e])),
                                            EinExpr([:d, :b, :o], Dict(i => sizes[i] for i in [:d, :b, :o])),
                                        ],
                                    ),
                                ],
                            ),
                            EinExpr([:c, :e, :h], Dict(i => sizes[i] for i in [:c, :e, :h])),
                        ],
                    ),
                    EinExpr([:k, :d, :h, :a, :n, :j], Dict(i => sizes[i] for i in [:k, :d, :h, :a, :n, :j])),
                ],
            ),
            EinExpr([:p, :k], Dict(i => sizes[i] for i in [:p, :k])),
        ],
    )

    cuttings = findslices(FlopsScorer(), expr, slices = 1000)

    @test prod(i -> size(expr, i), cuttings) >= 1000
end
