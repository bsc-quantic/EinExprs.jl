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
        [
            EinExpr(
                [
                    EinExpr(
                        [
                            EinExpr(
                                [
                                    EinExpr(
                                        [
                                            EinExpr(
                                                [
                                                    EinExpr(
                                                        [
                                                            EinExpr(
                                                                [
                                                                    Tensor(
                                                                        ones((sizes[i] for i in [:m, :f, :q])...),
                                                                        [:m, :f, :q],
                                                                    ),
                                                                    Tensor(
                                                                        ones((sizes[i] for i in [:g, :q])...),
                                                                        [:g, :q],
                                                                    ),
                                                                ],
                                                                [:m, :f, :g],
                                                            ),
                                                            Tensor(
                                                                ones((sizes[i] for i in [:o, :i, :m, :c])...),
                                                                [:o, :i, :m, :c],
                                                            ),
                                                        ],
                                                        [:f, :g, :o, :i, :c],
                                                    ),
                                                    Tensor(ones((sizes[i] for i in [:f, :l, :i])...), [:f, :l, :i]),
                                                ],
                                                [:g, :o, :c, :l],
                                            ),
                                            Tensor(ones((sizes[i] for i in [:g, :n, :l, :a])...), [:g, :n, :l, :a]),
                                        ],
                                        [:o, :c, :n, :a],
                                    ),
                                    EinExpr(
                                        [
                                            Tensor(ones((sizes[i] for i in [:b, :e])...), [:b, :e]),
                                            Tensor(ones((sizes[i] for i in [:d, :b, :o])...), [:d, :b, :o]),
                                        ],
                                        [:e, :d, :o],
                                    ),
                                ],
                                [:c, :n, :a, :e, :d],
                            ),
                            Tensor(ones((sizes[i] for i in [:c, :e, :h])...), [:c, :e, :h]),
                        ],
                        [:n, :a, :d, :h],
                    ),
                    Tensor(ones((sizes[i] for i in [:k, :d, :h, :a, :n, :j])...), [:k, :d, :h, :a, :n, :j]),
                ],
                [:k, :j],
            ),
            Tensor(ones((sizes[i] for i in [:p, :k])...), [:p, :k]),
        ],
        [:p, :j],
    )

    cuttings = findslices(FlopsScorer(), expr, slices = 1000)

    @test prod(i -> size(expr, i), cuttings) >= 1000
end