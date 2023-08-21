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
                                                                    Tensor(
                                                                        ones((sizes[i] for i in [:m, :f, :q])...),
                                                                        (:m, :f, :q),
                                                                    ),
                                                                    Tensor(
                                                                        ones((sizes[i] for i in [:g, :q])...),
                                                                        (:g, :q),
                                                                    ),
                                                                ],
                                                            ),
                                                            Tensor(
                                                                ones((sizes[i] for i in [:o, :i, :m, :c])...),
                                                                (:o, :i, :m, :c),
                                                            ),
                                                        ],
                                                    ),
                                                    Tensor(ones((sizes[i] for i in [:f, :l, :i])...), (:f, :l, :i)),
                                                ],
                                            ),
                                            Tensor(ones((sizes[i] for i in [:g, :n, :l, :a])...), (:g, :n, :l, :a)),
                                        ],
                                    ),
                                    EinExpr(
                                        [:e, :d, :o],
                                        [
                                            Tensor(ones((sizes[i] for i in [:b, :e])...), (:b, :e)),
                                            Tensor(ones((sizes[i] for i in [:d, :b, :o])...), (:d, :b, :o)),
                                        ],
                                    ),
                                ],
                            ),
                            Tensor(ones((sizes[i] for i in [:c, :e, :h])...), (:c, :e, :h)),
                        ],
                    ),
                    Tensor(ones((sizes[i] for i in [:k, :d, :h, :a, :n, :j])...), (:k, :d, :h, :a, :n, :j)),
                ],
            ),
            Tensor(ones((sizes[i] for i in [:p, :k])...), (:p, :k)),
        ],
    )

    cuttings = findslices(FlopsScorer(), expr, slices = 1000)

    @test prod(i -> size(expr, i), cuttings) >= 1000
end
