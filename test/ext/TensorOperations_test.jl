@testset "TensorOperations" begin
    using TensorOperations

    # Copied from exhaustive tests
    @testset "Contraction Order" begin
        config = Base.get_extension(EinExprs, :EinExprsTensorOperationsExt).Netcon()

        tensors = [
            EinExpr([:j, :b, :i, :h], Dict(i => 2 for i in [:j, :b, :i, :h])),
            EinExpr([:a, :c, :e, :f], Dict(i => 2 for i in [:a, :c, :e, :f])),
            EinExpr([:j], Dict(i => 2 for i in [:j])),
            EinExpr([:e, :a, :g], Dict(i => 2 for i in [:e, :a, :g])),
            EinExpr([:f, :b], Dict(i => 2 for i in [:f, :b])),
            EinExpr([:i, :h, :d], Dict(i => 2 for i in [:i, :h, :d])),
            EinExpr([:d, :g, :c], Dict(i => 2 for i in [:d, :g, :c])),
        ]

        path = einexpr(config, EinExpr(Symbol[], tensors))

        @test path isa EinExpr
        @test mapreduce(flops, +, Branches(path)) == 92

        @test all(splat(issetequal), zip(contractorder(path), [[:a, :e], [:c, :g], [:f], [:j], [:h, :i], [:b, :d]]))
    end

    @testset "ncon" begin
        tensor_exprs = [
            EinExpr([:A, :a, :b, :c], Dict(:A => 2, :a => 2, :b => 2, :c => 2))
            EinExpr([:b, :d, :e, :f], Dict(:b => 2, :d => 2, :e => 2, :f => 2))
            EinExpr([:a, :e, :g, :C], Dict(:a => 2, :e => 2, :g => 2, :C => 2))
            EinExpr([:c, :h, :d, :i], Dict(:c => 2, :h => 2, :d => 2, :i => 2))
            EinExpr([:f, :i, :g, :j], Dict(:f => 2, :i => 2, :g => 2, :j => 2))
            EinExpr([:B, :h, :k, :l], Dict(:B => 2, :h => 2, :k => 2, :l => 2))
            EinExpr([:j, :k, :l, :D], Dict(:j => 2, :k => 2, :l => 2, :D => 2))
        ]

        path = einexpr(Exhaustive(), EinExpr([:A, :B, :C, :D], tensor_exprs))

        result = ncon([rand(2, 2, 2, 2) for i in 1:length(leaves(path))], path)

        @test result isa Array
        @test size(result) == (2, 2, 2, 2)
    end
end
