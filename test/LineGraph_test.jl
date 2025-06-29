@testset "LineGraph" begin
    # connected
    @testset let network = SizedEinExpr(
            EinExpr(
                Symbol[],
                [
                    EinExpr([:i, :m])
                    EinExpr([:i, :j, :p])
                    EinExpr([:n, :j, :k])
                    EinExpr([:p, :k, :l])
                    EinExpr([:m, :n, :o])
                    EinExpr([:o, :l])
                ],
            ),
            Dict(:i => 2, :j => 2, :k => 2, :l => 2, :m => 2, :n => 2, :o => 2, :p => 2),
        )
        path1 = einexpr(Greedy(), network)
        path2 = einexpr(LineGraph(), network)
        @test mapreduce(flops, +, Branches(path1)) >= mapreduce(flops, +, Branches(path2)) - 10

        # TODO numerical test disabled due to circular dependency
        # @test contract(network; path = path1) ≈ contract(network; path = path2)
    end

    @testset let network = SizedEinExpr(
            EinExpr(
                Symbol[:i, :p],
                [
                    EinExpr([:i, :m])
                    EinExpr([:i, :j, :p])
                    EinExpr([:n, :j, :k])
                    EinExpr([:p, :k, :l])
                    EinExpr([:m, :n, :o])
                    EinExpr([:o, :l])
                ],
            ),
            Dict(:i => 2, :j => 2, :k => 2, :l => 2, :m => 2, :n => 2, :o => 2, :p => 2),
        )
        path1 = einexpr(Greedy(), network)
        path2 = einexpr(LineGraph(), network)
        @test mapreduce(flops, +, Branches(path1)) >= mapreduce(flops, +, Branches(path2)) - 10

        # TODO numerical test disabled due to circular dependency
        # @test contract(network; path = path1) ≈ contract(network; path = path2)
    end

    # disconnected
    @testset let network = SizedEinExpr(
            EinExpr(Symbol[], [EinExpr([:i, :j]), EinExpr([:i, :j]), EinExpr([:k, :l, :m]), EinExpr([:k, :l, :m])]),
            Dict(:i => 2, :j => 2, :k => 2, :l => 2, :m => 2),
        )
        path1 = einexpr(network; optimizer = Greedy())
        path2 = einexpr(network; optimizer = LineGraph())
        @test mapreduce(flops, +, Branches(path1)) >= mapreduce(flops, +, Branches(path2)) - 10

        # TODO numerical test disabled due to circular dependency
        # @test contract(network; path = path1) ≈ contract(network; path = path2)
    end

    @testset let network = SizedEinExpr(
            EinExpr(Symbol[:i], [EinExpr([:i, :j]), EinExpr([:i, :j]), EinExpr([:k, :l, :m]), EinExpr([:k, :l, :m])]),
            Dict(:i => 2, :j => 2, :k => 2, :l => 2, :m => 2),
        )
        path1 = einexpr(Greedy(), network)
        path2 = einexpr(LineGraph(), network)
        @test mapreduce(flops, +, Branches(path1)) >= mapreduce(flops, +, Branches(path2)) - 10
    end

    # TODO numerical test disabled due to circular dependency
    # @test contract(network; path = path1) ≈ contract(network; path = path2)

    @testset let network = SizedEinExpr(
            EinExpr(
                Symbol[:i, :k],
                [EinExpr([:i, :j]), EinExpr([:i, :j]), EinExpr([:k, :l, :m]), EinExpr([:k, :l, :m])],
            ),
            Dict(:i => 2, :j => 2, :k => 2, :l => 2, :m => 2),
        )
        path1 = einexpr(Greedy(), network)
        path2 = einexpr(LineGraph(), network)
        @test mapreduce(flops, +, Branches(path1)) >= mapreduce(flops, +, Branches(path2)) - 10

        # TODO numerical test disabled due to circular dependency
        # @test contract(network; path = path1) ≈ contract(network; path = path2)
    end
end
