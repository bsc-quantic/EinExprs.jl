@testset "ChainRulesCore" begin
    using ChainRulesCore
    using ChainRulesTestUtils

    # NOTE currently `contract` only supports pairwise contractions
    # TODO replace this hack when we implement our own `einsum` method
    unit = Tensor(ones(), ())

    @testset "contract" begin
        @testset "sum" begin
            expr = EinExpr([Tensor(rand(2, 2), (:i, :j)), unit], Symbol[])
            test_frule(contract, expr)
        end

        @testset "diagonal" begin
            expr = EinExpr([Tensor(rand(2, 2), (:i, :j)), unit], [:i])
            test_frule(contract, expr)
        end

        @testset "trace" begin
            expr = EinExpr([Tensor(rand(2, 2), (:i, :i)), unit], Symbol[])
            test_frule(contract, expr)
        end

        # TODO axis permutation

        @testset "outer product" begin
            expr = EinExpr([Tensor(rand(2), (:i,)), Tensor(rand(2), (:j,))])
            test_frule(contract, expr)
        end

        @testset "inner product" begin
            expr = EinExpr([Tensor(rand(2), (:i,)), Tensor(rand(2), (:i,))])
            test_frule(contract, expr)
        end

        # TODO matrix multiplication, hadamard product
    end
end