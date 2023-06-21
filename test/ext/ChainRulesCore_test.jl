@testset "ChainRulesCore" begin
    using ChainRulesCore
    using ChainRulesTestUtils

    @testset "contract" begin
        @testset "sum" begin
            expr = EinExpr([Tensor(rand(2, 2), (:i, :j))], Symbol[])
            test_frule(contract, expr)
            test_rrule(contract, expr, check_inferred=false, check_thunked_output_tangent=false)
        end

        @testset "diagonal" begin
            expr = EinExpr([Tensor(rand(2, 2), (:i, :j))], [:i])
            test_frule(contract, expr)
            test_rrule(contract, expr, check_inferred=false, check_thunked_output_tangent=false)
        end

        @testset "trace" begin
            expr = EinExpr([Tensor(rand(2, 2), (:i, :i))], Symbol[])
            test_frule(contract, expr)
            test_rrule(contract, expr, check_inferred=false, check_thunked_output_tangent=false)
        end

        # TODO axis permutation

        @testset "outer product" begin
            expr = EinExpr([Tensor(rand(2), (:i,)), Tensor(rand(2), (:j,))])
            test_frule(contract, expr)
            test_rrule(contract, expr, check_inferred=false, check_thunked_output_tangent=false)
        end

        @testset "hadamard product" begin
            expr = EinExpr([Tensor(rand(2), (:i,)), Tensor(rand(2), (:i,))], (:i,))
            test_frule(contract, expr)
            test_rrule(contract, expr, check_inferred=false, check_thunked_output_tangent=false)
        end

        @testset "inner product" begin
            expr = EinExpr([Tensor(rand(2), (:i,)), Tensor(rand(2), (:i,))])
            test_frule(contract, expr)
            test_rrule(contract, expr, check_inferred=false, check_thunked_output_tangent=false)
        end

        # TODO matrix multiplication
    end
end