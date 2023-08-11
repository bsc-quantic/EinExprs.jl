@testset "ChainRulesCore" begin
    using ChainRulesCore
    using ChainRulesTestUtils

    @testset "contract" begin
        @testset "sum" begin
            expr = EinExpr(Symbol[], [Tensor(rand(2, 2), (:i, :j))])
            test_frule(contract, expr)
            test_rrule(contract, expr, check_inferred = false, check_thunked_output_tangent = false)
        end

        @testset "diagonal" begin
            expr = EinExpr([:i], [Tensor(rand(2, 2), (:i, :i))])
            test_frule(contract, expr)
            test_rrule(contract, expr, check_inferred = false, check_thunked_output_tangent = false)
        end

        @testset "trace" begin
            expr = EinExpr(Symbol[], [Tensor(rand(2, 2), (:i, :i))])
            test_frule(contract, expr)
            test_rrule(contract, expr, check_inferred = false, check_thunked_output_tangent = false)
        end

        # TODO axis permutation

        @testset "outer product" begin
            expr = EinExpr([:i, :j], [Tensor(rand(2), (:i,)), Tensor(rand(2), (:j,))])
            test_frule(contract, expr)
            test_rrule(contract, expr, check_inferred = false, check_thunked_output_tangent = false)
        end

        @testset "hadamard product" begin
            expr = EinExpr((:i,), [Tensor(rand(2), (:i,)), Tensor(rand(2), (:i,))])
            test_frule(contract, expr)
            test_rrule(contract, expr, check_inferred = false, check_thunked_output_tangent = false)
        end

        @testset "inner product" begin
            expr = EinExpr((), [Tensor(rand(2), (:i,)), Tensor(rand(2), (:i,))])
            test_frule(contract, expr)
            test_rrule(contract, expr, check_inferred = false, check_thunked_output_tangent = false)
        end

        @testset "matrix multiplication" begin
            expr = EinExpr((:i, :j), [Tensor(rand(2, 3), (:i, :k)), Tensor(rand(3, 4), (:k, :j))])
            test_frule(contract, expr)
            test_rrule(contract, expr, check_inferred = false, check_thunked_output_tangent = false)
        end
    end
end
