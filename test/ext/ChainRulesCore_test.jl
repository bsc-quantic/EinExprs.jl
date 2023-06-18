@testset "ChainRulesCore" begin
    using ChainRulesCore
    using ChainRulesTestUtils

    # NOTE currently `contract` only supports pairwise contractions
    # TODO replace this hack when we implement our own `einsum` method
    unit = Tensor(ones(), ())

    function test_rrule_cotangents(args...; kwargs...)
        config = ChainRulesTestUtils.ADviaRuleConfig()
        test_rrule_cotangents(config, args...; kwargs...)
    end

    function test_rrule_cotangents(
        config::RuleConfig,
        f,
        args...;
        output_tangent=ChainRulesTestUtils.Auto(),
        check_thunked_output_tangent=true,
        fdm=ChainRulesTestUtils._fdm,
        rrule_f=ChainRulesCore.rrule,
        check_inferred::Bool=true,
        fkwargs::NamedTuple=NamedTuple(),
        rtol::Real=1e-9,
        atol::Real=1e-9,
        kwargs...)

        # To simplify some of the calls we make later lets group the kwargs for reuse
        isapprox_kwargs = (; rtol=rtol, atol=atol, kwargs...)

        # and define helper closure over fkwargs
        call(f, xs...) = f(xs...; fkwargs...)

        @testset "test_rrule: $f on $(ChainRulesTestUtils._string_typeof(args))" begin
            # Check correctness of evaluation.
            primals_and_tangents = ChainRulesTestUtils.auto_primal_and_tangent.((f, args...))
            primals = ChainRulesTestUtils.primal.(primals_and_tangents)
            accum_cotangents = ChainRulesTestUtils.tangent.(primals_and_tangents)

            if check_inferred && ChainRulesTestUtils._is_inferrable(primals...; fkwargs...)
                ChainRulesTestUtils._test_inferred(rrule_f, config, primals...; fkwargs...)
            end
            res = rrule_f(config, primals...; fkwargs...)
            res === nothing && throw(MethodError(rrule_f, typeof(ChainRulesTestUtils.primals)))
            y_ad, pullback = res
            y = call(primals...)
            ChainRulesTestUtils.test_approx(y_ad, y; isapprox_kwargs...)  # make sure primal is correct

            ȳ = output_tangent isa ChainRulesTestUtils.Auto ? ChainRulesTestUtils.rand_tangent(y) : output_tangent

            check_inferred && ChainRulesTestUtils._test_inferred(pullback, ȳ)
            ad_cotangents = pullback(ȳ)
            # @test_msg(
            #     "The pullback must return a Tuple (∂self, ∂args...)",
            #     ad_cotangents isa Tuple
            # )
            # @test_msg(
            #     "The pullback should return 1 cotangent for the primal and each primal input.",
            #     length(ad_cotangents) == length(primals)
            # )

            # Correctness testing via finite differencing.
            is_ignored = isa.(accum_cotangents, NoTangent)
            fd_cotangents = ChainRulesTestUtils._make_j′vp_call(fdm, call, ȳ, primals, is_ignored)
            foreach(accum_cotangents, ad_cotangents, fd_cotangents) do args...
                ChainRulesTestUtils._test_cotangent(args...; check_inferred=check_inferred, isapprox_kwargs...)
            end

            # if check_thunked_output_tangent
            #     ChainRulesTestUtils.test_approx(ad_cotangents, pullback(ChainRulesTestUtils.@thunk(ȳ)), "pulling back a thunk:")
            #     check_inferred && ChainRulesTestUtils._test_inferred(pullback, ChainRulesTestUtils.@thunk(ȳ))
            # end
        end
    end

    @testset "contract" begin
        @testset "sum" begin
            expr = EinExpr([Tensor(rand(2, 2), (:i, :j)), unit], Symbol[])
            test_frule(contract, expr)
            test_rrule_cotangents(contract, expr, check_inferred=false)
        end

        @testset "diagonal" begin
            expr = EinExpr([Tensor(rand(2, 2), (:i, :j)), unit], [:i])
            test_frule(contract, expr)
            test_rrule_cotangents(contract, expr, check_inferred=false)
        end

        @testset "trace" begin
            expr = EinExpr([Tensor(rand(2, 2), (:i, :i)), unit], Symbol[])
            test_frule(contract, expr)
            test_rrule_cotangents(contract, expr, check_inferred=false)
        end

        # TODO axis permutation

        @testset "outer product" begin
            expr = EinExpr([Tensor(rand(2), (:i,)), Tensor(rand(2), (:j,))])
            test_frule(contract, expr)
            test_rrule_cotangents(contract, expr, check_inferred=false)
        end

        @testset "hadamard product" begin
            expr = EinExpr([Tensor(rand(2), (:i,)), Tensor(rand(2), (:i,))], (:i,))
            test_frule(contract, expr)
            test_rrule_cotangents(contract, expr, check_inferred=false)
        end

        @testset "inner product" begin
            expr = EinExpr([Tensor(rand(2), (:i,)), Tensor(rand(2), (:i,))])
            test_frule(contract, expr)
            test_rrule_cotangents(contract, expr, check_inferred=false)
        end

        # TODO matrix multiplication
    end
end