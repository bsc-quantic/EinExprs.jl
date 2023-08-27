@testset "Makie" begin
    using CairoMakie
    using NetworkLayout: Spring

    sizes = Dict(
        :o => 2,
        :b => 2,
        :p => 2,
        :n => 2,
        :j => 2,
        :k => 2,
        :d => 2,
        :e => 2,
        :c => 2,
        :h => 2,
        :i => 2,
        :l => 2,
        :m => 2,
        :q => 2,
        :a => 2,
        :f => 2,
        :g => 2,
    )

    tensors = [
        EinExpr([:f, :l, :i], filter(p -> p.first ∈ [:f, :l, :i], sizes)),
        EinExpr([:b, :e], filter(p -> p.first ∈ [:b, :e], sizes)),
        EinExpr([:g, :n, :l, :a], filter(p -> p.first ∈ [:g, :n, :l, :a], sizes)),
        EinExpr([:o, :i, :m, :c], filter(p -> p.first ∈ [:o, :i, :m, :c], sizes)),
        EinExpr([:k, :d, :h, :a, :n, :j], filter(p -> p.first ∈ [:k, :d, :h, :a, :n, :j], sizes)),
        EinExpr([:m, :f, :q], filter(p -> p.first ∈ [:m, :f, :q], sizes)),
        EinExpr([:p, :k], filter(p -> p.first ∈ [:p, :k], sizes)),
        EinExpr([:c, :e, :h], filter(p -> p.first ∈ [:c, :e, :h], sizes)),
        EinExpr([:g, :q], filter(p -> p.first ∈ [:g, :q], sizes)),
        EinExpr([:d, :b, :o], filter(p -> p.first ∈ [:d, :b, :o], sizes)),
    ]

    path = einexpr(Exhaustive(), EinExpr([:p, :j], tensors))

    @testset "plot!" begin
        f = Figure()
        @testset "(default)" plot!(f[1, 1], path)
        @testset "with labels" plot!(f[1, 1], path; inds = true)
        @testset "3D" plot!(f[1, 1], path; layout = Spring(dim = 3))
    end

    @testset "plot" begin
        @testset "(default)" plot(path)
        @testset "with labels" plot(path; inds = true)
        @testset "3D" plot(path; layout = Spring(dim = 3))
    end
end
