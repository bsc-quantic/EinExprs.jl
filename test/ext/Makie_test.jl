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
        Tensor(ones((sizes[i] for i in [:f, :l, :i])...), [:f, :l, :i]),
        Tensor(ones((sizes[i] for i in [:b, :e])...), [:b, :e]),
        Tensor(ones((sizes[i] for i in [:g, :n, :l, :a])...), [:g, :n, :l, :a]),
        Tensor(ones((sizes[i] for i in [:o, :i, :m, :c])...), [:o, :i, :m, :c]),
        Tensor(
            ones((sizes[i] for i in [:k, :d, :h, :a, :n, :j])...),
            [:k, :d, :h, :a, :n, :j],
        ),
        Tensor(ones((sizes[i] for i in [:m, :f, :q])...), [:m, :f, :q]),
        Tensor(ones((sizes[i] for i in [:p, :k])...), [:p, :k]),
        Tensor(ones((sizes[i] for i in [:c, :e, :h])...), [:c, :e, :h]),
        Tensor(ones((sizes[i] for i in [:g, :q])...), [:g, :q]),
        Tensor(ones((sizes[i] for i in [:d, :b, :o])...), [:d, :b, :o]),
    ]

    path = einexpr(Greedy, EinExpr(tensors, [:p, :j]))

    @testset "plot!" begin
        f = Figure()
        @testset "(default)" plot!(f[1, 1], path)
        @testset "with labels" plot!(f[1, 1], path; labels = true)
        @testset "3D" plot!(f[1, 1], path; layout = Spring(dim = 3))
    end

    @testset "plot" begin
        @testset "(default)" plot(path)
        @testset "with labels" plot(path; labels = true)
        @testset "3D" plot(path; layout = Spring(dim = 3))
    end
end
