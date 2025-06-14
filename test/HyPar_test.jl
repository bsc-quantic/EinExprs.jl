@testset "HyPar" begin
    using KaHyPar

    @testset begin
        tensors = [
            EinExpr([:j, :b, :i, :h], Dict(i => 2 for i in [:j, :b, :i, :h])),
            EinExpr([:a, :c, :e, :f], Dict(i => 2 for i in [:a, :c, :e, :f])),
            EinExpr([:j], Dict(i => 2 for i in [:j])),
            EinExpr([:e, :a, :g], Dict(i => 2 for i in [:e, :a, :g])),
            EinExpr([:f, :b], Dict(i => 2 for i in [:f, :b])),
            EinExpr([:i, :h, :d], Dict(i => 2 for i in [:i, :h, :d])),
            EinExpr([:d, :g, :c], Dict(i => 2 for i in [:d, :g, :c])),
        ]
        sexpr = sum(tensors)

        path = einexpr(HyPar(imbalances = 420:420), sexpr)

        @test path isa SizedEinExpr

        @test mapreduce(flops, +, Branches(path)) <= 108
    end

    @testset begin
        tensors = [
            EinExpr([:F, :P, :V], Dict(:P => 5, :F => 8, :V => 5)),
            EinExpr([:T, :Y, :V, :X, :N, :B], Dict(:T => 5, :N => 2, :B => 5, :V => 5, :Y => 7, :X => 8)),
            EinExpr([:L, :K, :S], Dict(:K => 8, :L => 7, :S => 5)),
            EinExpr([:M, :J, :Q, :O], Dict(:M => 5, :J => 7, :Q => 7, :O => 6)),
            EinExpr([:c], Dict(:c => 2)),
            EinExpr([:I, :U, :E], Dict(:U => 8, :I => 5, :E => 6)),
            EinExpr([:N, :C], Dict(:N => 2, :C => 4)),
            EinExpr([:a, :K], Dict(:a => 3, :K => 8)),
            EinExpr([:d, :E, :M], Dict(:M => 5, :d => 6, :E => 6)),
            EinExpr([:B, :b, :D, :H, :L], Dict(:b => 5, :H => 7, :D => 8, :B => 5, :L => 7)),
            EinExpr([:c, :P, :X, :Q], Dict(:P => 5, :Q => 7, :c => 2, :X => 8)),
            EinExpr([:G], Dict(:G => 6)),
            EinExpr([:Z, :W], Dict(:Z => 9, :W => 6)),
            EinExpr([:Y, :H, :S], Dict(:H => 7, :S => 5, :Y => 7)),
            EinExpr([:O, :F, :b, :I], Dict(:b => 5, :I => 5, :F => 8, :O => 6)),
            EinExpr([:A, :J, :T, :G], Dict(:T => 5, :A => 6, :J => 7, :G => 6)),
            EinExpr([:Z, :D, :R], Dict(:Z => 9, :R => 8, :D => 8)),
            EinExpr([:R, :U], Dict(:U => 8, :R => 8)),
            EinExpr([:A, :W], Dict(:A => 6, :W => 6)),
            EinExpr([:a, :C, :d], Dict(:a => 3, :d => 6, :C => 4)),
        ]
        sexpr = sum(tensors)

        path = einexpr(HyPar(imbalances = 450:450), sexpr)

        @test path isa SizedEinExpr

        @test mapreduce(flops, +, Branches(path)) <= 19099592
    end
end
