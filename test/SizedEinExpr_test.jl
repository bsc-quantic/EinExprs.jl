@testset "SizedEinExpr" begin
    using LinearAlgebra

    tensor = EinExpr([:i, :j])
    expr = EinExpr([:i, :j], [tensor])
    sexpr = SizedEinExpr(expr, Dict(:i => 2, :j => 3))

    @test head(sexpr) === head(expr) === sexpr.head
    @test args(expr) === sexpr.args
    @test args(sexpr) == map(Base.Fix2(SizedEinExpr, Dict(:i => 2, :j => 3)), args(expr))
    @test EinExprs.nargs(sexpr) == EinExprs.nargs(expr)

    @test inds(sexpr) == inds(expr)
    @test ndims(sexpr) == ndims(expr)
    @test length(sexpr) == 6

    @test size(sexpr, :i) == 2
    @test size(sexpr, :j) == 3
    @test size(sexpr) == (2, 3)

    @test select(sexpr, :i) == SizedEinExpr[sexpr, SizedEinExpr(tensor, Dict(:i => 2, :j => 3))]
    @test select(sexpr, :j) == SizedEinExpr[sexpr, SizedEinExpr(tensor, Dict(:i => 2, :j => 3))]
end
