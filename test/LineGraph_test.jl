@testset "LineGraph" begin
    # connected
    network = TensorNetwork([
        Tensor(rand(2, 2), (:i, :m)),
        Tensor(rand(2, 2, 2), (:i, :j, :p)),
        Tensor(rand(2, 2, 2), (:n, :j, :k)),
        Tensor(rand(2, 2, 2), (:p, :k, :l)),
        Tensor(rand(2, 2, 2), (:m, :n, :o)),
        Tensor(rand(2, 2), (:o, :l)),
    ])

    path1 = einexpr(network; optimizer = Greedy())
    path2 = einexpr(network; optimizer = LineGraph())
    @test mapreduce(flops, +, Branches(path1)) >= mapreduce(flops, +, Branches(path2)) - 10
    @test contract(network; path = path1) ≈ contract(network; path = path2)

    path1 = einexpr(network; optimizer = Greedy(), outputs = [:i, :p])
    path2 = einexpr(network; optimizer = LineGraph(), outputs = [:i, :p])
    @test mapreduce(flops, +, Branches(path1)) >= mapreduce(flops, +, Branches(path2)) - 10
    @test contract(network; path = path1) ≈ contract(network; path = path2)

    # unconnected
    network = TensorNetwork([
        Tensor(rand(2, 2), (:i, :j)),
        Tensor(rand(2, 2), (:i, :j)),
        Tensor(rand(2, 2, 2), (:k, :l, :m)),
        Tensor(rand(2, 2, 2), (:k, :l, :m)),
    ])

    path1 = einexpr(network; optimizer = Greedy())
    path2 = einexpr(network; optimizer = LineGraph())
    @test mapreduce(flops, +, Branches(path1)) >= mapreduce(flops, +, Branches(path2)) - 10
    @test contract(network; path = path1) ≈ contract(network; path = path2)

    path1 = einexpr(network; optimizer = Greedy(), outputs = [:i])
    path2 = einexpr(network; optimizer = LineGraph(), outputs = [:i])
    @test mapreduce(flops, +, Branches(path1)) >= mapreduce(flops, +, Branches(path2)) - 10
    @test contract(network; path = path1) ≈ contract(network; path = path2)

    path1 = einexpr(network; optimizer = Greedy(), outputs = [:i, :k])
    path2 = einexpr(network; optimizer = LineGraph(), outputs = [:i, :k])
    @test mapreduce(flops, +, Branches(path1)) >= mapreduce(flops, +, Branches(path2)) - 10
    @test contract(network; path = path1) ≈ contract(network; path = path2)
end
