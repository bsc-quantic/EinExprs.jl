using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

push!(LOAD_PATH, "$(@__DIR__)/..")

using Documenter
using EinExprs

DocMeta.setdocmeta!(EinExprs, :DocTestSetup, :(using EinExprs); recursive = true)

makedocs(
    modules = [EinExprs],
    sitename = "EinExprs.jl",
    authors = "Sergio Sánchez Ramírez and contributors",
    pages = Any[
        "Home"=>"index.md",
        "Einsum Expressions"=>"einexpr.md",
        "Resource counting"=>"counters.md",
        "Optimizers"=>["Exhaustive" => "optimizers/exhaustive.md", "Greedy" => "optimizers/greedy.md", "LineGraph" => "optimizers/line_graph.md", "HyPar" => "optimizers/hy_par.md", "NesDis" => "optimizers/nes_dis.md"],
        "Slicing"=>"slicing.md",
        "Alternatives"=>"alternatives.md",
    ],
    format = Documenter.HTML(; assets = ["assets/style/images.css", "assets/favicon.ico"], sidebar_sitename = false),
    checkdocs = :exports,
    warnonly = true,
)

deploydocs(repo = "github.com/bsc-quantic/EinExprs.jl.git", devbranch = "master", push_preview = true)
