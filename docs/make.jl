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
        "Optimizers"=>["Exhaustive" => "optimizers/exhaustive.md", "Greedy" => "optimizers/greedy.md"],
        "Alternatives"=>"alternatives.md",
    ],
    format = Documenter.HTML(; assets = ["assets/style/images.css"]),
)
