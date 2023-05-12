using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

push!(LOAD_PATH, "$(@__DIR__)/..")

using Documenter
using EinExprs

DocMeta.setdocmeta!(EinExprs, :DocTestSetup, :(using EinExprs); recursive=true)

makedocs(
    modules=[EinExprs],
    sitename="EinExprs.jl",
    authors="Sergio Sánchez Ramírez and contributors",
    pages=Any[
        "Home"=>"index.md"
    ],
)
