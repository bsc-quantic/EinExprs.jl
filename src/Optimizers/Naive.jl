using AbstractTrees

struct Naive <: Optimizer end

einexpr(::Naive, path, _) = einexpr(Naive(), path)

function einexpr(::Naive, path)
    hist = Dict(i => count(∋(i) ∘ head, args(path)) for i in hyperinds(path))

    foldl(args(path)) do a, b
        expr = sum([a, b], skip = head(path) ∪ collect(keys(hist)))

        for i in Iterators.filter(∈(keys(hist)), ∩(head(a), head(b)))
            hist[i] -= 1
            hist[i] <= 2 && delete!(hist, i)
        end

        return expr
    end
end

einexpr(::Naive, path::SizedEinExpr) = SizedEinExpr(einexpr(Naive(), path.path), path.size)
