struct Naive <: Optimizer end

function einexpr(::Naive, path)
    hist = Dict(i => count(∋(i) ∘ head, path.args) for i in hyperinds(path))

    foldl(path.args) do a, b
        expr = sum([a, b], skip = path.head ∪ collect(keys(hist)))

        for i in Iterators.filter(∈(keys(hist)), ∩(head(a), head(b)))
            hist[i] -= 1
            hist[i] <= 2 && delete!(hist, i)
        end

        return expr
    end
end
