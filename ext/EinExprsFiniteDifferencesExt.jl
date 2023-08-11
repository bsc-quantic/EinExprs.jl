module EinExprsFiniteDifferencesExt

if isdefined(Base, :get_extension)
    using EinExprs
else
    using ..EinExprs
end

using FiniteDifferences

function FiniteDifferences.to_vec(expr::EinExpr)
    x_vecs_and_backs = map(to_vec, expr.args)
    x_vecs, x_backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    lengths = map(length, x_vecs)
    sz = typeof(lengths)(cumsum(collect(lengths)))
    function EinExpr_from_vec(v)
        args = map(x_backs, lengths, sz) do x_back, l, s
            return x_back(v[s-l+1:s])
        end
        EinExpr(head(expr), args)
    end
    return reduce(vcat, x_vecs), EinExpr_from_vec
end

end
