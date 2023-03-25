using AbstractTrees

struct EinExpr
    expr::Expr
    # TODO sizes or tensors

    function EinExpr(output, inputs)
        new(Expr(:call, :einsum, output, inputs))
    end
end

size(einexpr::EinExpr, i::Symbol) = ... # TODO

uncontractedinds(einexpr::EinExpr) = ... # TODO

flops(einexpr::EinExpr) = ...