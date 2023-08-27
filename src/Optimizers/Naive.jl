struct Naive <: Optimizer end

einexpr(::Naive, path) = foldl((a, b) -> sum([a, b]), path.args)
