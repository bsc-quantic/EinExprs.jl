module EinExprsMakieExt

function __init__()
    try
        Base.require(Main, :GraphMakie)
    catch
        @warn """Package GraphMakie not found in current path. It is needed to plot `EinExpr`s with `Makie`.
        - Run `import Pkg; Pkg.add(\"GraphMakie\")` or `]add GraphMakie` to install the GraphMakie package, then restart julia.
        """
    end
end

end
