var documenterSearchIndex = {"docs":
[{"location":"optimizers/greedy/#Greedy-optimizer","page":"Greedy","title":"Greedy optimizer","text":"","category":"section"},{"location":"optimizers/greedy/","page":"Greedy","title":"Greedy","text":"EinExprs.Greedy","category":"page"},{"location":"optimizers/greedy/#EinExprs.Greedy","page":"Greedy","title":"EinExprs.Greedy","text":"Greedy(; metric = removedsize, choose = pop!)\n\nGreedy contraction path solver. Greedily selects contractions that maximize a metric.\n\nKeywords\n\nmetric is a function that evaluates candidate pairwise tensor contractions. Defaults to removedsize.\nchoose is a function that extracts a pairwise tensor contraction between candidates. Defaults to candidate that maximize metric using pop!.\nouter If true, consider outer products as candidates. Defaults to false.\n\nImplementation\n\nThe implementation uses a binary heaptree to sort candidate pairwise tensor contractions. Then recursively,\n\nSelects and extracts a candidate from the heaptree using the choose function.\nUpdates the metric of the candidates which contain neighbouring indices to the one selected.\nAppend the selected index to the path and go back to step 1.\n\n\n\n\n\n","category":"type"},{"location":"alternatives/#Alternatives","page":"Alternatives","title":"Alternatives","text":"","category":"section"},{"location":"alternatives/","page":"Alternatives","title":"Alternatives","text":"EinExprs is deeply inspired by opt_einsum and cotengra. Actually most of the contraction path search and slicing algorithms have been rewritten from there. If you happen to be working in Python, you should definetely check out these libraries.","category":"page"},{"location":"alternatives/","page":"Alternatives","title":"Alternatives","text":"info: Differences with `opt_einsum`\nAlthough the differences are minimal, any user coming from opt_einsum should be aware that:The \"optimal\" contraction path solver in opt_einsum is known as Exhaustive in EinExprs.\nThe \"random-greedy\" contraction path solver in opt_einsum is the Greedy optimizer in EinExprs but with a random choose function.\nWhen counting FLOPs, opt_einsum gives a value times 2 higher than the EinExprs.flops counter.[1]","category":"page"},{"location":"alternatives/","page":"Alternatives","title":"Alternatives","text":"[1]: We are not sure of the reason behind this mismatch or which package gives the correct answer, but since the factor remains constant, it should not affect when comparing contraction paths during the minimization step.","category":"page"},{"location":"alternatives/","page":"Alternatives","title":"Alternatives","text":"Although we believe there is no similar project in the Julia world, there are some overlapping libraries that may suit you if EinExprs doesn't fit your case.","category":"page"},{"location":"alternatives/","page":"Alternatives","title":"Alternatives","text":"TensorOperations\nOMEinsum and OMEinsumContractionOrders","category":"page"},{"location":"einexpr/#Einsum-Expressions","page":"Einsum Expressions","title":"Einsum Expressions","text":"","category":"section"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"Let's take an arbitrary tensor network like the following.","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"<img class=\"invert-on-dark\" src=\"../assets/tensor-network.excalidraw.svg\" alt=\"An arbitrary tensor network\"/>","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"This graph is a graphical notation equivalent to the following equation.","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"sum_i j k l m n o p A_mi B_ijp C_jkn D_pkl E_mno F_ol","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"A naïve implementation of this equation is easy to implement.","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"result = zero(reduce(promote_type, eltype.([A,B,C,D,E,F])))\nfor (i,j,k,l,m,n,o,p) in Iterators.product(1:I, 1:J, 1:K, 1:L, 1:M, 1:N, 1:O, 1:P)\n    result += A[m,i] * B[i,j,p] * C[j,k,n] * D[p,k,l] * E[m,n,o] * F[o,l]\nend","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"But it has a cost of prod_alpha dim(alpha) where alpha in ijklmnop which is of mathcalO(exp(n)) time complexity.","category":"page"},{"location":"einexpr/#Basic-utilities","page":"Einsum Expressions","title":"Basic utilities","text":"","category":"section"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"The EinExpr type has a stable definition: it's secure to access its fields using the dot-syntax (e.g. path.head), but we recommend to use the following methods instead. They provide a much more stable API and can be used in a functional manner.","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"EinExprs.head\nEinExprs.args\nBase.size(::EinExpr)","category":"page"},{"location":"einexpr/#EinExprs.head","page":"Einsum Expressions","title":"EinExprs.head","text":"head(path::EinExpr)\n\nReturn the indices of the resulting tensor from contracting path.\n\nSee also: inds, args.\n\n\n\n\n\n","category":"function"},{"location":"einexpr/#EinExprs.args","page":"Einsum Expressions","title":"EinExprs.args","text":"args(path::EinExpr)\n\nReturn the children of the path, which correspond to input tensors for the contraction step in the top of the path.\n\nSee also: head.\n\n\n\n\n\nargs(sexpr::SizedEinExpr)\n\nNote\n\nUnlike args(::EinExpr), this function returns SizedEinExpr objects.\n\n\n\n\n\n","category":"function"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"Some other useful methods are:","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"Base.ndims(::EinExpr)\nEinExprs.inds\nEinExprs.suminds\nEinExprs.parsuminds\nEinExprs.contractorder\nEinExprs.select\nEinExprs.neighbours","category":"page"},{"location":"einexpr/#Base.ndims-Tuple{EinExpr}","page":"Einsum Expressions","title":"Base.ndims","text":"ndims(path::EinExpr)\n\nReturn the number of indices of the resulting tensor from contracting path.\n\n\n\n\n\n","category":"method"},{"location":"einexpr/#EinExprs.inds","page":"Einsum Expressions","title":"EinExprs.inds","text":"inds(path)\n\nReturn all the involved indices in path. If a tensor is passed, then it is equivalent to calling head.\n\nSee also: head.\n\n\n\n\n\n","category":"function"},{"location":"einexpr/#EinExprs.suminds","page":"Einsum Expressions","title":"EinExprs.suminds","text":"suminds(path)\n\nIndices of summation of an EinExpr.\n\nmathttpath equiv sum_j k l m n o p A_mi B_ijp C_jkn D_pkl E_mno F_ol\n\nsuminds(path) == [:j, :k, :l, :m, :n, :o, :p]\n\n\n\n\n\n","category":"function"},{"location":"einexpr/#EinExprs.parsuminds","page":"Einsum Expressions","title":"EinExprs.parsuminds","text":"parsuminds(path)\n\nIndices of summation of possible pairwise tensors contractions between children of path.\n\n\n\n\n\n","category":"function"},{"location":"einexpr/#EinExprs.contractorder","page":"Einsum Expressions","title":"EinExprs.contractorder","text":"contractorder(path::EinExpr)\n\nTransform path into a contraction order.\n\n\n\n\n\n","category":"function"},{"location":"einexpr/#EinExprs.select","page":"Einsum Expressions","title":"EinExprs.select","text":"select(path::EinExpr, i)\n\nReturn the child elements that contain i indices.\n\n\n\n\n\n","category":"function"},{"location":"einexpr/#EinExprs.neighbours","page":"Einsum Expressions","title":"EinExprs.neighbours","text":"neighbours(path::EinExpr, i)\n\nReturn the indices neighbouring to i.\n\n\n\n\n\n","category":"function"},{"location":"einexpr/#Construction-by-summation","page":"Einsum Expressions","title":"Construction by summation","text":"","category":"section"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"One option for constructing EinExprs manually is to use the sum methods. For example, imagine that we have the following tensor equation and we want to contract first tensors E_mno and F_ol. The resulting equation would be equivalent to adding a summatory to E and F as written in the right-hand side.","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"sum_i j k l m n o p A_mi B_ijp C_jkn D_pkl E_mno F_ol = sum_i j k l m n p A_mi B_ijp C_jkn D_pkl sum_o E_mno F_ol","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"In EinExprs, we advocate for code that it's almost as easy as writing math. As such, one can write sum([E, F]) to create a new EinExpr where common indices are contracted or sum!(path, :o) for the in-place version where E and F are children of path.","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"Base.sum!(::EinExpr, ::Any)\nBase.sum(::EinExpr, ::Union{Symbol, Tuple{Vararg{Symbol}}, AbstractVector{<:Symbol}})","category":"page"},{"location":"einexpr/#Base.sum!-Tuple{EinExpr, Any}","page":"Einsum Expressions","title":"Base.sum!","text":"sum!(path, indices)\n\nExplicit, in-place sum over indices.\n\nSee also: sum, suminds.\n\n\n\n\n\n","category":"method"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"In order to reverse the operation and unfix the contraction, the user may call the collapse! function.","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"EinExprs.collapse!","category":"page"},{"location":"einexpr/#EinExprs.collapse!","page":"Einsum Expressions","title":"EinExprs.collapse!","text":"collapse!(path::EinExpr)\n\nCollapses all sub-branches, merging all tensor leaves into the args field.\n\n\n\n\n\n","category":"function"},{"location":"einexpr/#AbstractTrees-integration","page":"Einsum Expressions","title":"AbstractTrees integration","text":"","category":"section"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"EinExpr type integrates with the AbstractTrees package in order to implement some of the tree-traversing algorithms. The interface is public and thus any user can use it to implement their own methods.","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"For example, the AbstractTrees.Leaves function returns an iterator through the leaves of any tree; i.e. the initial tensors in our case. We implement the Branches function in order to walk through the non-terminal nodes; i.e. the intermediate tensors.","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"EinExprs.Branches","category":"page"},{"location":"einexpr/#EinExprs.Branches","page":"Einsum Expressions","title":"EinExprs.Branches","text":"Branches(path::EinExpr)\n\nIterator that walks through the non-terminal nodes of the path tree.\n\nSee also: branches.\n\n\n\n\n\n","category":"function"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"EinExprs exports a variant of these methods which return collections.","category":"page"},{"location":"einexpr/","page":"Einsum Expressions","title":"Einsum Expressions","text":"EinExprs.leaves\nEinExprs.branches","category":"page"},{"location":"einexpr/#EinExprs.leaves","page":"Einsum Expressions","title":"EinExprs.leaves","text":"leaves(path::EinExpr[, i])\n\nReturn the terminal leaves of the path, which correspond to the initial input tensors. If i is specified, then only return the i-th tensor.\n\nSee also: branches.\n\n\n\n\n\n","category":"function"},{"location":"einexpr/#EinExprs.branches","page":"Einsum Expressions","title":"EinExprs.branches","text":"branches(path::EinExpr[, i])\n\nReturn the non-terminal branches of the path, which correspond to intermediate tensors result of contraction steps. If i is specified, then only return the i-th EinExpr.\n\nSee also: leaves, Branches.\n\n\n\n\n\n","category":"function"},{"location":"slicing/#Slicing","page":"Slicing","title":"Slicing","text":"","category":"section"},{"location":"slicing/","page":"Slicing","title":"Slicing","text":"selectdim\nview","category":"page"},{"location":"slicing/#Base.selectdim","page":"Slicing","title":"Base.selectdim","text":"selectdim(path::EinExpr, index, i)\n\nProject index to dimension i in a EinExpr. This is equivalent to tensor cutting aka slicing.\n\nArguments\n\npath Contraction path.\nindex Index to cut.\ni Dimension of index to select.\n\nSee also: view.\n\n\n\n\n\n","category":"function"},{"location":"slicing/#Base.view","page":"Slicing","title":"Base.view","text":"view(path::EinExpr, cuttings...)\n\nProject indices in contraction path to some of its dimensions. This is equivalent to:\n\nreduce(cuttings) do path, (index, i)\n    selectdim(path, index, i)\nend\n\nArguments\n\npath Target contraction path.\ncuttings List of Pair{Symbol,Int} representing the tensor cuttings aka slices.\n\nSee also: selectdim.\n\n\n\n\n\n","category":"function"},{"location":"slicing/","page":"Slicing","title":"Slicing","text":"findslices","category":"page"},{"location":"slicing/#EinExprs.findslices","page":"Slicing","title":"EinExprs.findslices","text":"findslices(scorer, path::EinExpr; size, slices, overhead, temperature = 0.01, skip = head(path))\n\nSearch for indices to be cut/sliced such that the conditions given by size, overhead and slices are fulfilled. Reimplementation based on contengra's SliceFinder algorithm.\n\nArguments\n\nscorer Heuristic function (or functor) that accepts a path and a candidate index for cutting, and returns a score.\npath The contraction path target for tensor cutting aka slicing.\n\nKeyword Arguments\n\nsize If specified, the largest intermediate tensor of the slice won't surpass this size (in number of elements).\nslices If specified, there will be at least slices different slices when cutting all returnt indices.\noverhead If specified, the amount of redundant operations between a slice and the original contraction won't supass this ratio.\ntemperature Temperature of the Boltzmann-like noise added for diffusing results.\nskip Indices not to be considered for slicing.\n\n\n\n\n\n","category":"function"},{"location":"slicing/","page":"Slicing","title":"Slicing","text":"FlopsScorer\nSizeScorer","category":"page"},{"location":"slicing/#EinExprs.FlopsScorer","page":"Slicing","title":"EinExprs.FlopsScorer","text":"FlopsScorer\n\nKeyword Arguments\n\nweight\n\n\n\n\n\n","category":"type"},{"location":"slicing/#EinExprs.SizeScorer","page":"Slicing","title":"EinExprs.SizeScorer","text":"SizeScorer\n\nKeyword Arguments\n\nweight\n\n\n\n\n\n","category":"type"},{"location":"optimizers/exhaustive/#exhaustive_optimizer","page":"Exhaustive","title":"Exhaustive optimizer","text":"","category":"section"},{"location":"optimizers/exhaustive/","page":"Exhaustive","title":"Exhaustive","text":"EinExprs.Exhaustive","category":"page"},{"location":"optimizers/exhaustive/#EinExprs.Exhaustive","page":"Exhaustive","title":"EinExprs.Exhaustive","text":"Exhaustive(; outer = false)\n\nExhaustive contraction path optimizers. It guarantees to find the optimal contraction path but at a large cost.\n\nKeywords\n\nouter instructs to consider outer products (aka tensor products) on the search for the optimal contraction path. It rarely provides an advantage over only considering inner products and thus, it is false by default.\n\nwarning: Warning\nThe functionality of outer = true has not been yet implemented.\n\nImplementation\n\nThe algorithm has a mathcalO(n) time complexity if outer = true and mathcalO(exp(n)) if outer = false.\n\n\n\n\n\n","category":"type"},{"location":"counters/#Resource-counting","page":"Resource counting","title":"Resource counting","text":"","category":"section"},{"location":"counters/","page":"Resource counting","title":"Resource counting","text":"As explained before, EinExprs are symbolic expressions representing Einstein summation equations (i.e. tensor summation, permutation, contraction, ...) so no tensor operation is actually performed on them. Many times, information about the execution cost is needed to optimize the contraction path. In general, this is a hard task but thanks to Einsteam summation notation only representing linear algebra operations, and these operations are easy to estimate, we can count the resource requirements of any contraction path.","category":"page"},{"location":"counters/","page":"Resource counting","title":"Resource counting","text":"Currently there are 3 resource counters:","category":"page"},{"location":"counters/","page":"Resource counting","title":"Resource counting","text":"flops\nremovedsize\nEinExprs.removedrank","category":"page"},{"location":"counters/#EinExprs.flops","page":"Resource counting","title":"EinExprs.flops","text":"flops(path::EinExpr)\n\nCount the number of mathematical operations will be performed by the contraction of the root of the path tree.\n\n\n\n\n\n","category":"function"},{"location":"counters/#EinExprs.removedsize","page":"Resource counting","title":"EinExprs.removedsize","text":"removedsize(path::EinExpr)\n\nCount the amount of memory that will be freed after performing the contraction of the root of the path tree.\n\n\n\n\n\n","category":"function"},{"location":"counters/#EinExprs.removedrank","page":"Resource counting","title":"EinExprs.removedrank","text":"removedrank(path::EinExpr)\n\nCount the rank reduction after performing the contraction of the root of the path tree.\n\n\n\n\n\n","category":"function"},{"location":"counters/","page":"Resource counting","title":"Resource counting","text":"tip: Tip\nThese methods only count the resources spent of the contraction on the root of the tree. In order to count the resources of the whole tree, use mapreduce with Branches:mapreduce(flops, +, Branches(path))","category":"page"},{"location":"","page":"Home","title":"Home","text":"<img class=\"light-mode-only\" src=\"assets/logo.svg\"/>\n<img class=\"dark-mode-only\" src=\"assets/logo-dark.svg\"/>","category":"page"},{"location":"","page":"Home","title":"Home","text":"EinExprs is a Julia package that provides EinExprs: symbolic expressions representing a Einstein summation. These summations may be used to represent contraction paths of large tensor networks.","category":"page"},{"location":"","page":"Home","title":"Home","text":"It is a complete redesign of OptimizedEinsum, which indeed was a Julia fork of opt_einsum. It powers Tenet but can easily be adapted to work with other packages.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Einsum Expressions\nOptimizers\nResource counting\nSlicing\nAlternatives","category":"page"},{"location":"#Planned-features","page":"Home","title":"Planned features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Optimizers\nHypergraph Partitioning\nBranch & Bound\nDynamic Programming\nSubtree reconfiguration\nResource counting\nArray structure-aware analysis\nExecution planning\nPermutation order optimization\nCompilation","category":"page"}]
}
