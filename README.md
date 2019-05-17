# LinearKernelModels

This package solves for multilinear kernels (i.e., multilinear regression) that satisfy a
least-square error criterion. Given a stimulus `S`, a matrix specifying the `i`th input
at time `τ` as `S[τ, i]`, and a response vector `r` such that `r[τ]` is the response at
time `τ`, find a multilinear kernel `k[τ,i]` such that

The following demo is taken from the help for `solve_for_kernel`:

```julia
julia> using LinearKernelModels, OffsetArrays

julia> S = rand(100,3);

julia> ktrue = OffsetArray(rand(5, 3), -4:0, 1:3)
OffsetArray(::Array{Float64,2}, -4:0, 1:3) with eltype Float64 with indices -4:0×1:3:
 0.155762  0.877222  0.457633
 0.88788   0.355702  0.0268748
 0.276609  0.914133  0.230344
 0.718295  0.158363  0.0402396
 0.195513  0.783832  0.5007

julia> ktrue[-4, 1] = 0
0

julia> isnz = ktrue .!= 0
OffsetArray(::BitArray{2}, -4:0, 1:3) with eltype Bool with indices -4:0×1:3:
 false  true  true
  true  true  true
  true  true  true
  true  true  true
  true  true  true

julia> r = compute_r(S, ktrue);

julia> k = solve_for_kernel(S, r, isnz)
OffsetArray(::Array{Float64,2}, -4:0, 1:3) with eltype Float64 with indices -4:0×1:3:
 6.99441e-15  0.877222  0.457633
 0.88788      0.355702  0.0268748
 0.276609     0.914133  0.230344
 0.718295     0.158363  0.0402396
 0.195513     0.783832  0.5007
```
