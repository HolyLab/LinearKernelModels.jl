module LinearKernelModels

using OffsetArrays, LinearAlgebra

export solve_for_kernel, compute_Cd, compute_r, compute_kerr

"""
    C, d = compute_Cd(S::AbstractVecOrMat, r::AbstractVector, trange::AbstractUnitRange)
    C, d = compute_Cd(S::AbstractVecOrMat, r::AbstractVector, nt::Integer)

Given one or more stimuli, where `S[t, i]` is the strength of the
`i`th stimulus at time `t`, and a response vs. time `r[t]`, calculate
`C` and `d` that can be used to solve for a kernel-matrix `k`, where
`k` convolved with `S` best reconstructs `r`. In particular,

    k = reshape(C \\ d, (nt, nstim))

`nt` (an integer) is the number of timepoints in the kernel.
If the kernel is allowed to look into `r`'s future, instead supply `trange = tpast:tfuture`
where `tpast` might typically be negative and `tfuture` typically positive.
(The meaning is the offset relative to the current moment `t=0`.)

If you have temporal gaps in your measured response, splice the blocks
together into a single vector, separating the blocks by stretches of
NaNs of length `nt` or larger. All entries in `S` should be
finite and represent the actual value of the stimulus at these
times. If you have periods of uncertainty about the stimulus history,
then for any time points within `nt` after these unknown
blocks, replace the corresponding entries in `r` with NaN. You can
then set the unknown entries in `S` to an arbitrary value (e.g.,
zero).

See also [`compute_r`](@ref), [`compute_kerrs`](@ref).
"""
function compute_Cd(S::AbstractVecOrMat, r::AbstractVector, trange::AbstractUnitRange)
    T, nstim = size(S, 1), size(S, 2)
    axes(S, 1) == 1:T || error("S must start indexing at t=1, got $(axes(S))")
    axes(r) == (1:T,) || error("r must agree with S's indexing, got $(axes(r)) versus $(axes(S))")
    kernelaxes = (trange, oftype(trange, Base.OneTo(nstim)))
    L = prod(length.(kernelaxes))
    C = zeros(L, L)
    LI = LinearIndices(OffsetArrays.IdentityUnitRange.(kernelaxes))
    w = (!).(isnan.(r))
    for i1 = 1:nstim, i2 = i1:nstim
        for τ1 in trange, τ2 in trange
            s = 0.0
            for t = max(1, 1-τ1, 1-τ2):min(T, T-τ1, T-τ2)
                s += w[t] * S[t+τ1, i1] * S[t+τ2, i2]
            end
            ind1, ind2 = LI[τ1, i1], LI[τ2, i2]
            C[ind1, ind2] = C[ind2, ind1] = s
        end
    end
    d = zeros(L)
    for i = 1:nstim
        for τ in trange
            s = 0.0
            for t = max(1, 1-τ):min(T, T-τ)
                p = r[t]*S[t+τ, i]
                s += ifelse(w[t], p, zero(p))
            end
            d[LI[τ, i]] = s
        end
    end
    return C, d
end
compute_Cd(S::AbstractVecOrMat, r::AbstractVector, nt::Integer) =
    compute_Cd(S, r, -(nt-1):0)

"""
    r = compute_r(S::AbstractVecOrMat, k::AbstractVecOrMat)

Given one or more stimuli, where `S[t, i]` is the strength of the
`i`th stimulus at time `t`, and a kernel `k[τ, i]` for each
stimulus, compute the summed-convolution between `S` and `k` to
predict the response `r`.
"""
function compute_r(S, k)
    T, nstim = size(S, 1), size(S, 2)
    trange = axes(k, 1)
    @assert(axes(k, 2) == 1:nstim)
    r = zeros(T)
    for i = 1:nstim
        for τ in trange
            kτi = k[τ,i]
            for t = max(1, 1-τ):min(T, T-τ)
                r[t] += S[t+τ,i]*kτi
            end
        end
    end
    return r
end

"""
    k = solve_for_kernel(S, r, isnz; rtol=sqrt(eps()))

Solve for a kernel, imposing constraints on entries that are allowed to be non-zero.
`S` is the stimulus and `r` the response vector. (See [`compute_Cd`](@ref) for more information.)
`isnz` must have the axes of the kernel, and be true in every slot where the solved kernel
may be non-zero. If you have no need for constraints, supply `trues(axs)` where `axs`
encodes the axes of the kernel.

# Example
```
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
"""
function solve_for_kernel(S::AbstractVecOrMat, r::AbstractVector, isnz::AbstractVecOrMat{Bool}; rtol=sqrt(eps()))
    trange = axes(isnz, 1)
    C, d = compute_Cd(S, r, trange)
    nconstrained = prod(length.(axes(isnz))) - sum(isnz)
    Q = zeros(size(C,1), nconstrained)
    col = 0
    for (i, isunconstrained) in enumerate(isnz)
        if !isunconstrained
            col += 1
            Q[i,col] = 1
        end
    end
    A = [C Q; Q' zeros(nconstrained, nconstrained)]
    dcat = [d; zeros(nconstrained)]
    F = svd(A)
    S = F.S
    Smax = maximum(S)
    flag = S .< rtol*Smax
    S[flag] .= Inf
    soln = F \ dcat
    reshape(soln[1:length(d)], axes(isnz))
end

"""
    kerr = compute_kerr(k, C, S, r)

Compute the estimated standard error in the parameters `k` of the
kernels, where `C` is determined from [`compute_Cd`](@ref), `S` is the
stimulus, and `r` the response.
"""
function compute_kerr(k, C, S, r)
    rbar = compute_r(S, k)
    N = 0
    Σ = 0.0
    for t = 1:length(r)
        rt = r[t]
        if !isnan(rt)
            Σ += (rt - rbar[t])^2
            N += 1
        end
    end
    lk = prod(length.(axes(k)))
    σ2 = Σ/(N-lk)  # replace with length(k)
    return reshape(sqrt(σ2) * sqrt.(diag(inv(C))), axes(k))
end

end # module
