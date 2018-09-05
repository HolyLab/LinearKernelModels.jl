module LinearKernelModels

export compute_Cd, compute_r, compute_kerr

"""
    C, d = compute_Cd(S::AbstractVecOrMat, r::AbstractVector, nt)

Given one or more stimuli, where `S[t, i]` is the strength of the
`i`th stimulus at time `t`, and a response vs. time `r[t]`, calculate
`C` and `d` that can be used to solve for a kernel-matrix `k`, where
`k` convolved with `S` best reconstructs `r`. In particular,

    k = reshape(C \ d, (nt, nstim))

`nt` (an integer) is the number of timepoints in the kernel.

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
function compute_Cd(S::AbstractVecOrMat, r::AbstractVector, nt::Integer)
    T, nstim = size(S, 1), size(S, 2)
    @assert(length(r) == T)
    blocksz = (nt, nstim)
    L = nstim*nt
    C = zeros(L, L)
    w = collect((!).(isnan.(r)))
    for i1 = 1:nstim, i2 = i1:nstim
        for τ1 = 0:nt-1, τ2 = 0:nt-1
            s = 0.0
            @inbounds @simd for t = max(τ1, τ2) + 1:T
                s += w[t] * S[t-τ1, i1] * S[t-τ2, i2]
            end
            ind1, ind2 = sub2ind(blocksz, τ1+1, i1), sub2ind(blocksz, τ2+1, i2)
            C[ind1, ind2] = C[ind2, ind1] = s
        end
    end
    d = zeros(L)
    for i = 1:nstim
        for τ = 0:nt-1
            s = 0.0
            @inbounds @simd for t = τ+1:T
                p = r[t]*S[t-τ, i]
                s += ifelse(w[t], p, zero(p))
            end
            d[sub2ind(blocksz, τ+1, i)] = s
        end
    end
    return C, d
end

"""
    r = compute_r(S::AbstractVecOrMat, k::AbstractVecOrMat)

Given one or more stimuli, where `S[t, i]` is the strength of the
`i`th stimulus at time `t`, and a kernel `k[τ, i]` for each
stimulus, compute the summed-convolution between `S` and `k` to
predict the response `r`.
"""
function compute_r(S, k)
    T, nstim = size(S, 1), size(S, 2)
    nt = size(k, 1)
    @assert(size(k, 2) == nstim)
    r = zeros(T)
    for i = 1:nstim
        for τ = 0:nt-1
            kτi = k[τ+1,i]
            @inbounds @simd for t = τ+1:T
                r[t] += S[t-τ,i]*kτi
            end
        end
    end
    return r
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
    σ2 = Σ/(N-length(k))
    return reshape(sqrt(σ2) * sqrt.(diag(inv(C))), size(k))
end

end # module
