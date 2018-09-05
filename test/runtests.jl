using LinearKernelModels
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

@testset "Single stimulus" begin
    S = rand(100)
    ktrue = randn(5)
    r = compute_r(S, ktrue)
    C, d = compute_Cd(S, r, 5)
    kcalc = C \ d
    @test kcalc ≈ ktrue
    nfailed = 0
    for i = 1:100
        rnoisy = r .+ randn(size(r))
        C, d = compute_Cd(S, rnoisy, 5)
        knoisy = C \ d
        kerr = compute_kerr(knoisy, C, S, rnoisy)
        nfailed += any((knoisy .- ktrue).^2 .> 3*kerr)
    end
    @test nfailed <= 5

    nfailed = 0
    for i = 1:100
        S = rand(10000)
        r = compute_r(S, ktrue)
        rnoisy = r .+ randn(size(r))
        C, d = compute_Cd(S, rnoisy, 5)
        knoisy = C \ d
        kerr = compute_kerr(knoisy, C, S, rnoisy)
        nfailed += any((knoisy .- ktrue).^2 .> 3*kerr)
    end
    @test nfailed <= 5
end

@testset "Multiple stimuli and NaNs" begin
    S = rand(100, 3)
    ktrue = randn(5, 3)
    r = compute_r(S, ktrue)
    C, d = compute_Cd(S, r, 5)
    kcalc = reshape(C \ d, (5, 3))
    @test kcalc ≈ ktrue

    r[5:10] = r[50:52] = NaN
    C, d = compute_Cd(S, r, 5)
    kcalc = reshape(C \ d, (5, 3))
    @test kcalc ≈ ktrue
end