using LinearKernelModels
using LinearAlgebra, OffsetArrays
using Test

@testset "Single stimulus" begin
    S = rand(100)
    ktrue = OffsetArray(randn(5), -4:0)
    r = compute_r(S, ktrue)
    C, d = compute_Cd(S, r, 5)
    kcalc = reshape(C \ d, axes(ktrue))
    @test kcalc ≈ ktrue
    nfailed = 0
    for i = 1:100
        rnoisy = r .+ randn(size(r))
        C, d = compute_Cd(S, rnoisy, 5)
        knoisy = reshape(C \ d, axes(ktrue))
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
        knoisy = reshape(C \ d, axes(ktrue))
        kerr = compute_kerr(knoisy, C, S, rnoisy)
        nfailed += any((knoisy .- ktrue).^2 .> 3*kerr)
    end
    @test nfailed <= 5
end

@testset "Single acausal stimulus" begin
    S = rand(100)
    ktrue = OffsetArray(randn(5), -2:2)
    r = compute_r(S, ktrue)
    C, d = compute_Cd(S, r, axes(ktrue, 1))
    kcalc = reshape(C \ d, axes(ktrue))
    @test kcalc ≈ ktrue
end

@testset "Multiple stimuli and NaNs" begin
    S = rand(100, 3)
    ktrue = OffsetArray(randn(5, 3), -4:0, 1:3)
    r = compute_r(S, ktrue)
    C, d = compute_Cd(S, r, 5)
    kcalc = reshape(C \ d, axes(ktrue))
    @test kcalc ≈ ktrue

    r[5:10] .= NaN
    r[50:52] .= NaN
    C, d = compute_Cd(S, r, 5)
    kcalc = reshape(C \ d, axes(ktrue))
    @test kcalc ≈ ktrue
end

@testset "Multiple acausal stimuli and NaNs" begin
    S = rand(100,3)
    ktrue = OffsetArray(randn(5,3), -2:2, 1:3)
    r = compute_r(S, ktrue)
    C, d = compute_Cd(S, r, -2:2)
    kcalc = reshape(C \ d, axes(ktrue))
    @test kcalc ≈ ktrue

    r[5:10] .= NaN
    r[50:52] .= NaN
    C, d = compute_Cd(S, r, -2:2)
    kcalc = reshape(C \ d, axes(ktrue))
    @test kcalc ≈ ktrue
end

@testset "Single stimulus with constraints" begin
    S = rand(100)
    ktrue = OffsetArray([rand(3); 0; 0], -2:2)
    r = compute_r(S, ktrue)
    kcalc = solve_for_kernel(S, r, ktrue.!=0)
    @test kcalc ≈ ktrue
end

@testset "Multiple stimuli with constraints" begin
    S = rand(100,3)
    ktrue = OffsetArray(randn(5,3), -2:2, 1:3)
    ktrue[rand(1:15,5,1)] .= 0
    r = compute_r(S,ktrue)
    kcalc = solve_for_kernel(S, r, ktrue.!=0)
    @test kcalc ≈ ktrue

    # With a degeneracy that is broken by the constraint
    S = zeros(20, 2)
    S[10,:] .= 1
    r = zeros(20)
    r[11:12] .= 0.5
    kmask = OffsetArray([true false; false true], -2:-1, 1:2)
    kcalc = solve_for_kernel(S, r, kmask)
    @test kcalc ≈ OffsetArray([0.5 0; 0 0.5], -2:-1, 1:2)
end

@testset "Multiple stimuli with constraints and Nans" begin
    S = rand(100,7)
    ktrue = OffsetArray(rand(5,7), -2:2, 1:7)
    ktrue[rand(1:35, 10)] .= 0
    r = compute_r(S,ktrue)
    r[5:10] .= NaN
    r[50:52] .= NaN
    kcalc = solve_for_kernel(S, r, ktrue.!=0)
    @test kcalc ≈ ktrue
end
