using LinearKernelModels
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end
using OffsetArrays

@testset "Single stimulus" begin
    S = rand(100)
    ktrue = OffsetArray(randn(5), -4:0)
    r = compute_r(S, ktrue)
    C, d = compute_Cd(S, r, 5)
    kcalc = reshape(C \ d, Compat.axes(ktrue))
    @test kcalc ≈ ktrue
    nfailed = 0
    for i = 1:100
        rnoisy = r .+ randn(size(r))
        C, d = compute_Cd(S, rnoisy, 5)
        knoisy = reshape(C \ d, Compat.axes(ktrue))
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
        knoisy = reshape(C \ d, Compat.axes(ktrue))
        kerr = compute_kerr(knoisy, C, S, rnoisy)
        nfailed += any((knoisy .- ktrue).^2 .> 3*kerr)
    end
    @test nfailed <= 5
end

@testset "Single acausal stimulus" begin
    S = rand(100)
    ktrue = OffsetArray(randn(5), -2:2)
    r = compute_r(S, ktrue)
    C, d = compute_Cd(S, r, Compat.axes(ktrue, 1))
    kcalc = reshape(C \ d, Compat.axes(ktrue))
    @test kcalc ≈ ktrue
end

@testset "Multiple stimuli and NaNs" begin
    S = rand(100, 3)
    ktrue = OffsetArray(randn(5, 3), -4:0, 1:3)
    r = compute_r(S, ktrue)
    C, d = compute_Cd(S, r, 5)
    kcalc = reshape(C \ d, Compat.axes(ktrue))
    @test kcalc ≈ ktrue

    r[5:10] = r[50:52] = NaN
    C, d = compute_Cd(S, r, 5)
    kcalc = reshape(C \ d, Compat.axes(ktrue))
    @test kcalc ≈ ktrue
end

@testset "Multiple acausal stimuli and NaNs" begin
    S = rand(100,3)
    ktrue = OffsetArray(randn(5,3), -2:2, 1:3)
    r = compute_r(S, ktrue)
    C, d = compute_Cd(S, r, 5)
    kcalc = reshape(C \ d, Compat.axes(ktrue))
    @test kcalc ≈ ktrue

    r[5:10] = r[50:52] = NaN
    C, d = compute_Cd(S, r, 5)
    kcalc = reshape(C \ d, Compat.axes(ktrue))
    @test kcalc ≈ ktrue
end

@testset "Single stimulus with constraints" begin
    S = rand(100)
    ktrue = OffsetArray([rand(3); 0; 0], -2:2)
    r = compute_r(S, ktrue)
    C, d = compute_Cd(S, r, ktrue.!=0)
    kcalc = reshape(C \ d, Compat.axes(ktrue))
    @test kcalc ≈ ktrue
end

@testset "Multiple stimuli with constraints" begin
    S = rand(100,3)
    ktrue = OffsetArray(randn(5,3), -2:2, 1:3)
    ktrue[rand(1:15,5,1)] = 0
    r = compute_r(S,ktrue)
    C, d = compute_Cd(S, r, ktrue.!=0)
    kcalc = reshape(C \ d, Compat.axes(ktrue))
    @test kcalc ≈ ktrue
end

@testset "Multiple stimuli with constraints and Nans" begin
    S = rand(100,7)
    ktrue = OffsetArray(rand(5,7), -2:2, 1:7)
    ktrue[rand(1:length(ktrue),10,1)] = 0
    r = compute_r(S,ktrue)
    r[5:10] = r[50:52] = NaN
    C, d = compute_Cd(S, r, ktrue.!=0)
    kcalc = reshape(C \ d, Compat.axes(ktrue))
    @test kcalc ≈ ktrue
end

