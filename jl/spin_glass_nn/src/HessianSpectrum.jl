module HessianSpectrum

using LinearAlgebra, Statistics

export eigenvalue_spectrum, ipr, participation_ratio

"""
    eigenvalue_spectrum(diag_H)

Given the diagonal Hessian approximation (parameter-axis curvatures Hₖₖ),
return them sorted. These are not true eigenvalues — for the full spectrum
use LinearAlgebra.eigen on the full Hessian.
"""
function eigenvalue_spectrum(diag_H::AbstractVector)
    sort(diag_H)
end

"""
    ipr(eigenvalues)

Inverse Participation Ratio: IPR = Σ(|Hₖₖ| / Σ|Hⱼⱼ|)^2
Range: 1/N (all diagonal entries equal weight, delocalised) to 1 (single dominant axis).
High IPR means the curvature budget is concentrated on few parameter axes:
low IPR ~ broad/uniform curvature (glassy); high IPR ~ few sharp axes (crystallised/ordered).
"""
function ipr(eigenvalues::AbstractVector)
    vals = abs.(eigenvalues)
    s = sum(vals)
    s ≈ 0.0 && return 1.0
    sum((v/s)^2 for v in vals)
end

"""
    participation_ratio(eigenvalues)

PR = 1/IPR — number of "effectively participating" eigenvalues.
"""
function participation_ratio(eigenvalues::AbstractVector)
    i = ipr(eigenvalues)
    i ≈ 0.0 && return Float64(length(eigenvalues))
    1.0 / i
end

end
