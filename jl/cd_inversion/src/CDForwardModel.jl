module CDForwardModel

export CDParams, REFERENCE_SPECTRA, predict_spectrum, add_noise

# Wavelength range: 190-250 nm at 1 nm steps
const WAVELENGTHS = collect(190.0:1.0:250.0)   # 61 points

Base.@kwdef struct CDParams
    wavelengths::Vector{Float64} = WAVELENGTHS
    noise_level::Float64 = 0.01
end

# Reference CD basis spectra from literature (millidegrees per nm)
# Based on classical CD spectroscopy for secondary structure
const REFERENCE_SPECTRA = let
    λ = WAVELENGTHS
    n = length(λ)

    # α-helix: strong negative at 208 and 222 nm, positive at 193 nm
    helix = zeros(n)
    for (i, wl) in enumerate(λ)
        helix[i]  = -30.0 * exp(-0.5*((wl-208)/5)^2)
        helix[i] += -25.0 * exp(-0.5*((wl-222)/4)^2)
        helix[i] +=  50.0 * exp(-0.5*((wl-193)/8)^2)
    end

    # β-sheet: negative around 216 nm, positive around 195 nm
    sheet = zeros(n)
    for (i, wl) in enumerate(λ)
        sheet[i]  = -15.0 * exp(-0.5*((wl-216)/7)^2)
        sheet[i] +=  20.0 * exp(-0.5*((wl-195)/6)^2)
    end

    # Random coil: weak negative around 200 nm, positive around 218 nm
    coil = zeros(n)
    for (i, wl) in enumerate(λ)
        coil[i]  = -8.0 * exp(-0.5*((wl-200)/6)^2)
        coil[i] +=  5.0 * exp(-0.5*((wl-218)/8)^2)
    end

    Dict("helix" => helix, "sheet" => sheet, "coil" => coil)
end

# Linear mixture model: CD = f_h * S_h + f_s * S_s + f_c * S_c
function predict_spectrum(helix_frac::Float64, sheet_frac::Float64, coil_frac::Float64,
                          params::CDParams=CDParams())::Vector{Float64}
    h = clamp(helix_frac, 0.0, 1.0)
    s = clamp(sheet_frac, 0.0, 1.0)
    c = clamp(coil_frac,  0.0, 1.0)
    # Normalise so fractions sum to 1
    tot = h + s + c
    tot < 1e-6 && (h = 1.0/3; s = 1.0/3; c = 1.0/3; tot = 1.0)
    h /= tot; s /= tot; c /= tot
    h .* REFERENCE_SPECTRA["helix"] .+ s .* REFERENCE_SPECTRA["sheet"] .+ c .* REFERENCE_SPECTRA["coil"]
end

function add_noise(spectrum::Vector{Float64}; σ::Float64=0.5, seed::Int=0)::Vector{Float64}
    rng = Random.MersenneTwister(seed)
    spectrum .+ σ .* randn(rng, length(spectrum))
end

end
