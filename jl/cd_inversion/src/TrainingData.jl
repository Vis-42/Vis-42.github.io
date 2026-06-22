module TrainingData

using ..CDForwardModel
using Random

export generate_cd_dataset

function generate_cd_dataset(N::Int; noise_σ::Float64=0.5, seed::Int=42)
    rng = MersenneTwister(seed)
    params = CDParams()
    wavelengths = params.wavelengths

    spectra      = zeros(N, length(wavelengths))
    compositions = zeros(N, 3)   # [helix, sheet, coil]

    for i in 1:N
        # Sample random composition on the 2-simplex
        h, s, c = _sample_simplex(rng)
        compositions[i, :] = [h, s, c]
        spec_clean = predict_spectrum(h, s, c, params)
        spectra[i, :] = spec_clean .+ noise_σ .* randn(rng, length(wavelengths))
    end

    wavelengths, spectra, compositions
end

function _sample_simplex(rng::AbstractRNG)
    # Sample uniformly from the 2-simplex
    x1, x2 = sort([rand(rng), rand(rng)])
    h = x1; s = x2 - x1; c = 1.0 - x2
    h, s, c
end

end
