module GoModel

using LinearAlgebra, Random

export ProteinConfig, TrpCageConfig, go_energy, contact_energy, radius_of_gyration, fraction_native_contacts

struct ProteinConfig
    n_residues::Int
    native_coords::Matrix{Float64}   # n_residues × 3
    native_contacts::Vector{Tuple{Int,Int}}  # pairs in contact
    epsilon::Float64    # Gō well depth
    sigma::Float64      # bead radius
    k_bond::Float64     # bond stiffness
    r0::Float64         # equilibrium bond length
    contact_cutoff::Float64
end

# Build a toy Trp-cage model: 25-bead linear chain, native contacts from distance
function TrpCageConfig(; epsilon::Float64=1.0, sigma::Float64=0.4, seed::Int=42)
    n = 25
    rng = MersenneTwister(seed)
    # Generate native structure as a compact helix-like shape
    coords = zeros(n, 3)
    for i in 1:n
        angle = 2π * i / 3.6           # α-helix: 3.6 residues/turn
        coords[i, 1] = 0.23 * cos(angle)  # helix radius
        coords[i, 2] = 0.23 * sin(angle)
        coords[i, 3] = 0.15 * i           # rise per residue
    end
    # Add a small globular tail
    for i in 15:n
        coords[i, 1] += 0.1 * sin(Float64(i))
        coords[i, 2] += 0.1 * cos(Float64(i))
        coords[i, 3] *= 0.7
    end
    # Native contacts: residue pairs within cutoff distance (not adjacent in sequence)
    contact_cutoff = 0.8
    contacts = Tuple{Int,Int}[]
    for i in 1:n, j in (i+3):n
        r = norm(coords[i,:] .- coords[j,:])
        r < contact_cutoff && push!(contacts, (i,j))
    end
    ProteinConfig(n, coords, contacts, epsilon, sigma, 100.0, 0.38, contact_cutoff)
end

# Total Gō model energy
function go_energy(coords::Matrix{Float64}, cfg::ProteinConfig)::Float64
    E_bond = bond_energy(coords, cfg)
    E_contact = contact_energy(coords, cfg)
    E_rep = repulsion_energy(coords, cfg)
    E_bond + E_contact + E_rep
end

# Harmonic bond energy along backbone
function bond_energy(coords::Matrix{Float64}, cfg::ProteinConfig)::Float64
    E = 0.0
    for i in 1:(cfg.n_residues-1)
        r = norm(coords[i+1,:] .- coords[i,:])
        E += 0.5 * cfg.k_bond * (r - cfg.r0)^2
    end
    E
end

# Lennard-Jones 12-10 Gō contact potential for native contacts
function contact_energy(coords::Matrix{Float64}, cfg::ProteinConfig)::Float64
    E = 0.0
    for (i,j) in cfg.native_contacts
        r_ij = norm(coords[i,:] .- coords[j,:])
        r_nat = norm(cfg.native_coords[i,:] .- cfg.native_coords[j,:])
        r_nat = max(r_nat, 0.2)
        x = r_nat / r_ij
        E += cfg.epsilon * (5.0 * x^12 - 6.0 * x^10)
    end
    E
end

# Soft repulsion for non-native pairs (excluded volume)
function repulsion_energy(coords::Matrix{Float64}, cfg::ProteinConfig)::Float64
    E = 0.0
    native_set = Set(cfg.native_contacts)
    n = cfg.n_residues
    for i in 1:n, j in (i+3):n
        (i,j) in native_set && continue
        r = norm(coords[i,:] .- coords[j,:])
        r = max(r, 0.1)
        E += cfg.epsilon * (cfg.sigma/r)^12
    end
    E
end

function radius_of_gyration(coords::Matrix{Float64})::Float64
    com = vec(mean(coords; dims=1))
    sqrt(mean(sum((coords .- com').^2; dims=2)))
end

function fraction_native_contacts(coords::Matrix{Float64}, cfg::ProteinConfig)::Float64
    n_formed = 0
    for (i,j) in cfg.native_contacts
        r_ij  = norm(coords[i,:] .- coords[j,:])
        r_nat = norm(cfg.native_coords[i,:] .- cfg.native_coords[j,:])
        r_nat < 0.2 && continue
        r_ij < 1.2 * r_nat && (n_formed += 1)
    end
    isempty(cfg.native_contacts) ? 0.0 : n_formed / length(cfg.native_contacts)
end

end
