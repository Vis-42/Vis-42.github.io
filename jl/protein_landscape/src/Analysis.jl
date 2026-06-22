module Analysis

using ..FreeEnergy, ..MonteCarlo
using Statistics

export save_summary

function save_summary(results, T_ladder, path::String)
    open(path, "w") do f
        write(f, "Protein Landscape — Replica Exchange MC Summary\n")
        write(f, "="^60 * "\n\n")
        write(f, "T,mean_Q,std_Q,mean_E,std_E,n_samples\n")
        for r in results
            mQ = isempty(r.Q_traj) ? 0.0 : mean(r.Q_traj)
            sQ = isempty(r.Q_traj) ? 0.0 : std(r.Q_traj)
            mE = isempty(r.E_traj) ? 0.0 : mean(r.E_traj)
            sE = isempty(r.E_traj) ? 0.0 : std(r.E_traj)
            write(f, "$(r.T),$(round(mQ;digits=4)),$(round(sQ;digits=4)),$(round(mE;digits=4)),$(round(sE;digits=4)),$(length(r.Q_traj))\n")
        end
        write(f, "\n--- Free Energy Profile ---\n")
        write(f, "Using lowest-temperature replica (T=$(minimum(r.T for r in results)))\n")
    end
end

end
