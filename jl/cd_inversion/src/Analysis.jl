module Analysis

using Statistics

export save_summary

function save_summary(results_with, results_without, path::String)
    open(path, "w") do f
        write(f, "CD Inversion Network — Summary\n")
        write(f, "="^60 * "\n\n")
        write(f, "condition,test_rmse,physics_satisfaction,spectral_r2\n")

        for (label, res) in [("with_physics", results_with), ("without_physics", results_without)]
            rmse = res[:test_rmse]
            phys = res[:physics_sat]
            r2   = res[:spectral_r2]
            write(f, "$(label),$(round(rmse;sigdigits=4)),$(round(phys;sigdigits=4)),$(round(r2;sigdigits=4))\n")
        end
    end
end

end
