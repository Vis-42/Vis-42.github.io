module Analysis

export save_summary

function save_summary(results, noise_levels, path::String)
    open(path, "w") do f
        write(f, "Differentiable Pendulum Parameter Inference — Summary\n")
        write(f, "="^60 * "\n\n")
        write(f, "noise_σ,m1_err%,m2_err%,L1_err%,L2_err%,final_loss\n")
        for (σ, res) in zip(noise_levels, results)
            e = res.param_error_pct
            write(f, "$(σ),$(round(e[1];digits=2)),$(round(e[2];digits=2)),$(round(e[3];digits=2)),$(round(e[4];digits=2)),$(round(res.final_loss;sigdigits=4))\n")
        end
    end
end

end
