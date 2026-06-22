module Analysis

export save_critical_exponents

function save_critical_exponents(results, path::String)
    open(path, "w") do f
        println(f, "dimension,N,p_c,S_max,chi_max")
        for r in results
            idx = argmax(r.chi)
            pc  = r.p_range[idx]
            Sm  = maximum(r.S)
            cm  = maximum(r.chi)
            println(f, "$(r.dimension),$(r.N),$pc,$Sm,$cm")
        end
    end
end

end
