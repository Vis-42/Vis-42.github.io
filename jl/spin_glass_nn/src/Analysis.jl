module Analysis

export save_summary

function save_summary(losses, iprs, path::String; snapshot_every::Int = 50)
    open(path, "w") do f
        println(f, "epoch,loss,ipr")
        for (i, (l, ip)) in enumerate(zip(losses, iprs))
            println(f, "$(i*snapshot_every),$l,$ip")
        end
    end
end

end
