function safe_deserialize_params(path)
    raw = deserialize(path)
    return map(x ->
        x == "Inf"  ? Inf :
        x == "-Inf" ? -Inf :
        x == "NaN"  ? NaN :
        x, raw)
end

function test(dirpath, u0=[2.0, -2.0, -1.0])
    datadir = dirpath

    @time begin
        system = TripleIntegrator()
        controlODE = ControlODE(system, u0)
        
        θ = safe_deserialize_params(datadir)
        # θ = replace(safe_θ, "Inf"=>Inf, "-Inf"=>-Inf, "NaN"=>NaN)
        sol_time, states_raw, controls = run_simulation(controlODE, θ)
    end

    control_input_graph(sol_time, states_raw, controls[:], "TEST Optimized policy with constraints $u0")
    # control_input_graph(sol_time, states_raw, controls[:], "Optimized policy with constraints (test)", false)
end