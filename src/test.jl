
function test(dirpath, u0=[2.0, -2.0, -1.0])
    datadir = dirpath

    @time begin
        system = TripleIntegrator()
        controlODE = ControlODE(system, u0)
        
        θ = deserialize(datadir)
        sol_time, states_raw, controls = run_simulation(controlODE, θ)
    end

    control_input_graph(sol_time, states_raw, controls[:], "TEST Optimized policy with constraints $u0")
    # control_input_graph(sol_time, states_raw, controls[:], "Optimized policy with constraints (test)", false)
end