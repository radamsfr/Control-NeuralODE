# Vassiliadis, V. S., Sargent, R. W. H., & Pantelides, C. C. (1994).
# Solution of a Class of Multistage Dynamic Optimization Problems. 2. Problems with Path Constraints.
# Industrial & Engineering Chemistry Research, 33(9), 2123–2133. https://doi.org/10.1021/ie00033a015

function triple_integrator(; store_results::Bool=false)
    datadir = nothing
    if store_results
        datadir = generate_data_subdir(@__FILE__)
    end
    graph_width = 41

    # initial state (original code used u0)
    u0 = [2.0, -2.0, -1.0]
    system = TripleIntegrator()
    controlODE = ControlODE(system, u0)

    θ = initial_params(controlODE.controller)

    # figure 1
    sol_time, states_raw, controls = run_simulation(controlODE, θ)
    # phase_portrait(
    #     controlODE,
    #     θ,
    #     # square_bounds(center, width)
    #     square_bounds(controlODE.u0, graph_width);
    #     projection=[1, 2],
    #     markers=states_markers(states_raw),
    #     title="Initial policy",
    # )

    # control_input_graph(sol_time, states_raw, controls[:], "Initial policy")

    collocation_model = van_der_pol_collocation(
        controlODE.u0,
        controlODE.tspan;
        num_supports=length(controlODE.tsteps),
        nodes_per_element=2,
        constrain_states=true,
    )
    collocation_results = extract_infopt_results(collocation_model)
    reference_controller = interpolant_controller(collocation_results; plot=:unicode)

    θ = preconditioner(
        controlODE,
        reference_controller;
        θ,
        x_tol=1f-7,
        g_tol=1f-2,
    )

    plot_simulation(controlODE, θ; only=:controls)
    store_simulation("precondition", controlODE, θ; datadir)

    # figure 2
    sol_time, states_raw, controls = run_simulation(controlODE, θ)
    # phase_portrait(
    #     controlODE,
    #     θ,
    #     square_bounds(controlODE.u0, graph_width);
    #     projection=[1, 2],
    #     markers=states_markers(states_raw),
    #     title="Preconditioned policy",
    # )

    # control_input_graph(sol_time, states_raw, controls[:],"Preconditioned policy")
    
    ### define objective function to optimize
    function loss(controlODE, params; kwargs...)
        Δt = Float32(controlODE.tsteps.step)
        sol = solve(controlODE, params) |> Array
        # size(sol) [3, T]

        # return Array(sol)[3, end]  # return last value of third variable ...to be minimized
        objective = 0.0f0
        control_penalty = 0.0f0

        for i in axes(sol, 2)
            s = sol[:, i]
            c = controlODE.controller(s, params)

            objective += s[1]^2 + 0.001*s[2]^2 + 0.001*s[3]^2
            control_penalty += c[1] * 0
        end
        objective *= Δt
        # control_penalty *= Δt

        # goal_state = [0, 0, 0]
        # weight = [10, 10, 10]
        # objective += sum(weight .* abs.(goal_state .- sol[:, end]))
        return objective + control_penalty
    end
    loss(params) = loss(controlODE, params)

    @info "Training..."
    grad!(g, params) = g .= Zygote.gradient(loss, params)[1]
    # θ = optimize_optim(θ, loss, grad!)
    θ = optimize_ipopt(θ, loss, grad!)

    # figure 3
    sol_time, states_raw, controls = run_simulation(controlODE, θ)
    phase_portrait(
        controlODE,
        θ,
        square_bounds(controlODE.u0, graph_width);
        projection=[1, 2],
        markers=states_markers(states_raw),
        title="Optimized policy",
    )

    control_input_graph(sol_time, states_raw, controls[:],"Optimized policy")

    store_simulation(
        "unconstrained",
        controlODE,
        θ;
        datadir,
        metadata=Dict(:loss => loss(θ), :constraint => "none"),
    )

    v_constraints = [-3.0, 3.0]
    a_constraints = [-3.0, 3.0]

    ### now add state constraint x1(t) > -0.4 with
    function losses(controlODE, params; α, δ, ρ)
        # integrate ODE system
        Δt = Float32(controlODE.tsteps.step)
        sol = solve(controlODE, params) |> Array
        objective = 0.0f0
        control_penalty = 0.0f0
        for i in axes(sol, 2)
            s = sol[:, i]
            c = controlODE.controller(s, params)

            objective += s[1]^2 + 0.001*s[2]^2 + 0.001*s[3]^2
            control_penalty += c[1]*0
        end
        objective *= Δt
        control_penalty *= Δt

        # goal state
        # goal_state = [0, 0, 0]
        # weight = [10, 10, 10]
        # objective += sum(weight .* abs.(goal_state .- sol[:, end]))

        # state constraints
        velocity_fault = map(v -> relaxed_log_barrier(v, v_constraints[1], v_constraints[2]; δ), sol[2, 1:end])
        acceleration_fault = map(a -> relaxed_log_barrier(a, a_constraints[1], a_constraints[2]; δ), sol[3, 1:end])

        state_penalty = Δt * α * (sum(velocity_fault) + sum(acceleration_fault))
        regularization = ρ * sum(abs2, params)
        return (; objective, state_penalty, control_penalty, regularization)
    end

    @info "Enforcing constraints..."
    ρ = 0f0
    θ, barrier_progression = constrained_training(
        losses,
        controlODE,
        θ;
        ρ,
        show_progressbar=false,
        datadir,
    )

    @info "Alpha progression" barrier_progression.α
    lineplot(log.(barrier_progression.α); title="Alpha progression") |> display

    @info "Delta progression" barrier_progression.δ
    lineplot(log.(barrier_progression.δ); title="Delta progression") |> display

    α_final = barrier_progression.α[end]
    δ_final = barrier_progression.δ[end]

    # penalty_loss(result.minimizer, constrained_prob, tsteps; α=penalty_coefficients[end])
    plot_simulation(controlODE, θ; only=:controls)

    objective, state_penalty, control_penalty, regularization = losses(controlODE, θ; α=α_final, δ=δ_final, ρ)
    store_simulation(
        "constrained",
        controlODE,
        θ;
        datadir,
        metadata=Dict(
            # :loss => penalty_loss(controlODE, θ; α=α0, δ),
            # :constraint => "quadratic x2(t) > -0.4",
            :objective => objective,
            :state_penalty => state_penalty,
            :control_penalty => control_penalty,
            :regularization => regularization,
        ),
    )

    # figure 4
    sol_time, states_opt, controls = run_simulation(controlODE, θ)
    function indicator(coords...)
        if coords[2] > v_constraints[1] && coords[2] < v_constraints[2]
            return true
        end
        return false
    end
    shader = ShadeConf(; indicator)
    phase_portrait(
        controlODE,
        θ,
        square_bounds(controlODE.u0, graph_width);
        shader,
        projection=[1, 2],
        markers=states_markers(states_opt),
        title="Optimized policy with constraints",
    )

    control_input_graph(sol_time, states_raw, controls[:], "Optimized policy with constraints")

    # u0 = [0f0, 1f0]
    # perturbation_specs = [
    #     (variable=1, type=:positive, scale=1.0f0, samples=3, percentage=1.0f-1)
    #     (variable=2, type=:negative, scale=1.0f0, samples=3, percentage=1.0f-1)
    #     # (variable=3, type=:positive, scale=20.0f0, samples=8, percentage=2.0f-2)
    # ]
    # constraint_spec = ConstRef(; val=-0.4, direction=:horizontal, class=:state, var=1)

    # plot_initial_perturbations_collocation(
    #     controlODE,
    #     θ,
    #     perturbation_specs,
    #     van_der_pol_collocation;
    #     refs=[constraint_spec],
    #     storedir=datadir,
    # )
    return
end
