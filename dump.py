
def optimize_max_auto(T, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H,
                 lambda_L, lambda_H, lambda_param_L, lambda_param_H, eta_max, num_steps_max, epsilon, delta, seed):
    
    torch.manual_seed(seed)

    # Ensure parameters require gradients
    mu_L = mu_L.clone().detach().requires_grad_(True)
    mu_H = mu_H.clone().detach().requires_grad_(True)
    Sigma_L = Sigma_L.clone().detach().requires_grad_(True)
    Sigma_H = Sigma_H.clone().detach().requires_grad_(True)

    # Define optimizers for each parameter
    optimizer_mu = torch.optim.Adam([mu_L, mu_H], lr=eta_max)
    optimizer_Sigma = torch.optim.Adam([Sigma_L, Sigma_H], lr=eta_max)

    theta_objectives_epoch = []
    for step in range(num_steps_max):
        # Zero gradients
        optimizer_mu.zero_grad()
        optimizer_Sigma.zero_grad()

        # Compute the maximization objective
        obj_theta = torch.tensor(0.0)
        for n, iota in enumerate(Ill):
            L_i = torch.from_numpy(LLmodels[iota].F).float()
            H_i = torch.from_numpy(HLmodels[omega[iota]].F).float()

            # Objective calculation (update this based on your original closed-form objective)
            obj_value_iota = oput.compute_objective_value(T, L_i, H_i, mu_L, mu_H, Sigma_L, Sigma_H,
                                                          lambda_L, lambda_H, hat_mu_L, hat_mu_H, hat_Sigma_L, hat_Sigma_H,
                                                          epsilon, delta)
            obj_theta += obj_value_iota

        # Average objective
        obj_theta = obj_theta / (n + 1)

        # Backpropagate
        (-obj_theta).backward()  # Negate for maximization

        # Update parameters
        optimizer_mu.step()
        optimizer_Sigma.step()

        # Projection onto Gelbrich balls
        mu_L.data, Sigma_L.data = oput.project_onto_gelbrich_ball(mu_L.data, Sigma_L.data, hat_mu_L, hat_Sigma_L, epsilon)
        mu_H.data, Sigma_H.data = oput.project_onto_gelbrich_ball(mu_H.data, Sigma_H.data, hat_mu_H, hat_Sigma_H, delta)

        # Verify constraints
        satisfied_L, dist_L, epsi = oput.verify_gelbrich_constraint(mu_L.data, Sigma_L.data, hat_mu_L, hat_Sigma_L, epsilon)
        satisfied_H, dist_H, delt = oput.verify_gelbrich_constraint(mu_H.data, Sigma_H.data, hat_mu_H, hat_Sigma_H, delta)
        oput.constraints_error_check(satisfied_L, dist_L, epsi, satisfied_H, dist_H, delt)

        # Log the objective value for monitoring
        theta_objectives_epoch.append(obj_theta)#.item())
        #print(f"Max step: {step+1}, Objective: {obj_theta.item()}")

    #return mu_L.detach(), Sigma_L.detach(), mu_H.detach(), Sigma_H.detach(), obj_theta, theta_objectives_epoch
    return mu_L, Sigma_L, mu_H, Sigma_H, obj_theta, theta_objectives_epoch