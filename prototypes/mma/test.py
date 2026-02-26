from src.solvers import MMASolver, CCSAMMASolver
import numpy as np

if __name__ == "__main__":
    n_vars = 3
    n_constraints = 2

    # Bounds (e.g., minimum and maximum element thicknesses)
    xmin = np.full(n_vars, 0.1)
    xmax = np.full(n_vars, 10.0)

    # Initialize the Dual MMA solver
    mma = MMASolver(n_vars, xmin, xmax)

    # Initial guess (start thick)
    x = np.array([5.0, 5.0, 5.0])

    print(
        f"{'Iter':<5} | {'x1':<8} | {'x2':<8} | {'x3':<8} | {'Objective':<10} | {'f1 (<=0)':<10} | {'f2 (<=0)':<10}"
    )
    print("-" * 75)

    for i in range(30):
        # 1. Evaluate Objective and Constraints
        f0 = x[0] + x[1] + x[2]
        f_i = np.array([8.0 / x[0] + 1.0 / x[1] - 4.0, 1.0 / x[1] + 2.0 / x[2] - 3.0])

        # 2. Evaluate Gradients (Analytical Sensitivities)
        grad_f0 = np.array([1.0, 1.0, 1.0])

        # Shape must be (n_constraints, n_vars)
        grad_f_i = np.array(
            [
                [-8.0 / (x[0] ** 2), -1.0 / (x[1] ** 2), 0.0],
                [0.0, -1.0 / (x[1] ** 2), -2.0 / (x[2] ** 2)],
            ]
        )

        print(
            f"{i:<5d} | {x[0]:<8.4f} | {x[1]:<8.4f} | {x[2]:<8.4f} | {f0:<10.4f} | {f_i[0]:<10.4f} | {f_i[1]:<10.4f}"
        )

        # 3. Take MMA Step
        x_new = mma.step(x, f0, grad_f0, f_i, grad_f_i)

        # Check convergence (L-infinity norm of the step)
        if np.max(np.abs(x_new - x)) < 1e-4:
            x = x_new
            print("-" * 75)
            print(f"Converged beautifully at iteration {i}!")
            break

        x = x_new

    print(f"Final Design: x = [{x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}]")

    n_vars = 3
    n_constraints = 2

    # Bounds (e.g., minimum and maximum element thicknesses)
    xmin = np.full(n_vars, 0.1)
    xmax = np.full(n_vars, 10.0)

    # Initialize the Globally Convergent MMA solver
    # We set a generous move_limit of 1.0 for this analytical problem
    mma = CCSAMMASolver(local_n_vars=n_vars, xmin=xmin, xmax=xmax, move_limit=1.0)

    # Initial guess (start thick)
    x = np.array([5.0, 5.0, 5.0])

    # 1. Define the "Forward Physics" function required for the inner loop checks
    def evaluate_physics(x_current):
        f0 = x_current[0] + x_current[1] + x_current[2]
        f_i = np.array(
            [
                8.0 / x_current[0] + 1.0 / x_current[1] - 4.0,
                1.0 / x_current[1] + 2.0 / x_current[2] - 3.0,
            ]
        )
        print(f"Evaluate func: {f0}, {f_i}")
        return f0, f_i

    # 2. Define the Adjoint/Gradient function
    def get_gradients(x_current):
        grad_f0 = np.array([1.0, 1.0, 1.0])
        grad_f_i = np.array(
            [
                [-8.0 / (x_current[0] ** 2), -1.0 / (x_current[1] ** 2), 0.0],
                [0.0, -1.0 / (x_current[1] ** 2), -2.0 / (x_current[2] ** 2)],
            ]
        )
        return grad_f0, grad_f_i

    print(
        f"{'Iter':<5} | {'x1':<8} | {'x2':<8} | {'x3':<8} | {'Objective':<10} | {'f1 (<=0)':<10} | {'f2 (<=0)':<10}"
    )
    print("-" * 75)

    for i in range(30):
        # Evaluate Current State
        f0, f_i = evaluate_physics(x)
        grad_f0, grad_f_i = get_gradients(x)

        print(
            f"{i:<5d} | {x[0]:<8.4f} | {x[1]:<8.4f} | {x[2]:<8.4f} | {f0:<10.4f} | {f_i[0]:<10.4f} | {f_i[1]:<10.4f}"
        )

        # Take GCMMA Step (Pass evaluate_physics as the eval_func callback)
        x_new = mma.step(
            x_local=x,
            f0_global=f0,
            grad_f0_local=grad_f0,
            f_i_global=f_i,
            grad_f_i_local=grad_f_i,
            eval_func=evaluate_physics,
        )

        # Check convergence (L-infinity norm of the step)
        if np.max(np.abs(x_new - x)) < 1e-4:
            x = x_new
            print("-" * 75)
            print(f"Converged beautifully at iteration {i}!")
            break

        x = x_new

    # Evaluate final state
    f0_final, f_i_final = evaluate_physics(x)
    print("-" * 75)
    print(f"Final Design: x = [{x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}]")
    print(f"Final Objective: {f0_final:.4f}")
    print(f"Final Constraints: f1 = {f_i_final[0]:.4f}, f2 = {f_i_final[1]:.4f}")
