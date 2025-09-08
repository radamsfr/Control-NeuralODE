import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

n = 3
Tf = 1.410
dt = 0.01
T = int(Tf / dt)
t = np.linspace(0, Tf, T)

# dynamics
A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
b = np.array([[0, 0, 1]]).T

# print("A:\n", A)
# print("b:\n", b)

# start state, goal state
x0 = [2.0, -2.0, -1.0]
# x0 = [2.0, 1.0, -2.0] 
xdes = [0.0, 0.0, 0.0]

# constraints
x_max = [0, 3, 3]
u_max = 10
alpha = 0.001

X = cp.Variable(shape=(n, T + 1))
u = cp.Variable(shape=(1, T))
objective = 0
control_penalty = 0

constraints = [
    # start and goal pos constraint
    X[:, 0] == x0,
    X[:, -1] == xdes,
]

for i in range(T):
    constraints.append(
        # dynamics constraint
        X[:, i + 1] == X[:, i] + (A @ X[:, i] + (b * u[0, i]).flatten('F')) * dt
    )
    # state constraints
    constraints.append(cp.abs(u[0, i].flatten('F')) <= u_max)
    # constraints.append(cp.abs(X[1, i]) <= x_max[1])
    # constraints.append(cp.abs(X[2, i]) <= x_max[2])

    objective += (X[0, i] - xdes[0])**2  + 0.001*(X[1, i] - xdes[1])**2 + 0.001*(X[2,i] - xdes[2])**2
    # objective += (X[0, i] - xdes[0])**2  + (X[1, i] - xdes[1])**2 + (X[2,i] - xdes[2])**2
    # objective += (X[0, i] - xdes[0])**2
    # control_penalty += 0.00001 * (u[0,i])**2
    # control_penalty += (u[0,i])**2

objective = cp.Minimize(objective + control_penalty)

import time
start_time = time.time()
cost = cp.Problem(objective, constraints).solve()
end_time = time.time()


print(f"Achieved a total cost of {cost:.02f}. 🚀")
print(f"Compute time: {end_time - start_time:.02f}")

def plot_solution(X: cp.Variable, u: cp.Variable):

    plt.plot(t, X[0, :-1].T.value, label="posisiton", color="navy")
    plt.plot(t, X[1, :-1].T.value, label="velocity", color="orange")
    plt.plot(t, X[2, :-1].T.value, label="acceleration", color="green")
    plt.plot(t, u.T.value, label="jerk", color="red")
    
    # plt.xlim(0,5)
    # plt.ylim(-10, 10)

    plt.title("CVXpy Optimal Trajectory")
    plt.xlabel("time (s)")
    plt.ylabel("state")
    plt.grid(True)
    plt.legend()
    plt.show()
    # return plt.gca()


plot_solution(X, u)
