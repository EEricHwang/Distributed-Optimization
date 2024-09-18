
# This is an example code for the ADMM algorithm

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

def Min_x(z_k,y_k,Qx,Qz,A,B,c,rho):
    x = cp.Variable(Qx.shape[0])
    z = z_k
    y = y_k
    Lp_xz_y = cp.quad_form(x,Qx) + cp.quad_form(z,Qz) + y.T @ (A @ x + B @ z - c) + (rho/2)*cp.sum_squares(A @ x + B @ z - c)
    prob = cp.Problem(cp.Minimize(Lp_xz_y))
    prob.solve()
    x_next = x.value
    
    return x_next

def Min_z(x_kp1,y_k,Qx,Qz,A,B,c,rho):
    z = cp.Variable(Qz.shape[0])
    x = x_kp1
    y = y_k
    Lp_xz_y = cp.quad_form(x,Qx) + cp.quad_form(z,Qz) + y.T @ (A @ x + B @ z - c) + (rho/2)*cp.sum_squares(A @ x + B @ z - c)
    prob = cp.Problem(cp.Minimize(Lp_xz_y))
    prob.solve()
    z_next = z.value
    
    return z_next


## Initial setup

x_real = np.arange(1,11)
z_real = np.arange(1,21)

x_dim  = len(x_real)
z_dim  = len(z_real) 

np.random.seed(123)

# Generate random matrices Qx and Qz for cost functions (Quadratic Form)

Qx = np.random.randint(-15, 16, (x_dim, x_dim))
Qx = Qx.T @ Qx   

Qz = np.random.randint(-15, 16, (z_dim, z_dim))
Qz = Qz.T @ Qz   

# Generate random matrices A and B

A = np.random.randint(-15, 16, (5, x_dim))
B = np.random.randint(-15, 16, (5, z_dim))

# Coupling constraint

C = A @ x_real + B @ z_real

y_dim = len(C)
x_val = np.ones(x_dim)
z_val = np.ones(z_dim)
y_val = np.ones(y_dim)

z_k = z_val
y_k = y_val

N = 200

x_cost_save = []
z_cost_save = []
constraint_cost_save = []

## MAIN ADMM LOOP

for i in range(N):
    
    if i == 0:
        rho = 1
    else:
        rho = 1/i

    x_next = Min_x(z_k,y_k,Qx,Qz,A,B,C,rho)
    z_next = Min_z(x_next,y_k,Qx,Qz,A,B,C,rho)
    y_next = y_k + rho*(np.dot(A,x_next) + np.dot(B,z_next) - C)

    z_k = z_next
    y_k = y_next

    x_cost_save.append(np.matmul(np.dot(x_next,Qx),x_next))
    z_cost_save.append(np.matmul(np.dot(z_next,Qz),z_next))
    constraint_cost_save.append(np.linalg.norm((np.dot(A,x_next) + np.dot(B,z_next) - C)))

## Plot for ADMM Simulation

plt.plot(range(N),x_cost_save,linewidth=2.5,label='$f(x)$')
plt.plot(range(N),z_cost_save,linewidth=2.5,label='$g(z)$')
plt.plot(range(N),constraint_cost_save,linewidth=2.5,label='$Ax+Bz-C$')
plt.xlabel('Iteration')
plt.ylabel('Cost Function Values')
plt.title('ADMM Simulation Example')
plt.legend()
plt.grid()
plt.show()
