import numpy as np
import matplotlib.pyplot as plt

def _two_dim_traj(traj, variables):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot()
    ax.plot(traj[:, 0], traj[:, 1])
    ax.set_xlabel('$' + str(variables[0]) + '$')
    ax.set_ylabel('$' + str(variables[1]) + '$')
    plt.show()

def _three_dim_traj(traj, variables):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
    ax.set_xlabel('$' + str(variables[0]) + '$')
    ax.set_ylabel('$' + str(variables[1]) + '$')
    ax.set_zlabel('$' + str(variables[1]) + '$')
    plt.show()

def plot_exponents(exp, y_lims=False, show=True):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot()
    exp_num = exp.shape[1]
    
    for i in range(exp_num):
        ax.plot(exp[:, i], label="Exponent #" + str(i+1))\
    
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Lyapunov Exponent")

    if y_lims:
        plt.ylim(np.min(exp[-1, :]-1), np.max(exp[-1, :]+1))

    if show:
        plt.show()
    else:
        return plt


def plot_both_exponents(exp_fwd, exp_bkw, y_lims=False, show=True):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot()
    exp_num = exp_fwd.shape[1]
    max_time = min(exp_fwd.shape[0], exp_bkw.shape[0])
    
    for i in range(exp_num):
        ax.plot(exp_fwd[:max_time, i], label="Forward Exp #" + str(i+1))

    for i in range(exp_num):
        ax.plot(exp_bkw[:max_time, i], label="Backward Exp #" + str(i+1))
    
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Lyapunov Exponent")

    if y_lims:
        plt.ylim(np.min(exp_fwd[-1, :]-1), np.max(exp_fwd[-1, :]+1))

    if show:
        plt.show()
    else:
        return plt