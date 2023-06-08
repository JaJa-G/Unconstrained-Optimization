import numpy as np
from scipy import optimize
from unconstrained_min import *
import matplotlib.pyplot as plt
from examples import f1,f2,f3,fRosenbrock,f5,f6

#Student ID: 806761#

def plot_all(f,x,a,b,l, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    paths={}
    flag = {}
    x1,y1,flag["gradient_descent"], paths["gradient_descent"] = gradient_descent(f,x,obj_tol,param_tol,max_iter)
    
    x2,y2,flag["newton"], paths["newton"] = newton(f,x,obj_tol,param_tol,max_iter)
    
    x3,y3,flag["BFGS"], paths["BFGS"] = BFGS(f,x,obj_tol,param_tol,max_iter)
  
    x4,y4,flag["SR1"], paths["SR1"] = SR1(f,x,obj_tol,param_tol,max_iter)

    
    # Define the contour grid
    x1 = np.linspace(-a, a, 100)
    x2 = np.linspace(-b, b, 100)
    X1, X2 = np.meshgrid(x1, x2)

    Z = np.zeros_like(X1)
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[i, j] = f([X1[i, j], X2[i, j]])

    fig, axes = plt.subplots(4, 1, figsize=(10, 20))
    for i, method in enumerate(paths):
        ax = axes[i]

        # Plot the contour lines
        ax.contour(X1, X2, Z, levels=l)

        # Plot the iteration paths
        path = paths[method]
        x_vals, y_vals = zip(*path)
        ax.plot(x_vals, y_vals, marker='o', linestyle='-', color='red')
        ax.plot(x_vals[-1], y_vals[-1], marker='o', linestyle='-', color='black')
        ax.annotate(f'({x_vals[-1]}, {y_vals[-1]})', xy=(x_vals[-1], y_vals[-1]), xytext=(x_vals[-1] + 0.1, y_vals[-1] - 0.5))

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_title(method+', iteration times:'+str(len(path)-1)+', Flag:'+str(flag[method]))
        ax.grid(True)

    plt.show()

x =np.array([1,1]).T
x4 =np.array([-1,2]).T

plot_all(f1,x,3,3,30)
plot_all(f2,x,1.5,1.5,30)
plot_all(f3,x,1.5,1.5,100)
plot_all(fRosenbrock,x4,a=70,b=10000,l=100, max_iter=10000)
plot_all(f5,x,a=700,b=700,l=30)
plot_all(f6,x,a=1.5,b=1.5,l=100)