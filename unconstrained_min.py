import numpy as np
from scipy import optimize
#Student ID: 806761#
def get_gradient(f,x):

    '''   
    eps=1e-12
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_eps = x.copy()
        x_eps[i] += eps
        grad[i] = (f(x_eps) - f(x)) / eps
        return grad
    ''' 
    return optimize.approx_fprime(x, f, epsilon=1e-12)

def wolfe_condition(f, grad, x, alpha=1.0, c1=0.5, beta=0.8):
    while f(x + alpha*grad) > f(x)+c1*alpha*np.dot((-grad).T, grad):
        alpha *= beta
    return alpha

def get_hessian(f, x):
    n = len(x)
    hessian = np.zeros((n, n))
    eps = 1e-6

    for i in range(n):
        for j in range(n):
            f_xx = (f(x + eps*np.eye(n)[i] + eps*np.eye(n)[j]) - f(x + eps*np.eye(n)[i]) -
                    f(x + eps*np.eye(n)[j]) + f(x)) / (eps**2)
            hessian[i, j] = f_xx

    return hessian

def gradient_descent(f,x,obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    path=[]
    #record start point of path
    path.append(x)
    flag = False
    for i in range(max_iter):
        grad = -get_gradient(f,x)
        
#        if (np.isclose(np.linalg.norm(grad),np.zeros_like(x)).all()):
#            flag = True
#            return x,f(x),flag, path
        
        step = wolfe_condition(f,grad,x)
        new_x = x + step*grad
        
        if (np.linalg.norm(step*grad) < param_tol) or (np.abs(f(new_x)-f(x)) < obj_tol):
            flag = True
            return x,f(x),flag, path
        
        path.append(new_x)
        print('For gradient descent, the iteration is', i+1, 'the current location is', new_x, 'and the current objective value is', f(new_x))
        x = new_x
    return x,f(x),flag, path

def newton(f,x, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    path=[]
    #record start point of path
    path.append(x)
    flag = False
    for i in range(max_iter):
        grad = -get_gradient(f,x)
        hessian = get_hessian(f, x)
        if np.linalg.det(hessian) == 0:
            return x,f(x),flag, path
        
        direction = np.linalg.solve(hessian, grad)
        step = wolfe_condition(f,direction,x)
        new_x = x + step*direction

        if (np.linalg.norm(step*direction) < param_tol) or (np.abs(f(new_x)-f(x)) < obj_tol):
            flag = True
            return x,f(x),flag, path

        path.append(new_x)
        print('For newton method, the iteration is', i+1, 'the current location is', new_x, 'and the current objective value is', f(new_x))
        x = new_x
    return x,f(x),flag, path

def BFGS(f,x, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    path=[]
    #record start point of path
    path.append(x)
    flag = False
    #initial B
    B=np.eye(len(x))
    for i in range(max_iter):
        grad = get_gradient(f,x)

        #B_kd=-gk
        direction = -np.linalg.solve(B, grad)
        step = wolfe_condition(f,direction,x)
        p = step*direction
        new_x = x + p

        if (np.linalg.norm(step*direction) < param_tol) or (np.abs(f(new_x)-f(x)) < obj_tol):
            flag = True
            return x,f(x),flag, path

        path.append(new_x)
        print('For BFGS, the iteration is', i+1, 'the current location is', new_x, 'and the current objective value is', f(new_x))

        #update B
        q = get_gradient(f,new_x)-get_gradient(f,x)
        dB = (np.outer(q, q) / np.dot(q, p)) - (np.outer(B @ p, B @ p) / np.dot(p, B @ p))

        #dB = np.dot(q,q.T)/np.dot(q.T,p)-np.dot(np.dot(np.dot(B,p),p.T),B)/np.dot(np.dot(p.T,B),p)
        B = B + dB
        x = new_x

    return x,f(x),flag, path

def SR1(f,x, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    path=[]
    #record start point of path
    path.append(x)
    flag = False
    #initial B
    B=np.eye(len(x))
    for i in range(max_iter):
        grad = get_gradient(f,x)

        direction = -np.dot(B,grad)
        step = wolfe_condition(f,direction,x)
        p = step*direction
        new_x = x + p
        if (np.linalg.norm(p) < param_tol) or (np.abs(f(new_x)-f(x)) < obj_tol):
            flag = True
            return x,f(x),flag, path

        path.append(new_x)
        print('For SR1, the iteration is', i+1, 'the current location is', new_x, 'and the current objective value is', f(new_x))

        #update B
        q = get_gradient(f,new_x)-get_gradient(f,x)

        dB = np.outer(p - np.dot(B, q), p - np.dot(B, q)) / np.dot(p - np.dot(B, q), q)
        #dB = np.dot((p-np.dot(B,q)),(p-np.dot(B,q)).T)/np.dot((p-np.dot(B,q)).T,q)
        B = B + dB
        x = new_x

    return x,f(x),flag, path