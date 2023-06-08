from unconstrained_min import get_gradient
from unconstrained_min import get_hessian
import numpy as np
#Student ID: 806761#.

def q1(x, y):
    return x ** 2 + y** 2

def f1(x):
    q=np.array([[1,0],[0,1]])
    x = np.array(x)
    return np.dot(np.dot((x.T),q),x)

def function1(x,bool_v=False):
    f=f1(x)
    g=get_gradient(f1,x)
    if bool_v==True:
        return f,g,get_hessian(f1,x)
    return f,g

def q2(x, y):
    return x ** 2 + 100*y** 2

def f2(x):
    q=np.array([[1,0],[0,100]])
    x = np.array(x)
    return np.dot(np.dot((x.T),q),x)

def function2(x,bool_v=False):
    f=f2(x)
    g=get_gradient(f2,x)
    if bool_v==True:
        return f,g,get_hessian(f2,x)
    return f,g

def f3(x):
    a=np.array([[np.sqrt(3)/2,-0.5],[0.5,np.sqrt(3)/2]]).T
    b=np.array([[100,0],[0,1]])
    c=np.array([[np.sqrt(3)/2,-0.5],[0.5,np.sqrt(3)/2]])
    q=np.dot(np.dot(a,b),c)
    x = np.array(x)
    return np.dot(np.dot((x.T),q),x)

def function3(x,bool_v=False):
    f=f3(x)
    g=get_gradient(f3,x)
    if bool_v==True:
        return f,g,get_hessian(f3,x)
    return f,g

def fRosenbrock(x):
    x = np.array(x)
    q=np.array([[1200*x[0]**2-400*x[1],-400*x[0]],[-400*x[0],200]])
    return np.dot(np.dot((x.T),q),x)

def qRosenbrock(x1,x2):
    return 100*(x2-x1**2)**2+(1-x1)

def function4(x,bool_v=False):
    f=fRosenbrock(x)
    g=get_gradient(fRosenbrock,x)
    if bool_v==True:
        return f,g,get_hessian(fRosenbrock,x)
    return f,g

def f5(x):
    a=np.array([3,5])
    x = np.array(x)
    return np.dot((a.T),x)

def function5(x,bool_v=False):
    f=f5(x)
    g=get_gradient(f5,x)
    if bool_v==True:
        return f,g,get_hessian(f5,x)
    return f,g

def q6(x1,x2):
    return np.e**(x1+3*x2-0.1)+np.e**(x1-3*x2-0.1)+np.e**(-x1-0.1)

def f6(x):
    return q6(x[0],x[1])

def function6(x,bool_v=False):
    f=f6(x)
    g=get_gradient(f6,x)
    if bool_v==True:
        return f,g,get_hessian(f6,x)
    return f,g