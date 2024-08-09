import numpy as np

def sigm(x):
    return (1/(1+np.exp(-x)))

def layer(X,w,b):
    f_x = X.T.dot(w) +b
    return sigm(f_x)



