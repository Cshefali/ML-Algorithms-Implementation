##LINEAR REGRESSION WITH MULTIPLE VARIABLES
##Predicting price of house using variables- size (in sqft.), number of floors,
##number of bedrooms and age of house (years). 

import numpy as np
import matplotlib.pyplot as plt
import math as math

#Data
x_train = np.array([[1.5, 1, 2, 5],
                    [1.75, 2, 3, 2],
                    [2.5, 2, 4, 4],
                    [3.1, 3, 6, 7],
                    [5, 2, 5, 3]])

y_train = np.array([80,95,120,450,500])

#function to compute model output for all observations
def compute_model_output(x_train, w, b):
        """
        Job: computes the price of house taking all variables.
        
        Arguments: 
        x_train- (m,n) dimensional array
        w- 1-D vector containing n weights for n variables.
        b- scalar, bias parameter. 
        
        Returns: 1-D array containing model's output price for each observation
        """
        
        m = x_train.shape[0]    #number of rows/observations
        fx = np.zeros(m)        #will contain model's o/p
        for i in range(m):
                fx[i] = np.dot(x_train[i],w) + b
        return fx

#function to compute model cost over all observations
def compute_cost(x_train, y_train, w, b):
        """
        Job: computes model cost over all observations
        
        Arguments: 
        x_train- (m,n) dimensional array; m observations, n features
        y_train- 1-D array containing target price of the houses; m values
        w- 1-D array containing n weights.
        b- scalar value, bias parameter.
        
        Returns: cost of the model
        """
        total_cost = 0.0
        m = x_train.shape[0]
        for i in range(m):
                fx = np.dot(x_train[i],w) + b
                loss = fx - y_train[i]
                total_cost += loss**2   #squared error loss
        total_cost = total_cost/(2*m)   #mean squared error
        return total_cost

#function to compute gradient for gradient descent algorithm
def compute_gradient(x_train, y_train, w, b):
    """
    Job: computes gradient for all parameters w and b
    
    Argument: 
    x_train- (m,n) dimensional array; m observations, n features
    y_train- 1-D array containing m target house prices
    w- 1-D array containing n weight parameters
    b- scalar value, containing bias parameter
    
    Returns:
    dj_dw- 1-D array containing n gradient values for n weights
    dj_db- scalar, gradient for bias parameter.
    """
    m,n = x_train.shape
    dj_dw_i = np.zeros(n)
    dj_db_i = 0.0
    for i in range(m):
        fx = np.dot(x_train[i],w) + b
        err = fx - y_train[i]
        for j in range(n):
            dj_dw_i[j] = dj_dw_i[j] + err*x_train[i][j]
        dj_db_i = dj_db_i + err
    dj_db = dj_db_i/m
    dj_dw = dj_dw_i/m
    return dj_dw, dj_db

#function for gradient descent
def gradient_descent(x_train, y_train, w, b, alpha, num_of_iters):
    """
    Job: execute gradient descent algorithm for specified number of iterations
    Arguments:
    x_train- training samples, m observations and n house features
    y_train- 1-D array; m target house prices
    w- 1-D array of n weights for n features
    b- scalar, bias parameter
    
    Returns:
    J_history- 1-D array containing cost computed in each iteration
    p_history- 1-D array containing weights and parameters in each iteration.
    w- 1-D array containing final value derived for weight parameters
    b- scalar, final value derived for bias parameter
    """
    w_in = w
    b_in = b
    J_history = []  #will store cost at each iteration.
    p_history = []  #will store weights and bias values per iteration.
    m = x_train.shape[0]
    for i in range(num_of_iters+1):
        #calling compute_gradient() to return gradients of w and b
        dj_dw,dj_db = compute_gradient(x_train, y_train, w_in, b_in)
        w_in = w_in - alpha*dj_dw
        b_in = b_in - alpha*dj_db
        
        if i < 500000:
            J_history.append(compute_cost(x_train, y_train, w_in, b_in))
            p_history.append([w_in,b_in])
            
        if i%math.ceil(num_of_iters/10) == 0:
            print(f"Iteration: {i} ", f"Cost: {J_history[i]:0.3}",
            f"w: {w_in}, b: {b_in}")
    return J_history, p_history, w_in, b_in

#initializing some parameters
w = np.zeros(x_train.shape[1])  #number of weight parameters same as no. of features
b = 0
alpha = 0.01
num_of_iters = 50000

#calling gradient descent
J_history, p_history, final_w, final_b = gradient_descent(x_train, y_train, w, b, alpha, num_of_iters)

#making a prediction for:
#house size = 3500 sqft., number of floors = 2, number of bedrooms = 4, age of house = 6 years

house_features = np.array([3.5, 2, 4, 6])
house_price = np.dot(house_features, final_w) + final_b
print(f"Price of house with 3500sqft, 2 floors, 4 bedrooms and 6 years old is: {house_price}")

        
            