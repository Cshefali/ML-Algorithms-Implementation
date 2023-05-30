##UNIVARIATE LINEAR REGRESSION- PYTHON
##Predicting price of house using variable- size of house

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import math

#data
x_train = np.array([1.0,2.0,2.5,3.0,3.7,4.0])   #size of house in 1000sqft.
y_train = np.array([80,160,180,230,265,315])    #price of house in 1000$

#this function computes output of our model (fx) for each observation.
#will be used to plot the model outputs regression line along with training data points
def compute_model_output(x_train, w, b):
    """
    Job- computes the output of this regression model using final values of w and b
    Arguments- 
    x_train- training samples
    w- final weight computed after gradient descent
    b- final bias computed after gradient descent
    Returns-
    A 1-D vector containing predicted output of our model for each training sample.
    """
    m = x_train.shape[0]
    fx = np.zeros(m)
    for i in range(m):
        fx[i] = w*x_train[i] + b
    return fx
    

#function to compute cost
def compute_cost(x_train, y_train, w, b):
    """
    Job- This function computes squared error between predicted output and target output
         averaged over all observations. 
    
    Arguments-
    x_train: numpy array containing all training samples (m)
    y_train: numpy 1-d array containing target outputs
    w- weight parameter
    b- bias parameter
    
    Returns- mean squared error cost function over all the observations.
    """
    m = x_train.shape[0]    #total number of observations.
    total_cost = 0.0
    for i in range(m):
        fx = w*x_train[i] + b
        err = fx - y_train[i]
        total_cost += err**2
    total_cost = total_cost/(2*m)
    return total_cost
    
#function to compute gradient for parameters w and b
def compute_gradient(x_train, y_train, w, b):
    """
    Job- This function computes gradient for both parameters w and b
    Arguments-
    x_train- numpy array containing m observations- here, size of house variable
    y_train- numpy 1-d array containing target outputs
    w- weight paramter
    b- bias parameter
    Returns- gradient of w and b
    """
    m = x_train.shape[0]
    dj_db_i = 0.0
    dj_dw_i = 0.0
    for i in range(m):
        fx = w*x_train[i] + b
        err = fx - y_train[i]
        dj_dw_i = dj_dw_i + err*x_train[i]
        dj_db_i = dj_db_i + err
    dj_dw = dj_dw_i/m
    dj_db = dj_db_i/m
    return dj_dw, dj_db

#function to run gradient descent algorithm
def gradient_descent(x_train, y_train, w, b, alpha, num_of_iters):
    """
    Job- runs gradient descent algo for specified number of iterations to get
        optimized values of w and b. (minimizing cost)
    
    Arguments-
    x_train- numpy array with house size as a variable; m observations.
    y_train- target output; price of house
    w,b- parameters for this univariate linear regression model
    alpha- learning rate
    num_of_iters- total number of iterations for which GD will run.
    
    Returns- history of computed costs, final values of parameters w and b
    """
    #this array will store values of cost for each iteration.
    J_history = []
    #below array stores history of w and b parameter for each iteration.
    p_history = []
    for i in range(num_of_iters):
        dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)
        #updating values of parameters
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        
        #following condition ensures the loop runs for a finite number of times, to prevent resource exhaustion
        if num_of_iters < 100000:
            J_history.append(compute_cost(x_train, y_train, w, b))
            p_history.append([w,b])
        
        #printing cost, values of gradients and parameters
        if i % math.ceil(num_of_iters/10) == 0:
            print(f"Iteration: {i}", f"Cost: {J_history[-1]} ",
            f"dj_dw: {dj_dw: 0.3e}", f"dj_db: {dj_db: 0.3e}",
            f"w: {w}, b: {b}")
    return J_history, p_history, w, b
    
#initializing some more parameters
alpha = 0.1 #learning rate
num_of_iters = 1000    #total number of iterations for GD
w = 0
b = 0

#calling gradient descent
J_history, p_history, final_w, final_b = gradient_descent(x_train, y_train, w, b, alpha, num_of_iters)

#making a prediction
house_size = 4.5    #4500 sqft.
print(f"Predicted price of house of size {house_size} = {final_w*house_size + final_b}", )

#computing fx values for all data points
fx_wb = compute_model_output(x_train, final_w, final_b)

#creating a plot of the data points and the model's regression line

#plotting regression line
plt.plot(x_train, fx_wb, c = "b", label = "Model prediction")

#plotting data points of training sample
plt.scatter(x_train, y_train, marker = "x", color = "r")
plt.title("Housing Price Prediction")
plt.ylabel("Price in 1000$")
plt.xlabel("Size of house in 1000sqft.")
plt.legend()
plt.show()

    

