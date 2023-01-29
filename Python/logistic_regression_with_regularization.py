##LOGISTIC REGRESSION 
##Predicting whether a student will be admitted in a university based on marks in 2 subjects
## The model output resolves the problem of overfitting through regularizatio.

import numpy as np
import matplotlib.pyplot as plt
import math


#following function normalizes x_train using min-max scaling.
#data is being normalized to prevent runtime error in logistic loss 
#(sigmoid func returning 1 and leading to nan as output of np.log(1-fx))
def normalize_data(x_train):
    norm = np.linalg.norm(x_train)
    x_train = x_train/norm
    return x_train
    

#sigmoid function that returns values between 0 and 1
def sigmoid(z):
        """
        Job- returns sigmoid value of the argument.
        Argument-
        z- a scalar value or a vector
        Returns-
        gz- scalar or vector depending on the input argument
        """
        
        gz = 1/(1 + np.exp(-z))
        return gz

#function to compute and store the output of our model
def compute_model_output(x_train, w, b):
        """
        Job- computes output of model for all observations
        Argument-
        x_train- (m,n) array; m observation and n features (2)
        w- 1-D array containing weight parameters; n values
        b- scalar, bias parameter
        Returns-
        fx- 1-D array with m values (0 or 1).
            If sigmoid output is 0.5 or less, keep output as 0 else 1.
        """
        
        m = x_train.shape[0]
        fx = np.zeros(m)
        for i in range(m):
                z= sigmoid(np.dot(x_train[i], w) + b)
                if z > 0.5:
                    fx[i] = 1
                else:
                    fx[i] = 0
        return fx

#function to compute cost of the model
def logistic_cost(x_train, y_train, w, b, lambda_):
        """
        Job- computes cost of the model over all m observations
        Returns- cost of the model over all observations
        """
        total_cost = 0.0
        regularized_cost = 0.0
        m,n = x_train.shape
        for i in range(m):
                fx = sigmoid(np.dot(x_train[i], w) + b)
                #print(f"Training Sample: {i}, ", f"y_train = {y_train[i]}", f"Sigmoid output = {fx}")
                loss = -y_train[i]*np.log(fx) - (1 - y_train[i])*np.log(1 - fx)
                total_cost = total_cost + loss
                #print(f"Loss= {loss}", f"total cost so far = {total_cost}")
        total_cost = total_cost/m
        
        for j in range(n):
                regularized_cost += w[j]**2
        
        #print(f"Total cost before Reg: {total_cost}" , f"Regularized cost: {regularized_cost}")
        total_cost = total_cost + (lambda_/(2*m))*regularized_cost
        #print(f"total cost after Reg: {total_cost}")
        return total_cost

#function to compute gradient of parameters w and b
def compute_gradient(x_train, y_train, w, b, lambda_):
        """
        Job- compute gradient of parameters w and b
        
        Arguments-
        x_train- training examples with m observations and n features, here marks of students
        y_trian- target outputs with values 0 or 1, 1- student admitted; 0- not admitted
        w- 1-D vector containing n weight values
        b- scalar value containing bias
        
        Returns- 
        dj_dw- gradient of all weights, 1-D array same as w
        dj_db- gradient of bias parameter
        """
        m, n = x_train.shape
        dj_dw_i = np.zeros(n)
        dj_db_i = 0.0
        for i in range(m):
                fx = sigmoid(np.dot(x_train[i], w) + b)
                err = fx - y_train[i]
                for j in range(n):
                        dj_dw_i += err*x_train[i][j]
                dj_db_i += err
        dj_dw = dj_dw_i/m
        dj_db = dj_db_i/m
        
        for j in range(n):
                dj_dw[j] = dj_dw[j] + (lambda_/m)*w[j]
        
        return dj_dw, dj_db

#function for gradient descent
def gradient_descent(x_train, y_train, w, b, alpha, iters, lambda_):
        """
        Job- implement gradient descent algo 'iters' number of times.
        Returns- 
        J_history- 1D vector containing cost computed at each iteration.
        w- 1D vector containing final values of weight parameters
        b- scalar value, final value of bias
        """
        J_history = []
        w_in = w
        b_in = b
        for i in range(iters):
                #print(f"weights= {w_in}", f"bias = {b_in}")
                dj_dw, dj_db = compute_gradient(x_train, y_train, w_in, b_in, lambda_)
                #print(f"Value of gradients of w: {dj_dw}", f"bias gradient: {dj_db}")
                #updating values of w and b
                w_in = w_in - alpha*dj_dw
                b_in = b_in - alpha*dj_db
                #print(f"Updated value of weights: {w_in}", f"bias= {b_in}")
                if i < 100000:
                        J_history.append(logistic_cost(x_train, y_train, w_in, b_in, lambda_))
                
                if i % math.ceil(iters/10) == 0:
                        print(f"Iteration: {i}", f"Cost: {J_history[i]: 0.3}")
        return J_history, w_in, b_in


#Data
x_train = np.array([[17, 25],
                    [5, 14],
                    [29, 28],
                    [10, 15],
                    [15, 15],
                    [15, 16],
                    [18, 15],
                    [3, 9],
                    [22, 19]])

y_train = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1])

#print("Training samples before normalizing", x_train)

#normalizing the training samples
x_train_norm = normalize_data(x_train)

#print("Training data after normalizing", x_train)


#initializing some more parameters
alpha = 0.8 #learning rate
iters = 1000   #number of iterations
w = np.array([2, 5])
b = 3
lambda_ = 0.07     #regularizion term

#calling gradient descent
J_history, final_w, final_b = gradient_descent(x_train_norm, y_train, w, b, alpha, iters, lambda_)

print("Final weights and bias value:")
print(f"Weights: {final_w}\n",f"Bias: {final_b}")
#making a prediction
student_marks = np.array([[14, 7],
                        [23, 29],
                        [17, 15],
                        [3, 11]])

prediction_test = compute_model_output(student_marks, final_w, final_b)
prediction_train = compute_model_output(x_train_norm, final_w, final_b)

print("Model output for training data:")
print(f"training marks: {x_train}", f"\nmodel output: {prediction_train}")
print(f"Marks of students: {student_marks}\n", f"Predicted Admission: {prediction_test}")