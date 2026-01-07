"""
Main model training function for logistic regression.
"""

import numpy as np
from .initialize import initialize_with_zeros
from .optimize import optimize
from .predict import predict


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to True to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    # initialize parameters with zeros
    # and use the "shape" function to get the first dimension of X_train
    w, b = initialize_with_zeros(X_train.shape[0])
    
    # Gradient descent 
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "params"
    w = params["w"]
    b = params["b"]
    
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    # Print train/test Errors
    if print_cost:
        train_acc = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
        test_acc = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
        print(f"\n[MODEL] Train accuracy: {train_acc:.2f}%")
        print(f"[MODEL] Test accuracy: {test_acc:.2f}%")
        print(f"[MODEL] Train predictions: {Y_prediction_train.flatten()[:min(10, Y_prediction_train.shape[1])]}")
        print(f"[MODEL] Train actual: {Y_train.flatten()[:min(10, Y_train.shape[1])]}")
        print(f"[MODEL] Test predictions (first 10): {Y_prediction_test.flatten()[:min(10, Y_prediction_test.shape[1])]}")
        print(f"[MODEL] Test actual (first 10): {Y_test.flatten()[:min(10, Y_test.shape[1])]}")
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train": Y_prediction_train, 
         "w": w, 
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    
    return d

