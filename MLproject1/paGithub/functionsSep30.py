##
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import csv

#HELPER FUNCTIONS
def load_data_project(path_dataset,sub_sample = True):
    """Load data and convert it to the metrics system."""

    data = np.genfromtxt(
        path_dataset, delimiter=",",skip_header=1, usecols=range(2,32))
    
    if(path_dataset=="train.csv"):
        labels = np.genfromtxt(
            path_dataset, dtype='str', delimiter=",", skip_header=1,usecols=[1])
        return data, labels

    else:
        return data
    # sub-sample
    #if sub_sample:
    #    data = data[::100]
    #    labels = labels[::100]


def standardize_columns(data, account_for_missing = True):
    """Standardize the original data set."""
        #account_for_missing indicates whether the average value & standard deviation for a parameter should be calculated included values        #-999 (which it obviously shouldn't, as -999 only indicates whether the value is present) or not. If it is true, after standarization missing values are set to 0, so as to not affect the output
    x = np.zeros(data.shape)
    if (account_for_missing == 0 or account_for_missing == 1 ): 
        print('We account for missing values')
        missing_values = np.zeros(data.shape,dtype = bool)
        mean_x= np.zeros(data.shape[1]) #value in position i is the average value of parameter i 
        std_x= np.zeros(data.shape[1]) #value in position i is the standard deviation of parameter i 
        for i in range(data.shape[1]):
            missing_values[data[:,i]==-999,i] = 1
            mean_x[i] = np.mean(data[missing_values[:,i]!=1,i])
            std_x[i] = np.std(data[missing_values[:,i]!=1,i])
            x[:,i] = (data[:,i] - mean_x[i]) / std_x[i]
        if(account_for_missing == 0):
            x[missing_values] = 0
        if(account_for_missing == 1):
            for i in range(data.shape[1]):
                x[missing_values[:,i],i] = np.random.normal(0, 1, sum(missing_values[:,i]))
                
            #alpha=np.random.random(1)[0]
            #x[missing_values] = mean_x[i]+alpha*std_x[i]
            #print('alpha',alpha)

    else: 
        missing_values = 0 
        mean_x = np.mean(data,0)
        x = data - mean_x
        std_x = np.std(x,0)
        x = x / std_x
    return x, mean_x, std_x, missing_values

#I ADDED 1 TO THE NAME TO DIFFERENTIATE FROM THE FUNCTION WE USED IN LAB 2
def build_model_data1(data, missing_values, input_missing, FEATURE_EXPANSION, degree):
    """Form (y,tX) to get regression data in matrix form."""
    print('Size of the data before expansion? ')
    print(data.shape)
    #variable input_missing indicates whether the positions of the missing values should be included as input
    if(FEATURE_EXPANSION):
        data = build_poly(data, degree)
    print('Size of the data after expansion? ')
    print(data.shape)
  
        
    if(input_missing): 
        tx = np.concatenate((np.ones((data.shape[0],1)), data, missing_values),1)
        #tx = np.c_[np.ones(data.shape[0]), data, input_missing]
    else: 
        tx = np.c_[np.ones(data.shape[0]), data]
    return tx


def compute_loss(y, tx, w):
    """Calculate the loss using MSE
    """
    #f_x = tx.dot(w)
    #return sum(pow(y-f_x,2))
    y = np.array(y)
    tx = np.array(tx)
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    grad_w=(y-tx.dot(w)).dot(tx)*(-1/len(y))
    return grad_w

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        grad_w,grad_b = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        w = w - np.multiply(grad_w, gamma)
        # TODO: update w by gradient
        # ***************************************************
        # store w and loss
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w



#copy pasted from what the professor gave us in lab2
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


            
#MODEL FUNCTIONS (THEY ALL RETURN LOSS, W)
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        grad_w = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        w = w - np.multiply(grad_w, gamma)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w

#this one is still not finished
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    batch_size = 1 #ideally this would be an input to the function
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y,tx,batch_size):
                gradients = compute_gradient(minibatch_y, minibatch_tx, w)
                w = w - np.multiply(gamma, gradients)
                loss = compute_loss(minibatch_y,minibatch_tx,w)

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
        #compute real loss for all samples
        loss = compute_loss(y,tx,w)
    return loss, w

def least_squares(y, tx): 
    tx = np.array(tx)
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    loss = compute_loss(y,tx,w)
    return loss, w

def ridge_regression(y,tx,lambda_):
    tx = np.array(tx)
    """Computes ridge regression, returns weights"""  
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w=np.linalg.solve(a, b)
    loss=compute_loss(y,tx,w)
    return loss,w

def predictions(tx,w_final):
    """Predicts the labels """
    predictions = tx.dot(w_final)
    predictions[predictions>0] = 1
    predictions[predictions<=0]=-1
    return predictions

def error_predicting(predictions,y):
    """Computes the error between the predicted value and the real one"""
    errors = np.zeros(predictions.shape)
    errors[predictions != y] = 1 
    nerrors = np.sum(errors)
    percentage=nerrors/len(y)
    print('you have obtained {n} errors: = {p}%'.format(n=nerrors,p=percentage))

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, model,k_indices, k, w_initial, max_iters, 
                                 gamma, lambda_):
    """return the loss calculated using crossvalidation.""" 
    x_te=[]
    x_tr=[]
    y_te=[]
    y_tr=[]
    loss_tr = []
    loss_te = []
    
    for i in range(k):
        x_te.extend(x[k_indices[i]])
        y_te.extend(y[k_indices[i]])
        for j in range(k):
            if (j!=i):
                x_tr.extend(x[k_indices[j]])
                y_tr.extend(y[k_indices[j]])
        
        #tx_tr = build_poly(x_tr, degree)
        #tx_te = build_poly(x_te, degree)
        
        loss,w_final = train_model (y_tr, x_tr, model, w_initial, max_iters, gamma, lambda_)
 
        #loss,w_final = ridge_regression(y_tr, tx_tr, lambda_)
        loss_tr.append(2*compute_loss(y_tr, x_tr, w_final))
        loss_te.append(2*compute_loss(y_te, x_te, w_final))
        
    loss_tr_mean=np.mean(loss_tr)
    loss_te_mean=np.mean(loss_te)
    return loss_tr_mean, loss_te_mean

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.figure()
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    plt.show()

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def train_model (y, tx, model, w_initial = 'Doesn-t apply', max_iters  = 'Doesn-t apply', 
                                 gamma = 'Doesn-t apply', lambda_  = 'Doesn-t apply'):
    if(model == 0):
        loss, w_final = least_squares_GD(np.array(y), np.array(tx), w_initial, max_iters, gamma)
    elif(model == 1):
        loss, w_final = least_squares_SGD(y, tx, w_initial, max_iters*100, gamma) #to work reasonably well, stochastic gradient
        #needs more iterations than gradient descent,as batch_size = 1
    elif(model == 2):
        loss, w_final = least_squares(y, tx) 
        print('heyyyyyyyyyyyyyyyyyyyyyy')
    elif(model == 3):
        loss,w_final = ridge_regression(y, tx, lambda_)
    #elif(model == 4):
        #TO DO 
    #elif(model == 5):
        #TO DO 
        
    return loss, w_final

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})



