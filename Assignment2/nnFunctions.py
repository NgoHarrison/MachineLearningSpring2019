import numpy as np
from scipy.optimize import minimize
from math import sqrt
import math
from numpy import exp as exp
import pickle
'''
You need to modify the functions except for initializeWeights() and preprocess()
'''

def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer
    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer
    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def preprocess(filename,scale=True):
    '''
     Input:
     filename: pickle file containing the data_size
     scale: scale data to [0,1] (default = True)
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    '''
    with open(filename, 'rb') as f:
        train_data = pickle.load(f)
        train_label = pickle.load(f)
        test_data = pickle.load(f)
        test_label = pickle.load(f)
    # convert data to double
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    # scale data to [0,1]
    if scale:
        train_data = train_data/255
        test_data = test_data/255

    return train_data, train_label, test_data, test_label

def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''
    # your code here - remove the next four lines
    #if np.isscalar(z):
        #s = 0
    #else:
        #s = np.zeros(z.shape)
    #return s
    sigmoid_function = 1 / (1 + exp(-z))
    return sigmoid_function



def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not including the bias node)
    % n_hidden: number of node in hidden layer (not including the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % train_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    '''
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2

    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    # Your code here
    #
    #
    #
    #
    #Most comments are to help me visualize the shape
    #print(W1.shape)         (3,6)
    #print(train_data.shape) (2,5)
    #(2,5) x (3,6)
    bias_input_array = np.ones((train_data.shape[0],1))
    train_data_with_bias = np.append(train_data,bias_input_array,1)
    #print(train_data)
    #print(train_data_with_bias.shape)    (2,6)
    w1_transpose = np.transpose(W1)
    #print(w1_transpose.shape)
    hidden_matrix = np.dot(train_data_with_bias,w1_transpose)
    #print(hidden_matrix.shape)   (2,3)
    #print(hidden_matrix)
    #print(sigmoid(hidden_matrix))
    sigmoid_hidden_matrix = sigmoid(hidden_matrix)
    #print(sigmoid_hidden_matrix.shape)   (2,3)
    #print(W2.shape)                      (2,4)
    bias_sigmoid = np.ones((sigmoid_hidden_matrix.shape[0],1))
    sigmoid_hidden_matrix_with_bias = np.append(sigmoid_hidden_matrix,bias_sigmoid,1)
    #print(sigmoid_hidden_matrix_with_bias.shape)    (2,4)
    w2_transpose = np.transpose(W2)
    output_matrix = np.dot(sigmoid_hidden_matrix_with_bias,w2_transpose)
    #print(output_matrix.shape)
    output_matrix_sigmoid = sigmoid(output_matrix)
    #print(output_matrix_sigmoid)
    train_label_k_encoding = one_of_k(train_label)
    #print(train_label_k_encoding)
    #print(train_label_k_encoding.shape)       (2,2)
    #print(output_matrix_sigmoid.shape)        (2,2)
    one_matrix_out = np.ones((output_matrix_sigmoid.shape))
    one_matrix_train = np.ones((train_label_k_encoding.shape))

    obj_val = (-1/output_matrix_sigmoid.shape[0]) * np.sum((train_label_k_encoding *np.log(output_matrix_sigmoid)) + ((1 - train_label_k_encoding)*np.log(1 - output_matrix_sigmoid)))
    delta = output_matrix_sigmoid - train_label_k_encoding
    #print(delta.shape)    (2,2)
    #print(sigmoid_hidden_matrix_with_bias.shape)    (2,3)

    # W2
    #partial_w2 = delta*sigmoid_hidden_matrix_with_bias
    partial_w2 = np.dot(np.transpose(delta), sigmoid_hidden_matrix_with_bias)
    grad_W2 = (1 / train_data.shape[0]) * (partial_w2 + (lambdaval * W2))
    # W1
    partial_w1 = ((1 - sigmoid_hidden_matrix_with_bias) * sigmoid_hidden_matrix_with_bias) * (np.dot(delta, W2))
    # print(partial_w1.shape) (2,4)
    # print(train_data_with_bias.shape)    (2,6)    (2,4)*(2,6) -> (4,2)*(2,6)
    partial_w1 = np.dot(np.transpose(partial_w1), train_data_with_bias)
    # print(W1.shape)   (3,6)
    # print(partial_w1)  (4,6)
    # The last row is all zeros, so I decided to erase it from the matrix to match the shapes.
    # This won't affect computation since we are just adding
    grad_W1 = (1 / train_data.shape[0]) * (partial_w1[0:partial_w1.shape[0] - 1] + (lambdaval * W1))

    new_obj_val = obj_val + (lambdaval / (2 * train_data.shape[0])) * ((np.sum(W1 ** 2)) + (np.sum(W2 ** 2)))

    # Make sure you reshape the gradient matrices to a 1D array. for instance if
    # your gradient matrices are grad_W1 and grad_W2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_W1.flatten(), grad_W2.flatten()), 0)
    # obj_grad = np.zeros(params.shape)

    # Make sure you reshape the gradient matrices to a 1D array. for instance if
    # your gradient matrices are grad_W1 and grad_W2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_W1.flatten(), grad_W2.flatten()),0)
    #obj_grad = np.zeros(params.shape)

    return (new_obj_val, obj_grad)

def one_of_k(train_label):
    #1ofK encoding for the train_label
    new_train_label = np.array(train_label,dtype=np.int)
    ret = np.zeros((train_label.size, np.unique(train_label).size))
    # print(ret)
    # Z = np.zeros((4,3))
    # z = [2,0,1,1]
    # (0,2),(1,0),(2,1),(3,1)
    # Y[(0,1,2,3),(2,0,1,1)]]=1
    index = np.arange(0, train_label.size, 1)

    ret[index, new_train_label[0:train_label.size]] = 1
    # print(ret)
    return ret

def forward_prop(W1 , W2, train_data):
    # Most comments are to help me visualize the shape
    # print(W1.shape)         (3,6)
    # print(train_data.shape) (2,5)
    # (2,5) x (3,6)
    bias_input_array = np.ones((train_data.shape[0], 1))
    train_data_with_bias = np.append(train_data, bias_input_array, 1)
    # print(train_data)
    # print(train_data_with_bias)
    w1_transpose = np.transpose(W1)
    hidden_matrix = np.dot(train_data_with_bias, w1_transpose)
    sigmoid_hidden_matrix = sigmoid(hidden_matrix)
    # print(sigmoid_hidden_matrix.shape)   (2,3)
    # print(W2.shape)                      (2,4)
    bias_sigmoid = np.ones((sigmoid_hidden_matrix.shape[0], 1))
    sigmoid_hidden_matrix_with_bias = np.append(sigmoid_hidden_matrix, bias_sigmoid, 1)
    # print(sigmoid_hidden_matrix_with_bias.shape)    (2,4)
    w2_transpose = np.transpose(W2)
    output_matrix = np.dot(sigmoid_hidden_matrix_with_bias, w2_transpose)
    print(output_matrix.shape)
    output_matrix_sigmoid = sigmoid(output_matrix)
    # print(output_matrix_sigmoid)
    return output_matrix_sigmoid

def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.
    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image
    % Output:
    % label: a column vector of predicted labels
    '''

    #labels = np.zeros((data.shape[0],))
    # Your code here

    #call forward pass
    final = forward_prop(W1,W2,data)
    #get the max values along each row
    #zero causes error so try axis = 1
    return np.argmax(final,1)


def main():
    # Paste your sigmoid function here

    # Paste your nnObjFunction here

    n_input = 5
    n_hidden = 3
    n_class = 2
    training_data = np.array([np.linspace(0, 1, num=5), np.linspace(1, 0, num=5)])
    training_label = np.array([0, 1])
    lambdaval = 0
    params = np.linspace(-5, 5, num=26)
    args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)
    objval, objgrad = nnObjFunction(params, *args)
    print("Objective value:")
    print(objval)
    print("Gradient values: ")
    print(objgrad)

main()
