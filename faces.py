from numpy import *
from numpy.linalg import norm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import *
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imresize, imread, imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import shutil
import sys

def f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return (1/(2*float(len(y))))*sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))

    return (-1/float(len(y)))*sum((y-dot(theta.T, x))*x, 1)
def f_m(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum(np.square((y - dot(theta.T,x).T)))

def df_m(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return 2*dot(x, (dot(theta.T,x)-y.T).T)

def get_finite_diff(x, y, theta, h):
    G = df_m(x, y, theta)
    row = [i for i in range(1025)]
    col = [i for i in range(6)]
    random.shuffle(row)
    random.shuffle(col)
    for i in range(5):
        H = np.zeros([1025, 6])
        H[row[i], col[i]] = h
        finite_diff = G[row[i], col[i]] - (f_m(x, y, theta + H) - f_m(x, y, theta))/h
        print("The difference between the gradient and approximation at" + "(" + str(row[i]) + ", " + str(col[i]) + ") is: "  + str(finite_diff))


def grad_descent_binary(f, df, x, y, init_t, alpha, iterations):
    EPS = 1e-5
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = iterations
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()

        d = df(x, y, t).reshape(1025, 1)
        t -= alpha*d

        iter += 1
    return t
def grad_descent_multi(f, df, x, y, init_t, alpha, iterations):
    EPS = 1e-5
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = iterations
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        d = df(x, y, t)
        t -= alpha*d

        iter += 1
    return t

def get_prediction_binary(theta, x):
    x = vstack( (ones((1, x.shape[1])), x))
    h = dot(theta.T,x)
    predictions = []
    for i in range(len(h[0])):
        if h[0][i] > 0:
            predictions.append(1)
        if h[0][i] < 0:
            predictions.append(-1)

    return predictions
def get_prediction_m(theta, x):
    x = vstack( (ones((1, x.shape[1])), x))
    h = dot(theta.T,x).T
    predictions = []
    for i in range(len(h)):
        max_idx = 0
        max = h[i][0]
        for j in range(len(h[i])):
            if h[i][j] > max:
                max = h[i][j]
                max_idx = j
        predictions.append(max_idx)
    return predictions


def num_correct_predictions(predictions, Y):
    num_correct = 0
    total = 0
    #print predictions
    #print Y
    #print len(Y)
    for i in range(len(predictions)):
        total += 1
        if predictions[i] == Y[i]:
            num_correct += 1
    return num_correct, total
def num_correct_predictions_m(predictions, Y):
    num_correct = 0
    total = 0
    for i in range(len(predictions)):
        total+= 1
        if Y[i][predictions[i]] == 1:
            num_correct+=1
    return num_correct, total


def get_sets(actors, training_size, test_size, validation_size):
    X = np.empty([0, 1024])
    training_X = []
    test_X = []
    validation_X = []
    training_Y = []
    test_Y = []
    validation_Y = []
    label = -1
    total_size = training_size + test_size + validation_size
    training_range = [i for i in range(70)]
    test_range = [i for i in range(80, 90)]
    validation_range = [i for i in range(70, 80)]
    random.shuffle(training_range)
    random.shuffle(test_range)
    random.shuffle(validation_range)
    #print training_range

    for bin in actors:
        for act in bin:
            j = 0
            i = 0
            while (j < training_size):
                filename = act + str(training_range[i]) + '.jpg'
                filename1 = act + str(training_range[i]) + '.jpeg'
                filename2 = act + str(training_range[i]) + '.JPG'
                filename3 = act + str(training_range[i]) + '.png'
                files = [filename, filename1, filename2, filename3]
                for filename in files:
                    if os.path.isfile("training/" + filename):
                        imarray = imread("training/" + filename)
                        X = vstack((X, reshape(np.ndarray.flatten(imarray), [1,1024])))
                        break
                training_Y.append(label)
                i+=1
                j+=1


            i = 0
            while (j < training_size + validation_size):
                filename = act + str(validation_range[i]) + '.jpg'
                filename1 = act + str(validation_range[i]) + '.jpeg'
                filename2 = act + str(validation_range[i]) + '.JPG'
                filename3 = act + str(validation_range[i]) + '.png'
                files = [filename, filename1, filename2, filename3]
                for filename in files:
                    if os.path.isfile("validation/" + filename):
                        imarray = imread("validation/" + filename)
                        X = vstack((X, reshape(np.ndarray.flatten(imarray), [1,1024])))
                        break
                validation_Y.append(label)
                i+=1
                j+=1

            i = 0
            while (j< training_size + test_size + validation_size):
                filename = act + str(test_range[i]) + '.jpg'
                filename1 = act + str(test_range[i]) + '.jpeg'
                filename2 = act + str(test_range[i]) + '.JPG'
                filename3 = act + str(test_range[i]) + '.png'
                files = [filename, filename1, filename2, filename3]
                for filename in files:
                    if os.path.isfile("test/" + filename):
                        imarray = imread("test/" + filename)
                        X = vstack((X, reshape(np.ndarray.flatten(imarray), [1,1024])))
                        break
                test_Y.append(label)
                i+=1
                j+=1

        label += 2

    X /=255
    for bin in actors:
        for i in range(len(bin)):
            training_X.extend(X[:training_size])
            validation_X.extend(X[training_size:validation_size+training_size])
            test_X.extend(X[validation_size+training_size:total_size])
            X = X[total_size:]


    #training_X.extend(X[total_size:total_size+training_size])
    #validation_X.extend(X[total_size+training_size:total_size+training_size+validation_size])
    #test_X.extend(X[total_size+validation_size+test_size:total_size+total_size])

    return np.array(training_X), np.array(test_X), np.array(validation_X), np.array(training_Y), np.array(test_Y), np.array(validation_Y)
def get_sets_multiclass(actors, training_size, test_size, validation_size):
    X = np.empty([0, 1024])
    Y = np.empty([0, len(actors)])
    training_X = []
    test_X = []
    validation_X = []
    training_Y = []
    test_Y = []
    validation_Y = []
    label_idx = 0
    total_size = training_size + test_size + validation_size
    training_range = [i for i in range(70)]
    test_range = [i for i in range(80, 90)]
    validation_range = [i for i in range(70, 80)]
    random.shuffle(training_range)
    random.shuffle(test_range)
    random.shuffle(validation_range)
    #print training_range

    for bin in actors:
        for act in bin:
            j = 0
            i = 0
            while (j < training_size):
                filename = act + str(training_range[i]) + '.jpg'
                filename1 = act + str(training_range[i]) + '.jpeg'
                filename2 = act + str(training_range[i]) + '.JPG'
                filename3 = act + str(training_range[i]) + '.png'
                files = [filename, filename1, filename2, filename3]
                for filename in files:
                    if os.path.isfile("training/" + filename):
                        imarray = imread("training/" + filename)
                        X = vstack((X, reshape(np.ndarray.flatten(imarray), [1,1024])))
                        break
                new_label = np.random.rand(1, len(actors))*(0)
                new_label[0][label_idx] = 1
                Y = vstack((Y, new_label))
                i+=1
                j+=1


            i = 0
            while (j < training_size + validation_size):
                filename = act + str(validation_range[i]) + '.jpg'
                filename1 = act + str(validation_range[i]) + '.jpeg'
                filename2 = act + str(validation_range[i]) + '.JPG'
                filename3 = act + str(validation_range[i]) + '.png'
                files = [filename, filename1, filename2, filename3]
                for filename in files:
                    if os.path.isfile("validation/" + filename):
                        imarray = imread("validation/" + filename)
                        X = vstack((X, reshape(np.ndarray.flatten(imarray), [1,1024])))
                        break
                new_label = np.random.rand(1, len(actors))*(0)
                new_label[0][label_idx] = 1
                Y = vstack((Y, new_label))
                i+=1
                j+=1

            i = 0
            while (j< training_size + test_size + validation_size):
                filename = act + str(test_range[i]) + '.jpg'
                filename1 = act + str(test_range[i]) + '.jpeg'
                filename2 = act + str(test_range[i]) + '.JPG'
                filename3 = act + str(test_range[i]) + '.png'
                files = [filename, filename1, filename2, filename3]
                for filename in files:
                    if os.path.isfile("test/" + filename):
                        imarray = imread("test/" + filename)
                        X = vstack((X, reshape(np.ndarray.flatten(imarray), [1,1024])))
                        break
                new_label = np.random.rand(1, len(actors))*(0)
                new_label[0][label_idx] = 1
                Y = vstack((Y, new_label))
                i+=1
                j+=1

        label_idx += 1

    X /=255
    for bin in actors:
        for i in range(len(bin)):
            training_X.extend(X[:training_size])
            validation_X.extend(X[training_size:validation_size+training_size])
            test_X.extend(X[validation_size+training_size:total_size])
            X = X[total_size:]

    for i in range(len(actors)):
        training_Y.extend(Y[i*total_size : i*total_size+training_size])
        validation_Y.extend(Y[i*total_size+training_size : i*total_size+training_size+validation_size])
        test_Y.extend(Y[i*total_size+training_size+validation_size : (i+1)*(total_size)])
    #training_X.extend(X[total_size:total_size+training_size])
    #validation_X.extend(X[total_size+training_size:total_size+training_size+validation_size])
    #test_X.extend(X[total_size+validation_size+test_size:total_size+total_size])

    return np.array(training_X), np.array(test_X), np.array(validation_X), np.array(training_Y), np.array(test_Y), np.array(validation_Y)

def part3():
    actors = [['baldwin'], ['carell']]
    training_X, test_X, validation_X, training_Y, test_Y, validation_Y = get_sets(actors, 70, 10, 10)
    theta0 = np.zeros([1025, 1])
    theta = grad_descent_binary(f, df, training_X.T, training_Y, theta0, 1E-3, 30000)
    predictions_training = get_prediction_binary(theta, training_X.T)
    predictions_validation = get_prediction_binary(theta, validation_X.T)


    num_correct_training, total_training = num_correct_predictions(predictions_training, training_Y)
    num_correct_validation, total_validation = num_correct_predictions(predictions_validation, validation_Y)

    percentage_correct_training = (float(num_correct_training)/float(total_training))*100
    percentage_correct_validation = (float(num_correct_validation)/float(total_validation))*100
    cost_training = f(training_X.T, training_Y, theta)
    cost_validation = f(validation_X.T, validation_Y, theta)
    print("The percentage of correct predictions on the training set is: " + str(percentage_correct_training) + "%")
    print("The percentage of correct predictions on the validation set is: " + str(percentage_correct_validation) + "%")
    print("The value of the cost function on the training set is: " + str(cost_training))
    print("The value of the cost function on the validation set is: " + str(cost_validation))
    #predictions_test = get_prediction_binary(theta, test_X.T)
    #num_correct_test, total_test = num_correct_predictions(predictions_test, test_Y)
    #print("The number of correct predictions on the test set is: " + str(float(num_correct_test*100)/float(total_test)) + "%")
def part4a():
    actors = [['baldwin'], ['carell']]
    training_X_full, test_X_full, validation_X_full, training_Y_full, test_Y_full, validation_Y_full = get_sets(actors, 70, 10, 10)
    training_X_part, test_X_part, validation_X_part, training_Y_part, test_Y_part, validation_Y_part = get_sets(actors, 10, 10, 10)
    theta0 = np.zeros([1025, 1])
    theta_full = grad_descent_binary(f, df, training_X_full.T, training_Y_full, theta0, 1E-3, 30000)
    theta_full = theta_full*255
    theta_part = grad_descent_binary(f, df, training_X_part.T, training_Y_part, theta0, 1E-3, 30000)
    theta_part = theta_part*255

    fig=plt.figure(figsize=(100, 100))
    fig.add_subplot(1, 2, 1)
    plt.imshow(theta_full[1:].reshape(32, 32), cm.coolwarm)
    fig.add_subplot(1, 2, 2)
    plt.imshow(theta_part[1:].reshape(32, 32), cm.coolwarm)
    plt.show()

def part4b():
    actors = [['baldwin'], ['carell']]
    training_X_full, test_X_full, validation_X_full, training_Y_full, test_Y_full, validation_Y_full = get_sets(actors, 70, 10, 10)
    theta0 = np.zeros([1025, 1])
    theta1 = grad_descent_binary(f, df, training_X_full.T, training_Y_full, theta0, 1E-3, 30000)
    theta2 = grad_descent_binary(f, df, training_X_full.T, training_Y_full, theta0, 1E-3, 10)
    theta1*=255
    theta2*=255
    fig=plt.figure(figsize=(100, 100))
    fig.add_subplot(1, 2, 1)
    plt.imshow(theta1[1:].reshape(32, 32), cm.coolwarm)
    fig.add_subplot(1, 2, 2)
    plt.imshow(theta2[1:].reshape(32, 32), cm.coolwarm)
    plt.show()
def part5():
    act = [['bracco', 'gilpin', 'harmon'],['baldwin', 'hader', 'carell']]
    other_act = [['drescher', 'ferrera', 'chenoweth'],['butler', 'vartan', 'radcliffe']]
    training_sizes = [10, 20, 30, 40, 50, 60, 70]
    performance_training = []
    performance_validation = []
    performance_other = []
    training_X_other, test_X_other, validation_X_other, training_Y_other, test_Y_other, validation_Y_other = get_sets(other_act, 70, 10, 10)

    for training_size in training_sizes:
        training_X, test_X, validation_X, training_Y, test_Y, validation_Y = get_sets(act, training_size, 8, 10)
        theta0 = np.zeros([1025, 1])
        theta = grad_descent_binary(f, df, training_X.T, training_Y, theta0, 1E-3, 30000)
        predictions_training = get_prediction_binary(theta, training_X.T)
        predictions_validation = get_prediction_binary(theta, validation_X.T)
        num_correct_training, total_training = num_correct_predictions(predictions_training, training_Y)
        num_correct_validation, total_validation = num_correct_predictions(predictions_validation, validation_Y)
        percentage_correct_training = (float(num_correct_training)/float(total_training))*100
        percentage_correct_validation = (float(num_correct_validation)/float(total_validation))*100
        performance_training.append(percentage_correct_training)
        performance_validation.append(percentage_correct_validation)
        predictions_other = get_prediction_binary(theta, validation_X_other.T)
        num_correct_other, total_other = num_correct_predictions(predictions_other, validation_Y_other)
        percentage_correct_other = (float(num_correct_other)/float(total_other))*100
        performance_other.append(percentage_correct_other)
        print("The percentage of correct predictions on the validation set of actors not trained on after training on a set of size " + str(training_size) + " is: "  + str(percentage_correct_other) + "%")
    plt.plot(training_sizes, performance_training, color='g', linewidth=2, marker='o', label="Training Set Performance")
    plt.plot(training_sizes, performance_validation, color='k', linewidth=2, marker='o', label="Validation Set Performance")
    plt.plot(training_sizes, performance_other, color='r', linewidth=2, marker='o', label="Other Validation Set Performance")

    plt.title('Training Size vs. Percentage of correct predictions on some datasets')
    plt.ylim([0,110])
    plt.xlabel('Training Size')
    plt.ylabel('Percentage of correct predictions')
    plt.legend()
    plt.show()
def part6():
    act = [['bracco'], ['gilpin'], ['harmon'],['baldwin'], ['hader'], ['carell']]
    training_X, test_X, validation_X, training_Y, test_Y, validation_Y = get_sets_multiclass(act, 70, 8, 10)
    theta0 = np.zeros([1025, 6])
    theta = grad_descent_multi(f, df_m, training_X.T, training_Y, theta0, 1E-6, 3000)
    get_finite_diff(training_X.T, training_Y, theta, 0.000001)

def part7():
    act = [['bracco'], ['gilpin'], ['harmon'],['baldwin'], ['hader'], ['carell']]
    training_X, test_X, validation_X, training_Y, test_Y, validation_Y = get_sets_multiclass(act, 70, 8, 10)
    theta0 = np.zeros([1025, 6])
    theta = grad_descent_multi(f, df_m, training_X.T, training_Y, theta0, 1E-6, 30000)
    predictions = get_prediction_m(theta, validation_X.T)
    predictions_training = get_prediction_m(theta, training_X.T)
    predictions_validation = get_prediction_m(theta, validation_X.T)


    num_correct_training, total_training = num_correct_predictions_m(predictions_training, training_Y)
    num_correct_validation, total_validation = num_correct_predictions_m(predictions_validation, validation_Y)

    percentage_correct_training = (float(num_correct_training)/float(total_training))*100
    percentage_correct_validation = (float(num_correct_validation)/float(total_validation))*100
    print("The percentage of correct predictions on the training set is: " + str(percentage_correct_training) + "%")
    print("The percentage of correct predictions on the validation set is: " + str(percentage_correct_validation) + "%")

def part8():
    act = [['bracco'], ['gilpin'], ['harmon'],['baldwin'], ['hader'], ['carell']]
    training_X, test_X, validation_X, training_Y, test_Y, validation_Y = get_sets_multiclass(act, 70, 8, 10)
    theta0 = np.zeros([1025, 6])
    theta = grad_descent_multi(f, df_m, training_X.T, training_Y, theta0, 1E-6, 3000)
    theta = theta*255
    theta = theta.T
    fig=plt.figure(figsize=(100, 100))
    for i in range(len(theta)):
        fig.add_subplot(2, 3, i + 1)
        plt.imshow(theta[i][1:].reshape(32, 32), cm.coolwarm)

    plt.show()


if __name__ == "__main__":

    if sys.argv[1] == "part3":
        part3()
    if sys.argv[1] == "part4a":
        part4a()
    if sys.argv[1] == "part4b":
        part4b()
    if sys.argv[1] == "part5":
        part5()
    if sys.argv[1] == "part6":
        part6()
    if sys.argv[1] == "part7":
        part7()
    if sys.argv[1] == "part8":
        part8()
