# Important note: you do not have to modify this file for your homework.
import os
import util
import numpy as np
from matplotlib import pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y)) + 0.0001 * np.linalg.norm(theta) # Regularization term

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 1

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)

        # learning_rate /= i**2

        theta = theta - (learning_rate) * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)

            print('Training loss = %f' % np.mean(np.log(1 + np.exp(-Y * X.dot(theta)))))
            
            print('||theta_k - theta_{k-1}|| = %f' % np.linalg.norm(prev_theta - theta))
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_path_a = os.path.join(data_path, 'ds1_a.csv')
    train_path_b = os.path.join(data_path, 'ds1_b.csv')
    pred_path = os.path.join(os.path.dirname(__file__), 'output', 'P1')
    
    # get data
    Xa, Ya = util.load_csv(train_path_a, add_intercept=True)
    Xb, Yb = util.load_csv(train_path_b, add_intercept=True)

    # check what datasets look like
    dataset1a = plt.figure()
    plt.plot(Xa[Ya==1, 1], Xa[Ya==1, 2], 'bx', linewidth=2)
    plt.plot(Xa[Ya==-1, 1], Xa[Ya==-1, 2], 'go', linewidth=2)
    plt.title("Dataset A")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    dataset1a.savefig(os.path.join(pred_path, 'Dataset-1A.jpeg'))

    dataset1b = plt.figure()
    plt.plot(Xb[Yb==1, 1], Xb[Yb==1, 2], 'bx', linewidth=2)
    plt.plot(Xb[Yb==-1, 1], Xb[Yb==-1, 2], 'go', linewidth=2)
    plt.title("Dataset B")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    dataset1b.savefig(os.path.join(pred_path, 'Dataset-1B.jpeg'))

    # Logistic regression dataset A
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv(train_path_a, add_intercept=True)
    logistic_regression(Xa, Ya)

    # Logistic regression dataset B
    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv(train_path_b, add_intercept=True)
    logistic_regression(Xb, Yb)


if __name__ == '__main__':
    main()
