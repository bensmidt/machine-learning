import numpy as np
import util
import os

from matplotlib import pyplot as plt
from linear_model import LinearModel


def main():
    """Problem 3(d): Poisson regression with gradient ascent.
    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Define file paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_path = os.path.join(data_dir, 'ds4_train.csv')
    valid_path = os.path.join(data_dir, 'ds4_valid.csv')
    pred_path = os.path.join(os.path.dirname(__file__), 'output', 'P3')

    # Get data
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)
    
    # Fit a Poisson Regression model
    Model = PoissonRegression(step_size = 0.0000001, eps = 1e-5)
    Model.fit(x_train, y_train)
    train_pred = Model.predict(x_train)
    val_pred = Model.predict(x_valid)

    # plot validation set
    plt.figure()
    plt.plot(y_valid, val_pred, 'cx')
    plt.savefig(os.path.join(pred_path, 'valid.jpeg'))

    # plot train set
    plt.figure()
    plt.plot(y_train, train_pred, 'cx')
    plt.savefig(os.path.join(pred_path, 'train.jpeg'))

    # save results
    np.savetxt(os.path.join(pred_path, 'valid.txt'), val_pred)


class PoissonRegression(LinearModel):
    """Poisson Regression.
    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def grad(self, x, y): 
        """Calculates the gradient for poisson regression
        Args: 
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m, ).
        """
        m, n = x.shape
        return x.T.dot(y - np.exp(self.theta.dot(x.T))) / m

    def fit(self, x, y, verbose=False):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        m, n = x.shape

        # initialize theta vector
        self.theta = np.zeros(n)
        # find gradient
        accpt_err = False
        i = 0

        while accpt_err == False: 
            i += 1
            theta = np.copy(self.theta)
            self.theta += self.step_size * self.grad(x, y)
            # check if acceptable error met
            theta_diff = np.linalg.norm((theta - self.theta), ord = 1)
            accpt_err = (theta_diff < self.eps)

            if verbose == True and (i % 100 == 0): 
                print('Iterations = {}.'.format(i), 'theta_diff = {}'. format(theta_diff))
        
        return


    def predict(self, x):
        """Make a prediction given inputs x.
        Args:
            x: Inputs of shape (m, n).
        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        return np.exp(x.dot(self.theta))

if __name__ == "__main__": 
    main()