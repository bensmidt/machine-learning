import matplotlib.pyplot as plt
import numpy as np
import util
import os

from linear_model import LinearModel


def main():
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """

    # get paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_path = os.path.join(data_dir, 'ds5_train.csv')
    valid_path = os.path.join(data_dir, 'ds5_valid.csv')
    pred_path = os.path.join(os.path.dirname(__file__), 'output', 'P5')

    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    Model = LocallyWeightedLinearRegression(tau = 0.5)
    Model.fit(x_train, y_train)

    # Load validation set
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    predictions = Model.predict(x_valid)

    # Get MSE value on the validation set
    mse = np.mean((predictions - y_valid)**2)
    print(mse)

    # Plot validation predictions on top of training set
    plt.figure()
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_valid, predictions, 'go')
    plt.savefig(os.path.join(pred_path, 'p05b_lwr.jpeg'))
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        self.x = x
        self.y = y

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        
        # get shape of training data
        m, n = x.shape

        # create predictions vector
        predictions = np.zeros(m)

        # get weights for each input
        for i in range(m): 
            w = np.exp( -np.sum((self.x - x[i])**2, axis = 1) / (2*(self.tau**2)) )
            w = np.diag(w)
            theta = np.linalg.inv(self.x.T @ w @ self.x) @ self.x.T @ w @ self.y
            predictions[i] = theta @ x[i].T

        return predictions

if __name__ == "__main__": 
    main()