import numpy as np
import util
import os
from cmath import inf

from linear_model import LinearModel


def main():
    """Problem 1(b): Logistic regression with Newton's Method.
    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # *** START CODE HERE ***
    
    # Data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    # data set 1 paths
    train1_path = os.path.join(data_dir, 'ds1_train.csv')
    valid1_path = os.path.join(data_dir, 'ds1_valid.csv')

    # data set 2 paths
    train2_path = os.path.join(data_dir, 'ds2_train.csv')
    valid2_path = os.path.join(data_dir, 'ds2_valid.csv')

    # prediction path
    pred_path = os.path.join(os.path.dirname(__file__), 'output', 'P1')

    # get data for dataset 1
    x_train1, y_train1 = util.load_dataset(train1_path, add_intercept=True)
    x_val1, y_val1 = util.load_dataset(valid1_path, add_intercept=True)

    # get data for dataset 2
    x_train2, y_train2 = util.load_dataset(train2_path, add_intercept=True)
    x_val2, y_val2 = util.load_dataset(valid2_path, add_intercept=True)

    # fit first dataset
    Model1 = LogisticRegression(eps=1e-5)
    Model1.fit(x_train1, y_train1)
    # prediction
    train1_pred = Model1.predict(x_train1)
    val1_pred = Model1.predict(x_val1)

    # results
    print("Dataset 1 training accuracy:", np.mean((train1_pred >= 0.5) == y_train1))
    print("Dataset 1 validation accuracy:", np.mean((val1_pred >= 0.5) == y_val1))

    # plot results
    util.plot(x_train1, y_train1, Model1.theta, os.path.join(pred_path, 'logreg_train1.jpeg') )
    util.plot(x_val1, y_val1, Model1.theta, os.path.join(pred_path, 'logreg_val1.jpeg') )
    # save predictions
    np.savetxt(os.path.join(pred_path, 'logreg_val1.txt'), val1_pred)

    # fit second dataset
    Model2 = LogisticRegression(eps=1e-5)
    Model2.fit(x_train2, y_train2)
    # prediction
    train2_pred = Model2.predict(x_train2)
    val2_pred = Model2.predict(x_val2)

    # results
    print("Dataset 2 training accuracy:", np.mean((train2_pred >= 0.5) == y_train2))
    print("Dataset 2 validation accuracy:", np.mean((val2_pred >= 0.5) == y_val2))
    
    # plot results
    util.plot(x_train2, y_train2, Model2.theta, os.path.join(pred_path, 'logreg_train2.jpeg') )
    util.plot(x_val2, y_val2, Model2.theta, os.path.join(pred_path, 'logreg_val2.jpeg') )
    # save predictions
    np.savetxt(os.path.join(pred_path, 'logreg_val2.txt'), val2_pred)

    
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.
    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def sigmoid(self, x): 
        return 1 / (1 + np.exp( x @ self.theta ))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n)

        # define euclidean distance between old and updated theta
        diff = inf
        count = 0
        while diff > self.eps: 
            theta_cp = self.theta
            # gradient
            grad = 1/m * x.T @ (y - self.sigmoid(x))
            # hessian
            A = (self.sigmoid(x) * (1 - self.sigmoid(x))).reshape( (1, -1) )
            hessian = 1/m * (x.T * A) @ x
            # updated theta
            self.theta = theta_cp - np.linalg.inv(hessian) @ grad
            diff = np.linalg.norm( self.theta - theta_cp )

            count += 1
            if count == 1000: 
                print("{} iterations complete".format(count))

        return

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.
        Args:
            x: Inputs of shape (m, n).
        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return self.sigmoid(x)
        # *** END CODE HERE ***
        
if __name__ == "__main__": 
    main()