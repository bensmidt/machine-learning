import numpy as np
import util
import os

from linear_model import LinearModel


def main():
    """Problem 1(e): Gaussian discriminant analysis (GDA)
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
    x_train1, y_train1 = util.load_dataset(train1_path, add_intercept=False)
    x_train1_int, y_train1_int = util.load_dataset(train1_path, add_intercept=True) # intercept for prediction/results
    x_val1, y_val1 = util.load_dataset(valid1_path, add_intercept=True)

    # get data for dataset 2
    x_train2, y_train2 = util.load_dataset(train2_path, add_intercept=False)
    x_train2_int, y_train2_int = util.load_dataset(train2_path, add_intercept=True) # intercept for prediction/results
    x_val2, y_val2 = util.load_dataset(valid2_path, add_intercept=True)

    # fit first dataset
    Model1 = GDA()
    Model1.fit(x_train1, y_train1)
    # prediction
    train1_pred = Model1.predict(x_train1_int)
    val1_pred = Model1.predict(x_val1)

    # results
    print("Dataset 1 training accuracy:", np.mean((train1_pred >= 0.5) == y_train1))
    print("Dataset 1 validation accuracy:", np.mean((val1_pred >= 0.5) == y_val1))

    # plot results
    util.plot(x_train1_int, y_train1_int, Model1.theta, os.path.join(pred_path, "GDA_train1.jpeg") )
    util.plot(x_val1, y_val1, Model1.theta, os.path.join(pred_path, "GDA_val1.jpeg") )
    # save predictions
    np.savetxt(os.path.join(pred_path, "GDA_val1.txt"), val1_pred)

    # fit second dataset
    Model2 = GDA()
    Model2.fit(x_train2, y_train2)
    # prediction
    train2_pred = Model2.predict(x_train2_int)
    val2_pred = Model2.predict(x_val2)

    # results
    print("Dataset 2 training accuracy:", np.mean((train2_pred >= 0.5) == y_train2))
    print("Dataset 2 validation accuracy:", np.mean((val2_pred >= 0.5) == y_val2))

    # plot results
    util.plot(x_train2, y_train2, Model2.theta, os.path.join(pred_path, "GDA_train2.jpeg"))
    util.plot(x_val2, y_val2, Model2.theta, os.path.join(pred_path, "GDA_val2.jpeg"))
    # save predictions
    np.savetxt(os.path.join(pred_path, "GDA_val2.txt"), train2_pred)

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.
    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def sigmoid(self, x): 
        return 1 / (1 + np.exp( -x @ self.theta ))

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n+1)
        
        num_y1 = np.sum(y == 1)
        num_y0 = np.sum(y == 0)
        
        # get parameters from MLE derivations
        phi = num_y1 / m
        u1 = (y.T @ x) / num_y1
        u0 = np.sum(x[y==0], axis=0) / (m - num_y1)
        cov = (x[y==0] - u0).T @ (x[y==0]-u0) + (x[y==1]-u1).T @ (x[y==1]-u1)
        
        # get theta values
        self.theta[0] = 1/2 * (u1 + u0) @ np.linalg.inv(cov) @ (u0 - u1) - np.log(1/phi - 1)
        self.theta[1:] = np.linalg.inv(cov) @ (u1 - u0)
        
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
        # *** END CODE HERE

if __name__ == "__main__": 
    main()