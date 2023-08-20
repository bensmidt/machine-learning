import matplotlib.pyplot as plt
import numpy as np
import util
import os

from cmath import inf

from p05b_lwr import LocallyWeightedLinearRegression as LWLR


def main():
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # get data and output paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_path = os.path.join(data_dir, 'ds5_train.csv')
    valid_path = os.path.join(data_dir, 'ds5_valid.csv')
    test_path = os.path.join(data_dir, 'ds5_test.csv')
    pred_path = os.path.join(os.path.dirname(__file__), 'output', 'P5')

    # Load training, validation, and test sets
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # Search tau_values for the best tau (lowest MSE on the validation set)
    taus = [1, 0.8, 0.5, 0.2, 0.15, 0.1, 0.08, 0.05, 0.04, 0.03, 0.02]
    mse_best = inf
    tau_best = 1
    for tau in taus: 
        Model = LWLR(tau)
        Model.fit(x_train, y_train)

        predictions = Model.predict(x_valid)
        mse = np.mean((predictions - y_valid)**2)

        if mse < mse_best: 
            mse_best = mse
            tau_best = tau

        plt.figure()
        plt.plot(x_valid, y_valid, 'bx')
        plt.plot(x_valid, predictions, 'go')
        plt.savefig(os.path.join(pred_path, 'c_tau={}.jpeg'.format(tau))) 

    print("Best tau on validation set =", tau_best, "with MSE of", mse_best)

    # get test set results with best tau 
    Model = LWLR(tau_best)
    Model.fit(x_train, y_train)
    test_pred = Model.predict(x_test)

    # MSE
    mse = np.mean((test_pred - y_test)**2)
    print("Test set MSE with a tau of {} =".format(tau_best), mse)

    # plot results
    plt.figure()
    plt.plot(x_test, y_test, 'bx')
    plt.plot(x_test, test_pred, 'go')
    plt.savefig(os.path.join(pred_path, 'c_test_tau={}.jpeg'.format(tau_best)))

if __name__ == "__main__": 
    main()