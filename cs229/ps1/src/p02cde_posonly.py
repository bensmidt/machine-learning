import numpy as np
import util
import os

from p01b_logreg import LogisticRegression

def main():
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    #################################################################################
    # PART C: USING TRUE LABELS
    # Data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    # data set 1 paths
    train_path = os.path.join(data_dir, 'ds3_train.csv')
    valid_path = os.path.join(data_dir, 'ds3_valid.csv')
    test_path = os.path.join(data_dir, 'ds3_test.csv')
    # prediction path
    pred_path = os.path.join(os.path.dirname(__file__), 'output', 'P2')

    # Load train and test data with tr labels
    x_train, t_train = util.load_dataset(train_path, label_col = 't', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col = 't', add_intercept=True)

    # Fit the Model and predict
    Model_t = LogisticRegression(eps=1e-5)
    Model_t.fit(x_train, t_train)
    train_t_pred = Model_t.predict(x_train)
    t_pred = Model_t.predict(x_test)

    # Accuracy
    print("Train set accuracy using true labels:", np.mean((train_t_pred >= 0.5) == t_train) )
    print("Test set accuracy using true labels:", np.mean((t_pred >= 0.5) == t_test) )

    # Plot and save data
    util.plot(x_train, t_train, Model_t.theta, os.path.join(pred_path, 't_labels_train.jpeg'))
    util.plot(x_test, t_test, Model_t.theta, os.path.join(pred_path, 't_labels_test.jpeg'))
    np.savetxt(os.path.join(pred_path, 't_labels_test.txt'), t_pred)

    #################################################################################
    # PART D: USING Y LABELS

    # Load train and test data with y labels
    x_train, y_train = util.load_dataset(train_path, label_col = 'y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col = 'y', add_intercept=True)

    # Fit the model and predict
    Model_y = LogisticRegression(eps=1e-5)
    Model_y.fit(x_train, y_train)
    train_y_pred = Model_y.predict(x_train)
    y_pred = Model_y.predict(x_test)

    # Accuracy
    print("Train set accuracy using positive only labels:", np.mean((train_y_pred >= 0.5) == t_train) )
    print("Test set accuracy using positive only labels:", np.mean((y_pred >= 0.5) == t_test) )

    # Plot and save data
    util.plot(x_train, y_train, Model_y.theta, os.path.join(pred_path, 'y_labels_train2.jpeg') )
    util.plot(x_test, y_test, Model_y.theta, os.path.join(pred_path, 'y_labels_test2.jpeg') )
    np.savetxt(os.path.join(pred_path, 'y_labels_test.txt'), y_pred)

    #################################################################################
    # PART E: ESTIMATING ALPHA W/ VALIDATION SET

    # Load validation data
    x_valid, y_valid = util.load_dataset(valid_path, label_col = 'y', add_intercept=True)

    # Estimate alpha
    y_pred_pos = Model_y.predict(x_valid[y_valid==1])
    alpha = np.mean(y_pred_pos)
    theta_corrected = Model_y.theta.copy()
    theta_corrected[0] = theta_corrected[0] - np.log(2/alpha - 1) 

    # Accuracy
    print("Train set accuracy using positive only labels with alpha correction:", np.mean((train_t_pred/alpha >= 0.5) == t_train) )
    print("Test set accuracy using positive only labels with alpha correction:", np.mean((t_pred/alpha >= 0.5) == t_test) )

    util.plot(x_test, t_test, theta_corrected, os.path.join(pred_path, 'alpha_test3.jpeg'))

if __name__ == "__main__": 
    main()