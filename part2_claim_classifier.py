import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset

from NeuralNet import NeuralNet
from statistics import mean # This can be imported, its python
import matplotlib.pyplot as plt

# METRICS
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


class ClaimClassifier():

    def __init__(self):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self.trained_model = None
        self.scaler = None


    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray (Pandas DataFrame)
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """
        if self.scaler == None:
            self.scaler = preprocessing.MinMaxScaler()
        # Normalisation (also saved for testing data later)
        return self.scaler.fit_transform(X_raw)

    def fit(self, X_raw, y_raw, weighting=1, learning_rate=0.001, batch_size=20, num_epochs=20, hidden_size=50):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray (pandas DataFrame)
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """
        # Shuffle data
        state = np.random.get_state()
        X_raw = X_raw.sample(frac=1).reset_index(drop=True)
        np.random.set_state(state)
        y_raw = y_raw.sample(frac=1).reset_index(drop=True)

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)

        # Split data
        percentile_60 = int(X_clean.shape[0] * 0.6)
        percentile_80 = int(X_clean.shape[0] * 0.8)

        train_data = X_clean[:percentile_60]
        train_labels = y_raw[:percentile_60]

        test_data = X_clean[percentile_60:percentile_80]
        test_labels = y_raw[percentile_60:percentile_80]
        self.test_data = test_data
        self.test_labels = test_labels

        val_data = X_clean[percentile_80:]
        val_labels = y_raw[percentile_80:]

        # Convert from numpy to tensors for train data and corresponding labels
        # NB X_clean is already a numpy array
        x_train = torch.tensor(train_data)
        y_train = torch.tensor(train_labels)

        x_test = torch.tensor(test_data)
        y_test = torch.tensor(test_labels.values)

        x_val = torch.tensor(val_data)
        y_val = torch.tensor(val_labels.values)

        # Training dataset
        train_ds = TensorDataset(x_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Testing dataset
        test_ds = TensorDataset(x_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Validations dataset
        val_ds = TensorDataset(x_val, y_val)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Input and Output
        num_inputs = train_data.shape[1]
        output_size = 1

        # Create a model with hyperparameters
        model = NeuralNet(num_inputs, hidden_size, output_size)

        # Weight positive samples higher
        pos_weight = torch.ones([1])
        pos_weight.fill_(weighting)

        # Loss criterion and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # TODO - to delete
        epochs_list = []
        training_loss = []
        batch_loss = []

        for epoch in range(num_epochs):
            for xb, yb in train_dl:

                # Forwards pass
                preds = model(xb.float())  # Why do I need to add float() here?
                loss = criterion(preds.flatten(), yb.float())

                # TODO - delete this: For calculating the average loss and accuracy
                batch_loss.append(loss.item())
                #batch_accuracy.append(model.accuracy(preds, yb, batch_size))

                # Backward and optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epochs_list.append(epoch)
            # print(epoch)
            training_loss.append(mean(batch_loss))
            # accuracy_list.append(mean(batch_accuracy))

        # plt.plot(epochs_list, accuracy_for_one, 'b', label='Accuracy for 1')

        # plt.plot(epochs_list, training_loss, 'g', label='Training loss')
        # plt.title('Training loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()

        self.trained_model = model

        return model  # TODO - not correct thing to do?

    def get_test_data(self):
      return [self.test_data, self.test_labels]

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of belonging to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)

        # Predict
        #all_outputs = []
        #all_labels = []
        #all_raw = []

        x_test = torch.tensor(X_clean)

        with torch.no_grad():
            outputs = self.trained_model(x_test.float())
            # Convert the outputs to probabilities
            predictions = F.sigmoid(outputs)
            print("prediction shape: ", predictions.shape)
            # print(predictions.flatten().numpy())

        return predictions.flatten().numpy()  # PREDICTED CLASS LABELS (as probabilites)

    def evaluate_architecture(self, probabilities, labels):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """

        # Need to convert the probabilities into binary classification
        sigmoid = nn.Sigmoid()
        predictions = probabilities.round()
        print(predictions)

        target_names = ['not claimed', 'claimed']

        print(classification_report(labels, predictions, target_names=target_names))

        print(confusion_matrix(labels, predictions))
        auc_score = roc_auc_score(labels, probabilities)
        print(f'auc: {auc_score}')
        print("Accuracy: ", accuracy_score(labels, predictions))
        # print("Labels: ", labels)
        # print("Predictions: ", predictions)
        return auc_score

    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        return pickle.load(target)

# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION


def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """
    # try layer count (net depth)
    # try different optimizer
    # try #neurons per layer

    # List of hyperparameters
    # params = {
    #     'weighting': list(range(2, 10)),
    #     'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
    #     'hidden_size': list(range(5, 500)),
    #     'batch_size': list(range(10, 100)),
    #     'num_epochs': list(range(30, 200)),
    # }

    # Hyperparameters
    learn_rate = 0.001
    batch_size = 20

    df1 = pd.read_csv('part2_training_data.csv')
    X = df1.drop(columns=["claim_amount", "made_claim"])
    y = df1["made_claim"]

    # weighting_attempts = [0.5,2,4,8,8.5,9,9.5,10,10.5,11,12,14,15,20,50,100]
    weighting_attempts = [9]
    sampling_size = 5
    weighting_auc_scores = [[None]*sampling_size for i in weighting_attempts]

    claimClassifier = ClaimClassifier()

    for i, weighting in enumerate(weighting_attempts):
        for take in range(sampling_size):
            print("Weighting: ", weighting, ", sample: ", sampling_size)
            model = claimClassifier.fit(X, y, weighting=weighting)
            [test_data, test_labels] = claimClassifier.get_test_data()
            # print("Test data: ",test_data)
            probabilities = claimClassifier.predict(pd.DataFrame(test_data))
            print("probabilities: ",probabilities)
            auc_score = claimClassifier.evaluate_architecture(probabilities, test_labels)
            weighting_auc_scores[i][take] = (auc_score)

    print(weighting_auc_scores)
    auc_scores_per_weight = list(map(lambda x: sum(x)/len(x), weighting_auc_scores))
    plt.plot(weighting_attempts, auc_scores_per_weight)

    # find the highest weighting attempt, and set all future weighting to this.

    return  # Return the chosen hyper parameters


# We found the optimal hyperparameters using ClaimClassifierHyperParameterSearch
# This is to avoid re-running the code again.
def test_save_and_load():
    df1 = pd.read_csv('part2_training_data.csv')
    X = df1.drop(columns=["claim_amount", "made_claim"])
    y = df1["made_claim"]
  
    # train here
    claimClassifier = ClaimClassifier()
    weighting = 9
    claimClassifier.fit(X, y, weighting=weighting)
    claimClassifier.save_model()

    # tested here
    claimClassifier = load_model()
    [test_data, test_labels] = claimClassifier.get_test_data()
    probabilities = claimClassifier.predict(pd.DataFrame(test_data))
    claimClassifier.evaluate_architecture(probabilities, test_labels)

def main():
    ClaimClassifierHyperParameterSearch()
    # test_save_and_load()

if __name__ == "__main__":
    main()
