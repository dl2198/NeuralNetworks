import pandas as pd
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset

from statistics import mean
import matplotlib.pyplot as plt


from NeuralNet import NeuralNet


def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities

        self.scaler = None
        self.trained_model = None
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.base_classifier = None  # ADD YOUR BASE CLASSIFIER HERE

    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD

    def _preprocessor(self, X_raw, train=False):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # # PART2 FORM
        # normalised_X_raw = preprocessing.MinMaxScaler().fit_transform(X_raw)
        # return normalised_X_raw

        features = X_raw

        pol_coverage_categories = ["Maxi", "Median1", "Median2", "Mini"]
        pol_pay_frequency_categories = [
            "Biannual", "Monthly", "Quarterly", "Yearly"]
        pol_usage_categories = ["AllTrips",
                                "Professional", "Retired", "WorkPrivate"]
        vh_fuel_categories = ["Diesel", "Gasoline", "Hybrid"]

        if train is True:
            # Create and store one hot encoding binarizers
            self.pol_coverage_binarizer = preprocessing.LabelBinarizer().fit(
                pol_coverage_categories)
            self.pol_pay_frequency_binarizer = preprocessing.LabelBinarizer().fit(
                pol_pay_frequency_categories)
            self.pol_usage_binarizer = preprocessing.LabelBinarizer().fit(
                pol_pay_frequency_categories)
            self.vh_fuel_binarizer = preprocessing.LabelBinarizer().fit(
                vh_fuel_categories)

            # Create and store binary binarizers
            self.pol_payd_binarizer = preprocessing.LabelBinarizer().fit(features.drv_drv2)
            self.drv_drv2_binarizer = preprocessing.LabelBinarizer().fit(features.pol_payd)
            self.drv_sex1_binarizer = preprocessing.LabelBinarizer().fit(features.drv_sex1)
            self.vh_type_binarizer = preprocessing.LabelBinarizer().fit(features.vh_type)

        # PART 3 Preprocessing
        temp = pd.DataFrame(
            self.pol_coverage_binarizer.transform(features.pol_coverage))
        for i, category in enumerate(pol_coverage_categories):
            features[category] = temp[i]

        temp = pd.DataFrame(
            self.pol_pay_frequency_binarizer.transform(features.pol_pay_freq))
        for i, category in enumerate(pol_pay_frequency_categories):
            features[category] = temp[i]

        features.pol_payd = self.pol_payd_binarizer.transform(
            features.pol_payd)

        temp = pd.DataFrame(
            self.pol_usage_binarizer.transform(features.pol_usage))
        for i, category in enumerate(pol_usage_categories):
            features[category] = temp[i]

        features.drv_drv2 = self.drv_drv2_binarizer.transform(
            features.drv_drv2)
        features.drv_sex1 = self.drv_sex1_binarizer.transform(
            features.drv_sex1)

        temp = pd.DataFrame(
            self.vh_fuel_binarizer.transform(features.vh_fuel))
        for i, category in enumerate(vh_fuel_categories):
            features[category] = temp[i]

        features.vh_type = self.vh_type_binarizer.transform(features.vh_type)
        features = features.drop(
            columns=['id_policy', 'pol_bonus', 'pol_sit_duration', 'pol_insee_code'], axis=1)
        features = features.drop(columns=['pol_coverage'], axis=1)
        features = features.drop(columns=['pol_pay_freq'], axis=1)
        features = features.drop(columns=['pol_usage'], axis=1)
        features = features.drop(columns=["drv_age2"])
        features = features.drop(columns=["drv_sex2"])
        features = features.drop(columns=["drv_age_lic1", "drv_age_lic2"])
        features = features.drop(columns=['vh_fuel', 'vh_model', 'vh_make'])
        features = features.drop(columns=["town_mean_altitude", "town_surface_area", "population",
                                          "commune_code", "canton_code", "city_district_code", "regional_department_code"])
        normalised_features = preprocessing.MinMaxScaler().fit_transform(features)
        return normalised_features

    def fit(self, X_raw, y_raw, claims_raw, weighting=9, learning_rate=0.001, batch_size=20, num_epochs=30, hidden_size=50):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])  # Average of all claims
        # =============================================================

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        # if self.calibrate:
        #     self.base_classifier = fit_and_calibrate_classifier(
        #         self.base_classifier, X_clean, y_raw)
        # else:
        #     self.base_classifier = self.base_classifier.fit(X_clean, y_raw)
        # return self.base_classifier

        # Reshuffle data, we can't trust input if main() was used (e.g. LabTS?)
        # state = np.random.get_state()
        # X_raw = X_raw.sample(frac=1).reset_index(drop=True)
        # np.random.set_state(state)
        # y_raw = y_raw.sample(frac=1).reset_index(drop=True)

        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw, train=True)

        # Split train and validation
        split_index = int(X_clean.shape[0] * 0.8)
        train_set = X_clean[:split_index]
        train_labels = y_raw[:split_index]
        val_set = X_clean[split_index:]
        val_labels = y_raw[split_index:]

        # # Ensure train_labels is correct
        # print(type(train_labels))
        # print(train_labels.value_counts())

        # Training dataset
        train_ds = TensorDataset(torch.tensor(
            train_set), torch.tensor(train_labels))
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True)

        # Validations dataset. Tune your hyperparams with this.
        validation_dataset = TensorDataset(torch.tensor(val_set),
                                           torch.tensor(val_labels.values))
        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False)

        # Input and Output
        num_inputs = train_set.shape[1]
        output_size = 1

        # Create a model with hyperparameters
        model = NeuralNet(num_inputs, hidden_size, output_size)

        # Weight positive samples higher
        pos_weight = torch.ones([1])
        pos_weight.fill_(weighting)

        # Loss criterion and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        epochs_list = []
        training_loss = []

        for epoch in range(num_epochs):
            batch_loss = []
            accuracy = []
            batch_accuracy = []
            for xb, yb in train_loader:
                # Forwards pass
                y = model(xb.float())
                loss = criterion(y.flatten(), yb.float())
                batch_loss.append(loss.item())
                y_classified = (torch.sigmoid(y).round()).view(-1)
                batch_accuracy.append(accuracy_score(
                    yb, y_classified.detach().numpy()))
                # print(loss)

                # Backward and optimize
                loss.backward()
                # Arbitrary value as per stack overflow
                # Prevent loss 'blowing up' https://stackoverflow.com/questions/33962226/common-causes-of-nans-during-training
                # clipping_value = 1
                # nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
                optimizer.step()
                optimizer.zero_grad()

            print(
                f"Epoch: {epoch}, Batch loss: {mean(batch_loss)}, Accuracy: {mean(batch_accuracy)}")

            epochs_list.append(epoch)
            training_loss.append(mean(batch_loss))
            accuracy.append(mean(batch_accuracy))

        #plt.plot(epochs_list, training_loss, 'g', label='Training loss')
        #plt.title('Training loss')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        #plt.legend()
        #plt.show()

        self.trained_model = model

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)

        # Predict
        #all_outputs = []
        #all_labels = []
        #all_raw = []

        x_test = torch.tensor(X_clean)

        with torch.no_grad():
            outputs = self.trained_model(x_test.float())
            # Convert the outputs to probabilities
            predictions = torch.sigmoid(outputs)
            # print(predictions.flatten().numpy())

        # return probabilities for the positive class (label 1)
        return predictions.flatten().numpy()

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        # print("Predict premium", type(X_raw), X_raw.shape)
        #print(f'premium: {premium}')
        #return premium
        return self.predict_claim_probability(X_raw) * self.y_mean * 0.3
        

    #TODO - delete
    def evaluate_architecture(self, probabilities, labels):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        predictions = probabilities.round()

        target_names = ['not claimed', 'claimed']
        print(f'labels {labels} of size {labels.shape}')
        print(f'predictions {predictions} of size {predictions.shape}')
        print(f'probabilities {probabilities} of size {probabilities.shape}')

        print(classification_report(labels.astype(int), predictions.astype(int)))

        print(confusion_matrix(labels, predictions))
        auc_score = roc_auc_score(labels, probabilities)
        print(f'auc: {auc_score}')
        print("Accuracy: ", accuracy_score(labels, predictions))
        #print("Labels: ", labels)
        #print("Predictions: ", predictions)
        return auc_score

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        # File name was changed, according to Marek's Piazza
        with open('part3_pricing_model_linear.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model_linear.pickle', 'rb') as target:
        return pickle.load(target)


def main():
    # ClaimClassifierHyperParameterSearch()

    # Load pandas dataframe from csv & shuffle
    df1 = pd.read_csv('part3_training_data.csv')
    df1 = df1.sample(frac=1).reset_index(drop=True)
    split_index = int(df1.shape[0] * 0.8)

    # Split train & test. Final evaluation only on test_set. NOT hyperparameter searching
    train_data = df1.iloc[:split_index]
    train_set = train_data.drop(columns=["made_claim", "claim_amount"])
    train_labels = train_data["made_claim"]
    train_claims_raw = train_data["claim_amount"]

    test_data = df1.iloc[split_index:]
    test_data.reset_index(drop=True, inplace=True)
    test_set = test_data.drop(columns=["made_claim", "claim_amount"])
    test_labels = test_data["made_claim"]
    # test_claims_raw = test_data["claim_amount"]

    # # In the form of part 2 (need to change _preprocessor to part2 form too). Tested to get auc: 0.6320862281258937
    # train_set = df1.filter([
    #     "drv_age1",
    #     "vh_age",
    #     "vh_cyl",
    #     "vh_din",
    #     "pol_bonus",
    #     "vh_sale_begin",
    #     "vh_sale_end",
    #     "vh_value",
    #     "vh_speed",
    # ])
    # train_labels = df1["made_claim"]
    # claims_raw = df1["claim_amount"]

    # pricingModel = PricingModel()
    # pricingModel.fit(train_set, train_labels, claims_raw)
    # pricingModel.save_model()

    # probabilities = pricingModel.predict_claim_probability(train_set)
    # predictions = pricingModel.predict_premium(train_set)
    # pricingModel.evaluate_architecture(probabilities, train_labels.to_numpy())
    # # end if the form of part 2

    pricingModel = PricingModel()
    pricingModel.fit(train_set, train_labels, train_claims_raw)
    pricingModel.save_model()

    predictions = pricingModel.predict_premium(test_set)
    print("Main predictions")
    print(predictions)

    probabilities = pricingModel.predict_claim_probability(test_set)
    print("Main probabilities")
    print(probabilities)

    pricingModel.evaluate_architecture(probabilities, test_labels.to_numpy())


if __name__ == "__main__":
    main()
