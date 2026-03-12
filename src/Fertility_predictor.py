import pandas as pd

from src.Fertility_model import FertilityModel

class FertilityPredictor(FertilityModel):
    ''' Extends FertilityModel with input validation and patient prediction.

    Inherits all training, evaluation, and persistence functionality from
    FertilityModel and adds methods for validating patient input and
    producing human-readable prediction reports.
    '''

    def __init__(self):
       '''  Initialize the FertilityPredictor with default untrained state.

        Calls the parent FertilityModel initializer and sets up a container
        to store the last prediction result.
       '''
       super().__init__()
       self._last_result: dict | None = None

    @property
    def last_result(self):
        ''' Return the last prediction result produced by this predictor.

        Returns:
            dict | None: The last result dictionary if a prediction has been
                made, otherwise None.
        '''
        return self._last_result

    # Provides the summary of model performance
    def __str__(self):
        ''' Return a human-readable summary of the predictor's status and performance.

        If the model is not trained, returns a short placeholder message.
        If trained, includes accuracy, number of features, and the last
        prediction result if one is available.

        Returns:
            str: A summary string describing the predictor's current state.
        '''
        if not self._is_trained:
            return "Predictor is not trained yet"
        result = (
            f"PatientPredictor\n"
            f"  Accuracy : {self._accuracy * 100:.2f}%\n"
            f"  Features : {len(self._original_features)}")
        if self._last_result:
            result += f"\n  Last prediction : {self._last_result['prediction']} ({self._last_result['confidence']:.1f}%)"

        return result

    @staticmethod
    def _display_result(result: dict):
        ''' Print a formatted prediction report to the console.

        Displays the predicted class, confidence percentage, and a visual
        probability bar chart for each class. Also prints a disclaimer
        reminding the user this is not a medical diagnosis.

        Args:
            result: A result dictionary containing:
                - prediction (str): Predicted class label.
                - confidence (float): Confidence percentage.
                - probabilities (dict | None): Mapping of class label to
                  probability percentage.
        '''
        print(f"\n{'*' * 70}")
        print("PREDICTION RESULTS")
        print(f"{'*' * 70}")
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']:.2f}%")
        if result['probabilities']:
            print("\n  Probability breakdown:")
            for cls, prob in result['probabilities'].items():
                bar = "=" * int(prob / 2)
                print(f"    {cls:20s} {bar} {prob:.1f}%")
        print(f"\n{'*' * 70}")
        print("!!  This is a prediction, not a medical diagnosis  !!")
        print("Please consult a qualified healthcare professional.")
        print(f"{'*' * 70}\n")

    def predict_patient(self, patient_data: dict) -> dict:
        ''' Validate patient input and return a decoded prediction result.

        Checks that all required features are present and numeric, then
        runs the model and returns the prediction with confidence and
        per-class probabilities.

        Args:
            patient_data: Mapping of feature name to value. Must contain
                exactly the features in self._original_features.

        Returns:
            A result dictionary containing:
                - prediction (str): Decoded predicted class label.
                - confidence (float): Confidence percentage.
                - probabilities (dict): Mapping of class label to probability
                  percentage.
                - patient_data (dict): The validated and converted input data.

        Raises:
            RuntimeError: If the model has not been trained yet.
            ValueError: If required features are missing or any feature value
                cannot be converted to float.
        '''
        # make a prediction using patient's data
        if not self._is_trained:
            raise RuntimeError("Predictor is not trained yet")
        #Check that all features are present
        if set(self._original_features) != set(patient_data.keys()):
            raise ValueError(f"Missing features: {set(self._original_features) - set(patient_data.keys())}")
        #Check that the data is numeric
        for feature in self._original_features:
            try:
                patient_data[feature] = float(patient_data[feature])
            except (ValueError, TypeError):
                raise ValueError(f"Feature '{feature}' must be numeric, got: {patient_data[feature]!r}")


        user_df = pd.DataFrame([patient_data])
        raw_pred, probs = self.predict_encoded(user_df)
        predicted_class, confidence, prob_dict = self.decode_prediction(raw_pred, probs)

        result = {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "patient_data": patient_data
        }

        self._last_result = result
        return result

    def interactive_prediction(self):
        ''' Run an interactive command-line session to collect patient data and display a prediction.

        Prompts the user to enter 0 or 1 for each binary feature, and a
        numeric value for Age. Validates each input before proceeding,
        then calls predict_patient and displays the formatted result.

        Raises:
            RuntimeError: If the model has not been trained yet.
        '''
        print("Patient Data")
        print("*" * 70)
        print("Enter 1 (yes) or 0 (no) for each question, except for age")
        patient_data = {}
        
        for feature in self._original_features:
            while True:
                value = input(f"{feature}: ")
                if feature == 'Age':
                    try:
                        patient_data[feature] = float(value)
                        break
                    except ValueError:
                        print("Please enter a valid number for age")
                else:
                    if value in ('0', '1'):
                        patient_data[feature] = float(value)
                        break
                    else:
                        print("Please enter 0 or 1")

        print(f"\nProcessing patient data")
        result = self.predict_patient(patient_data)
        self._last_result = result
        self._display_result(result)



