from PimaPrediction import fill_0, feature_col_names, lr_model
import numpy as np

# Manually input patient data in a dictionary
patient_data = {
    'num_preg': 3,
    'glucose_conc': 100,
    'diastolic_bp': 70,
    'thickness': 25,
    'insulin': 0,  # Replace with the actual value if available
    'bmi': 30,
    'diab_pred': 0.4,
    'age': 35
}

# Convert the patient data into a numpy array and impute missing values (if any)
patient_features = np.array([[patient_data[feature] for feature in feature_col_names]])
patient_features = fill_0.transform(patient_features)

# Make the prediction using the logistic regression model
prediction = lr_model.predict(patient_features)

# The prediction will be an array with 0 or 1 indicating 'False' or 'True' for diabetes
if prediction[0] == 0:
    print("The patient is predicted to NOT have diabetes.")
else:
    print("The patient is predicted to HAVE diabetes.")
