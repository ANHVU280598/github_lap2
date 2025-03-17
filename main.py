import json
from ml_code.data_load import load_data
from ml_code.pre_proccessing import preprocess_data
from ml_code.models import ModelFactory
from ml_code.train import train_and_evaluate
from ml_code.metrics import print_metrics
from test.unit_test import TestDataLoader

with open('config.json') as config_file:
    config = json.load(config_file)

# Step 1: Load data
data = load_data(r'C:\Users\anhvu\OneDrive\Desktop\github_lap2\data\data.csv')
print(data.head())

# Step 2: Preproccess data
X_train, X_test, y_train, y_test =  preprocess_data(data)

# Step 3: Get the Model from ModelFactory
model = ModelFactory.get_model(config["model_type"])

# Step 4: Train and Evaluate the Model 
accuracy, cm, y_test, y_prob = train_and_evaluate(model, X_train, X_test, y_train, y_test)

# Step 5: Print metrics
print_metrics(accuracy, cm, y_test, y_prob)
