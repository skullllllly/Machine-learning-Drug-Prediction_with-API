import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv(r"D:\API\API_WITH_ML\Machine-learning-Drug-Prediction_with-API\drug200.csv")
print(data.info())
print(data.head(10))


X = data.drop(["Drug"], axis=1)



le = LabelEncoder()
y = le.fit_transform(data["Drug"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Pipeline Architecture
categorical_features = ["Sex", "BP", "Cholesterol"]
numerical_features = ["Age", "Na_to_K"]



# Define preprocessing transformations
preprocessor = ColumnTransformer(
    
    transformers=[
        ('cat', OrdinalEncoder(), categorical_features)
    ],
    remainder='passthrough' # Leaves numerical features untouched
)



# Bundle preprocessing and model into a single pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeClassifier(max_depth=2, min_samples_leaf=20))
])


pipeline.fit(X_train, y_train)

#Evaluation Phase
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"Training Score: {train_score}")
print(f"Testing Score: {test_score}")
print(f"Generalization Gap: {train_score - test_score}")



plt.figure(figsize=(12, 8))
plot_tree(
    pipeline.named_steps['regressor'], 
    filled=True, 
    feature_names=categorical_features + numerical_features, 
    class_names=list(le.classes_)
)
plt.show()

#-----------------------------------------------------------------------------------
#  Production Inference (Testing New Information Safely)


test_info = {
    "Age": [13], 
    "Sex": ["M"], 
    "BP": ["NORMAL"], 
    "Cholesterol": ["HIGH"], 
    "Na_to_K": [6.07]
}
new_info = pd.DataFrame(test_info)

# The pipeline automatically applies the fitted OrdinalEncoder transformations to new_info
new_pred_encoded = pipeline.predict(new_info)

# Decode the prediction back to the original drug label
new_pred_label = le.inverse_transform(new_pred_encoded.astype(int))
print(f"Predicted Drug: {new_pred_label[0]}")


"""
print("----------------------------------------------------- \n")

while(True):
    result=input(" to enter ur information  press  any key : or press q to quite: \n ").lower()
    
    
    if result=='q':
        print("exit succesfully")
        break
    else:
        age=int(input("your age:"))
        sex=input("your sex:")
        bp=input("your BP:")
        cholesterol=input("your cholestrol:")
        na=input("your na_to_k:")
        na=float(na)
      
        
        new_test_info={
            "Age": [age], 
            "Sex": [sex], 
            "BP": [bp], 
            "Cholesterol": [cholesterol], 
            "Na_to_K":[na]
        }
        new_info = pd.DataFrame(new_test_info)

        
        new_pred_encoded = pipeline.predict(new_info)

        new_pred_label = le.inverse_transform(new_pred_encoded.astype(int))
        print(f"Predicted Drug: {new_pred_label[0]}")
        
"""


#-------------------------------------------
import joblib

# Save the complete pipeline (preprocessor + decision tree)
joblib.dump(pipeline , 'drug_pipeline_model.pkl')

# Save the label encoder so you can decode predictions back to drug names
joblib.dump(le, 'label_encoder.pkl')

print("Model artifacts successfully exported.")
#-------------------------------------------











