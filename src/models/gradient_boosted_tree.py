from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ----------------------------
# 1. Read all CSV files and label them by simulator
# ----------------------------
path = Path('data/treestats')
all_dfs = []

for config_folder in path.iterdir():  # iterate over each subfolder
    if config_folder.is_dir():
        for file in config_folder.glob('*.csv'):
            df = pd.read_csv(file)
            
            # Label simulator based on filename
            if 'msprime' in file.name:
                df['simulator'] = 'msprime'
            elif 'msms' in file.name:
                df['simulator'] = 'msms'
            elif 'ms_' in file.name or 'ms.csv' in file.name:
                df['simulator'] = 'ms'
            else:
                continue  # skip unexpected files
            
            all_dfs.append(df)

# Concatenate all dataframes
data = pd.concat(all_dfs, ignore_index=True)

# ----------------------------
# 2. Handle Inf and missing values
# ----------------------------
# Replace Inf/-Inf with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Find columns that contain any NaN
cols_with_nan = [col for col in data.columns if data[col].isna().any()]

print(f"Columns to be dropped ({len(cols_with_nan)}):")
for col in cols_with_nan:
    print(f"- {col}")

# Drop these columns
data.drop(columns=cols_with_nan, inplace=True)

print(f"Number of columns remaining: {data.shape[1]}")

# ----------------------------
# 3. Prepare features and target
# ----------------------------
dep_var = 'simulator'  # target variable
cont_names = list(data.columns)
cont_names.remove(dep_var)  # all other columns are features

# Encode target as integers
y = LabelEncoder().fit_transform(data[dep_var])
X = data[cont_names]

# ----------------------------
# 4. Split into train and validation sets
# ----------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2
)

# ----------------------------
# 5. Train Gradient Boosted Tree (XGBoost)
# ----------------------------
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    objective='multi:softmax',  # multi-class classification
    num_class=3,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# ----------------------------
# 6. Evaluate model
# ----------------------------
preds = model.predict(X_valid)
print("Validation Accuracy:", accuracy_score(y_valid, preds))


# Plot feature importance
xgb.plot_importance(model, max_num_features=20, importance_type='weight')  # count of splits
plt.show()

xgb.plot_importance(model, max_num_features=20, importance_type='gain')    # improvement in loss
plt.show()