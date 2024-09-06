import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import networkx as nx
# Load the dataset
df = pd.read_csv('/content/forestfires.csv')

# Encode categorical variables
df['month'] = df['month'].astype('category').cat.codes
df['day'] = df['day'].astype('category').cat.codes

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check the columns to ensure there are no leading/trailing spaces
print(df.columns)

# Ensure correct data types
df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
df['RH'] = pd.to_numeric(df['RH'], errors='coerce')
df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
df['rain'] = pd.to_numeric(df['rain'], errors='coerce')
df['month'] = pd.to_numeric(df['month'], errors='coerce')
df['day'] = pd.to_numeric(df['day'], errors='coerce')
df['area'] = pd.to_numeric(df['area'], errors='coerce')

# Calculate covariance matrix
cov_matrix = df[['temp', 'RH', 'wind', 'rain', 'month', 'day', 'area']].cov()

print("Covariance matrix:")
print(cov_matrix)





# Define a custom dataset class for the data
class ForestFireDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        features = torch.tensor([sample['temp'], sample['RH'], sample['wind'], sample['rain'], sample['month'], sample['day']], dtype=torch.float32)
        target = torch.tensor([sample['area']], dtype=torch.float32)
        return features, target
#  the dataset and data loaders
dataset = ForestFireDataset(df)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#  neural network model
class ForestFireModel(nn.Module):
    def __init__(self):
        super(ForestFireModel, self).__init__()
        self.fc1 = nn.Linear(6, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Initialize the models, loss functions, and optimizers
nn_model = ForestFireModel()
nn_criterion = nn.MSELoss()
nn_optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

svm_model = make_pipeline(
    StandardScaler(),
    SVR(kernel='rbf', C=1.0, epsilon=0.1)
)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
lr_model = LinearRegression()
# Train the models
for epoch in range(100):
    for features, target in data_loader:
        nn_optimizer.zero_grad()
        nn_output = nn_model(features)
        nn_loss = nn_criterion(nn_output, target)
        nn_loss.backward()
        nn_optimizer.step()
    print(f'Epoch {epoch+1}, NN Loss: {nn_loss.item()}')
svm_model.fit(dataset.df[['temp', 'RH', 'wind', 'rain', 'month', 'day']], dataset.df['area'])
rf_model.fit(dataset.df[['temp', 'RH', 'wind', 'rain', 'month', 'day']], dataset.df['area'])
lr_model.fit(dataset.df[['temp', 'RH', 'wind', 'rain', 'month', 'day']], dataset.df['area'])
# Evaluate the models
nn_model.eval()
total_nn_loss = 0
nn_predictions = []
with torch.no_grad():
    for features, target in data_loader:
        nn_output = nn_model(features)
        nn_loss = nn_criterion(nn_output, target)
        total_nn_loss += nn_loss.item()
        nn_predictions.extend(nn_output.cpu().numpy().flatten())
mean_nn_loss = total_nn_loss / len(data_loader)
print(f'Mean NN Loss: {mean_nn_loss:.6f}')

svm_predictions = svm_model.predict(dataset.df[['temp', 'RH', 'wind', 'rain', 'month', 'day']])
rf_predictions = rf_model.predict(dataset.df[['temp', 'RH', 'wind', 'rain', 'month', 'day']])
lr_predictions = lr_model.predict(dataset.df[['temp', 'RH', 'wind', 'rain', 'month', 'day']])




# Calculate the total loss of vegetation for each model
total_loss_vegetation_svm = np.sum(svm_predictions)
print(f'Total Loss of Vegetation (SVM): {total_loss_vegetation_svm:.4f} ha')

total_loss_vegetation_rf = np.sum(rf_predictions)
print(f'Total Loss of Vegetation (Random Forest): {total_loss_vegetation_rf:.4f} ha')

total_loss_vegetation_lr = np.sum(lr_predictions)
print(f'Total Loss of Vegetation (Linear Regression): {total_loss_vegetation_lr:.4f} ha')

# Mean accuracy and distance covered for neural network model
nn_predictions = np.clip(np.array(nn_predictions), 0, np.max(df['area']))
df['area'] = np.clip(df['area'], 0.001, np.max(df['area']))  # Clip area values to avoid division by zero
mean_accuracy = np.mean(np.abs(nn_predictions - df['area']) / (df['area'] + 1e-8))  # Add a small value to avoid division by zero
print(f'Mean Accuracy (Neural Network): {mean_accuracy:.4f}')

distance_covered = np.sum(nn_predictions) * 0.1  # Assuming 0.1 km per ha
print(f'Distance Going to Cover (Neural Network): {distance_covered:.4f} km')

# Calculate total burnt area by models used
total_burnt_by_models = total_loss_vegetation_svm + total_loss_vegetation_rf + total_loss_vegetation_lr + np.sum(nn_predictions)

# Calculate average burnt area per model
average_burnt_by_models = total_burnt_by_models / 4  # Divide by 4 for the number of models used

print(f'Total Burnt Area by Models: {total_burnt_by_models:.2f} ha')
print(f'Average Burnt Area by Models: {average_burnt_by_models:.2f} ha')
# Histogram of Prediction Errors
plt.figure(figsize=(8, 6))
plt.hist(np.abs(nn_predictions - df['area']) / df['area'], bins=50, alpha=0.5, label='Neural Network')
plt.xlabel('Relative Prediction Error')
plt.ylabel('Frequency')
plt.title('Histogram of Prediction Errors')
plt.legend()
plt.show()

# Scatter Plot of Actual vs. Predicted Burnt Area
plt.figure(figsize=(8, 6))
plt.scatter(df['area'], nn_predictions, label='Neural Network')
plt.xlabel('Actual Burnt Area (ha)')
plt.ylabel('Predicted Burnt Area (ha)')
plt.title('Actual vs. Predicted Burnt Area')
plt.legend()
plt.show()
# Feature Importance Plots
importance = rf_model.feature_importances_
plt.figure(figsize=(8, 6))
plt.bar(range(len(importance)), importance, align='center')
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.title('Feature Importance (Random Forest)')
plt.show()

# Pie Chart of Model Performance
plt.figure(figsize=(8, 6))
plt.pie([total_loss_vegetation_svm, total_loss_vegetation_rf, total_loss_vegetation_lr, mean_nn_loss], labels=['SVM', 'Random Forest', 'Linear Regression', 'Neural Network'], colors=['r', 'g', 'b', 'y'], autopct='%1.5f%%')
plt.title('Comparison of Model Performance')
plt.show()

# Comparison of Model Performance
models = ['SVM', 'Random Forest', 'Linear Regression', 'Neural Network']
losses = [total_loss_vegetation_svm, total_loss_vegetation_rf, total_loss_vegetation_lr, mean_nn_loss]
colors = ['red', 'lightgreen', 'blue', 'yellow']
plt.figure(figsize=(8, 6))
plt.bar(models, losses, color=colors, align='center')
plt.xlabel('Model')
plt.ylabel('Loss')
plt.title('Comparison of Model Performance')
plt.show()
# Line Chart
plt.figure(figsize=(8, 6))
plt.plot(nn_predictions, label='Neural Network')
plt.xlabel('Time')
plt.ylabel('Burnt Area (ha)')
plt.title('Burnt Area Over Time')
plt.legend()
plt.show()
