import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
import torch
import csv
from Metrics import spearman_correlation, KendallTau, PearsonCorrelation, RMSE
from ReadData import readJsonData
from LSTM_MLP_Model import LSTM_MLP_Model
from GetDataTensors import getDataTensors
from LossFunction import CustomLoss
from Parameters import training_show_every as show_every, training_epochs as epochs
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

csv_fields = ['epoch', 'train_loss', 'test_loss', 'person', 'person_p', 'spearman', 'spearman_p', 'rmse']
csv_file_name = 'LSTM-MLP-Results.csv'

csv_data_points_fields = ['real', 'predicted']
csv_data_points_file_name = 'LSTM-MLP-Data-Points.csv'

preprocessed_data, number_of_videos = readJsonData('./data.json')
        
print("Number of videos: ", number_of_videos)

features_tensor = torch.tensor([item[3] for item in preprocessed_data])
importance_tensor = torch.tensor([item[2] for item in preprocessed_data]).unsqueeze(1)

prev_loss = 1000000

with open(csv_file_name, mode='w', newline='') as csv_file, open(csv_data_points_file_name, mode='w', newline='') as csv_data_points_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_fields)
    csv_data_points_writer = csv.writer(csv_data_points_file)
    csv_data_points_writer.writerow(csv_data_points_fields)
    for lmbda in [1.15 + (0.05 * i) for i in range(1)]:
        csv_writer.writerow([f'Lambda: {lmbda}'])
        print(f'Lambda: {lmbda}')
        
        model = LSTM_MLP_Model()

        # Loss and optimizer
        criterion = CustomLoss(lmbda)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Get the data tensors
        features_training_tensor, features_testing_tensor, importance_training_tensor, importance_testing_tensor = getDataTensors(features_tensor, importance_tensor)

        # Train the model (dummy example, replace with your actual data)
        features_training_tensor.requires_grad_()
        
        for epoch in range(1, epochs+1):
            outputs = model(features_training_tensor)
            targets = importance_training_tensor  # Targets are the same as input for reconstruction
            loss = criterion(outputs.view(-1,1), targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                    test_outputs = model(features_testing_tensor)
                    test_loss = criterion(test_outputs.view(-1,1), importance_testing_tensor)
            
            pearson, pearson_p = PearsonCorrelation(test_outputs.view(-1), importance_testing_tensor.view(-1))
            spearman, spearman_p = spearman_correlation(test_outputs.view(-1), importance_testing_tensor.view(-1))
            rmse = RMSE(test_outputs.view(-1), importance_testing_tensor.view(-1))
        
            csv_writer.writerow([epoch, loss.item(), test_loss.item(), pearson, pearson_p, spearman, spearman_p, rmse.item()])
            
            if epoch % show_every == 0 or epoch == 1:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Pearson: {pearson:.4f}, Pearson p: {pearson_p:.4f}, Spearman: {spearman:.4f}, Spearman p: {spearman_p:.4f}, RMSE: {rmse:.4f}')
                if test_loss.item() > prev_loss:
                    print("Overfitting...")
                    break
                else:
                    prev_loss = test_loss.item()
        
        data_points = list(zip(importance_testing_tensor.view(-1).tolist(), test_outputs.view(-1).tolist()))
        csv_data_points_writer.writerow(f'Lambda: {lmbda}')
        csv_data_points_writer.writerows(data_points)
        # Save the model
        torch.save(model.state_dict(), f'LSTM-MLP-Lambda-{lmbda}.pth')

