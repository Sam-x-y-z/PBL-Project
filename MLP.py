# Json file format:  [{"video": "name", "clips": [{"segment": [i, f], "importance": 1mportance, "features": [512 feature]} * number of segments]} * number of videos]

import json
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

with open('output.json', 'r') as f:
    input_data = json.load(f)

preprocessed_data = []
for item in input_data:
    video_name = item["video"]
    clips = item["clips"]
    for clip in clips:
        segment = clip["segment"]
        importance = clip["importance"]
        features = clip["features"]
        preprocessed_data.append((video_name, segment, importance, features))
        
print("Number of videos: ", len(input_data))

train_data, test_data = train_test_split(preprocessed_data, test_size=0.2, shuffle=False)

print(f"Training set size: {len(train_data)}")
print(f"Testing set size: {len(test_data)}")


# MLP Training

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the hyperparameters
input_size = 512
hidden_size = 256
output_size = 1

# Create an instance of the MLP model
model = MLP(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Convert the train_data to tensors
train_features = torch.tensor([data[3] for data in train_data])
train_importance = torch.tensor([data[2] for data in train_data])
    
# Convert the test_data to tensors
test_features = torch.tensor([data[3] for data in test_data])
test_importance = torch.tensor([data[2] for data in test_data])

# Train the MLP model
for epoch in range(5000):
    # Forward pass
    outputs = model(train_features)
    loss = criterion(outputs.view(-1), train_importance)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss for every 10 epochs
    if (epoch+1) % 250 == 0:
        with torch.no_grad():
            test_outputs = model(test_features)
            test_loss = criterion(test_outputs.view(-1), test_importance)
        print(f'Epoch [{epoch+1}/{5000}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
        
# Save the model
torch.save(model.state_dict(), 'mlp_model.pth')

