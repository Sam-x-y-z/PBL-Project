from Parameters import lstm_input_size as input_size, lstm_sequence_length as sequence_length
import torch
from sklearn.model_selection import train_test_split

def getDataTensors(features_tensor, importance_tensor):
    features_training_tensor, features_testing_tensor, importance_training_tensor, importance_testing_tensor = train_test_split(features_tensor, importance_tensor, test_size=0.2, shuffle=False)

    # padding tensors to make their length a multiple of sequence_length
    features_training_tensor = torch.cat((features_training_tensor, torch.zeros(sequence_length - features_training_tensor.size(0) % sequence_length, input_size)))
    importance_training_tensor = torch.cat((importance_training_tensor, torch.zeros(sequence_length - importance_training_tensor.size(0) % sequence_length, 1)))
    features_testing_tensor = torch.cat((features_testing_tensor, torch.zeros(sequence_length - features_testing_tensor.size(0) % sequence_length, input_size)))
    importance_testing_tensor = torch.cat((importance_testing_tensor, torch.zeros(sequence_length - importance_testing_tensor.size(0) % sequence_length, 1)))

    features_training_tensor = features_training_tensor.view(sequence_length, -1, input_size)
    features_testing_tensor = features_testing_tensor.view(sequence_length, -1, input_size)
    
    return features_training_tensor, features_testing_tensor, importance_training_tensor, importance_testing_tensor