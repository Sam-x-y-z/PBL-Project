import torch
from LSTM_MLP_Model import LSTM_MLP_Model
import json
from GetDataTensors import getDataTensors
from Parameters import lstm_input_size as input_size, lstm_sequence_length as sequence_length

# model = LSTM_MLP_Model()
# model_data= torch.load('./LSTM-MLP-Lambda-1.05(final).pth', map_location='cpu')
# model.load_state_dict(model_data)
# model.eval()

with open('./data.json','r') as f:
    data = json.load(f)

data_testing = data[41:]

with open("TestingData.json",'w') as f:
    json.dump(data_testing,f)

# data_new={}

# for vid in data_testing:
#     features_tensor = torch.tensor([item["features"] for item in vid["clips"]])
#     features_tensor = torch.cat((features_tensor, torch.zeros(sequence_length - features_tensor.size(0) % sequence_length, input_size)))
#     features_tensor = features_tensor.view(sequence_length, -1, input_size)

#     with torch.no_grad():
#         test_outputs = model(features_tensor).view(-1).tolist()

#     data_new[vid["video"]] = []
#     for indx in range(len(vid["clips"])):
#         data_new[vid["video"]].append((vid["clips"][indx]["segment"], vid["clips"][indx]["importance"], test_outputs[indx]))

# with open("KTS-input.json",'w') as f:
#     json.dump(data_new,f)



    
