import torch
from torch import nn
 
class CustomLoss(nn.Module):
    def __init__(self, lambda_value = 1.5, mu_value = 1.5):
        super(CustomLoss, self).__init__()
        self.lambda_value = lambda_value
        self.mu_value = mu_value

    def forward(self, predicted, true):
        n = predicted.size(0)  # Number of samples
        
        predicted_deviation = torch.std(predicted)
        real_deviation = torch.std(true)
        deviation_term = (predicted_deviation - real_deviation) ** 2
        
        sum_x = torch.sum(predicted)
        sum_x_sqr = torch.sum(predicted ** 2)
        sum_xy = torch.sum(predicted * true)
        sum_y = torch.sum(true)
        sum_y_sqr = torch.sum(true ** 2)
        
        correlation_term = 2 / ((n * sum_xy - sum_x * sum_y) / ((n*sum_x_sqr-sum_x**2)*(n*sum_y_sqr-sum_y**2))**0.5 + 1) - 1
        
        error_term = (-torch.sum(torch.log(1 - torch.abs(predicted - true) / 5))) / n
        
        loss =  error_term + self.lambda_value * deviation_term + self.mu_value * correlation_term ** 2
        return loss
