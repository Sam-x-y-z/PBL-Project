import torch
from torch import nn
 
class CustomLoss(nn.Module):
    def __init__(self, lambda_value = 3):
        super(CustomLoss, self).__init__()
        self.lambda_value = lambda_value

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
        
        term_1 = torch.sum(((predicted >= true) & (true >= 2.5)).float() * torch.log(1 - (predicted - true) / 5))
        term_2 = torch.sum(((predicted < true) & (true < 2.5)).float() * torch.log(1 - (true - predicted) / 5))
        
        term_3 = torch.sum(((predicted >= true) & (true < 2.5)).float() * torch.log(1 - (predicted - true) / 5))
        term_4 = torch.sum(((predicted < true) & (true >= 2.5)).float() * torch.log(1 - (true - predicted) / 5))
        
        if torch.isnan(term_1):
            term_1 = torch.tensor(0, requires_grad=True, dtype=torch.float32)
        if torch.isnan(term_2):
            term_2 = torch.tensor(0, requires_grad=True, dtype=torch.float32)
        if torch.isnan(term_3):
            term_3 = torch.tensor(0, requires_grad=True, dtype=torch.float32)
        if torch.isnan(term_4):
            term_4 = torch.tensor(0, requires_grad=True, dtype=torch.float32)
        
        error_term = (- (term_1 + term_2) - self.lambda_value * (term_3 + term_4)) / n
        
        loss =  error_term + 1.5 * deviation_term + 1.5 * correlation_term ** 2
        return loss
