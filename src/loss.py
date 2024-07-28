# template
class FakeLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FakeLoss, self).__init__()
 
    def forward(self, inputs, targets, extra_params):        
        
        inputs = F.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        
        return intersection
