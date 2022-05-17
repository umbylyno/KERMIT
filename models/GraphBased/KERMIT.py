import torch, torchvision
from torch import nn
with: {predizione.item()} confidence'

class KERMIT(nn.Module):
    def __init__(self, **kwargs):
        super(ImageEncoder, self).__init__()
        
        self.model = torchvision.models.resnet50(pretrained=True)

        #2048 feature vector 
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

        #Freeze of parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        #image features
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 768)
        
        
        #concat image+text+graph features
        self.fc3 = nn.Linear(768 + 768 + 768, 1300)
        self.fc4 = nn.Linear(1300, 700)
        self.fc5 = nn.Linear(700, 250)
        self.fc6 = nn.Linear(250, 50)
        self.fc7 = nn.Linear(50, 1)

        #dropout layer
        self.dropout = nn.Dropout(0.3)

    def forward(self, image, text, graph):

        #Forward image
        x1 = self.model(image).squeeze()
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        
        #text embedding
        x2 = text.squeeze(dim=0)  
        
        #graph embedding
        x3 = graph
        
        #concat
        x = torch.cat((x1, x2, x3), dim=1)
        
        #classification head
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.fc7(x)

        return torch.sigmoid(x)
