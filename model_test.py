import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms


class CovidNN(nn.Module):
    def __init__(self) :
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1)
        
        #nn
        self.fc1 = nn.Linear(16*30*30, 1000)
        self.fc2 = nn.Linear(1000, 200)
        self.fc3 = nn.Linear(200, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2,2)
        
        x = x.view(x.size(0),-1)

        #nn
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    

model = CovidNN()
model.load_state_dict(torch.load('./xRayCNN.pt'))
model.eval()


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def predict(image_path):
    with Image.open(image_path) as img:
        img = transform(img).view((1,1,128,128))
    with torch.no_grad():
        pred = model(img)
    results = ["COVID-19 (coronavirus)", "Healthy Lung", "Viral Pneumania"]
    return results[pred.argmax()]

input_path = input('Enter Image path to predict: ')
# input_path = './images_model_didnt_see/pneumonia.png'

print(predict(input_path))