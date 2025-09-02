import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from PIL import Image
import numpy

MODEL_PATH = "model/model.pth"

LETTER_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
class ASLCNN(nn.Module):
    def __init__(self, num_classes=26):
        super(ASLCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ImageClassifier():
    def __init__(self):
        self.device = torch.device("cpu")

        self.loadedModel = ASLCNN()
        self.loadedModel = self.loadedModel.to(self.device)
        self.loadedModel.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=False))
        self.loadedModel.eval()

        self.testtransform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def classifyImage(self, img_path):
        image = Image.open(img_path)
        image = self.testtransform(image)
        image = image.to(self.device)

        with torch.no_grad():
            self.loadedModel.eval()
            output = self.loadedModel(image.unsqueeze(0))

            k = 5
            normalizedOutput = output
            normalizedOutput = F.softmax(normalizedOutput, dim=1)
            topFiveValues, topFiveIndices = torch.topk(normalizedOutput, k = k, largest = True, dim = 1)
            topFiveValues = topFiveValues.squeeze()
            topFiveIndices = topFiveIndices.squeeze()

        normalizedOutputDict = {
            LETTER_LIST[idx.item()]: f"{val.item() * 100:.2f}"
            for val, idx in zip(topFiveValues, topFiveIndices)
        }
        return normalizedOutputDict

    def idx_to_class(self, idx):
        letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        return letter_list[idx]