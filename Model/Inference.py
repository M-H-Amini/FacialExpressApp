import torch
import torchvision
import numpy as np
from model import *
import matplotlib.pyplot as plt

class FaceExpressionRecognition():
    def __init__(self, model_adr, device="cpu"):
        self.model_adr = model_adr
        self.model = Model()
        self.loss_func = torch.nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        

        self.device = device

        self.emotion_labels = {
            0: 'angry', 
            1: 'happy', 
            2: 'sad', 
            3: 'neutral',
        }

        self.pretrained_model_loading()


    def pretrained_model_loading(self):
        loaded_model = torch.load(self.model_adr, map_location=self.device)
        self.model.load_state_dict(loaded_model['model_state_dict'])
        self.optimizer.load_state_dict(loaded_model['optimizer_state_dict'])



    def apply_transform(self, image):
        image = image / 255
        image = np.float32(image)
        image = torchvision.transforms.ToTensor()(image)
        image = torchvision.transforms.Resize((224, 224))(image)
        image = torch.cat((image, image, image), dim=0)
        image = torch.unsqueeze(image, dim=0)

        return image



    def predict(self, inp_image, verbose=False):
        self.model.eval()
        transformed_image = self.apply_transform(inp_image)
        output = self.model(transformed_image)

        return self.visualize(output, verbose=verbose)

    

    def visualize(self, x, verbose=False):
        softmax = torch.nn.Softmax(dim=1)
        max_value, index = torch.max(softmax(x), dim=1)

        class_label = self.emotion_labels[index.numpy()[0]]
        certainty = max_value.detach().numpy()[0]*100
        print("----------------------")
        print("Result: %s %2.2f%% "%(class_label, certainty))
        print("----------------------")
        return class_label, certainty

