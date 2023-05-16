from inference import FaceExpressionRecognition
import torch
import matplotlib.pyplot as plt


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fer = FaceExpressionRecognition(model_adr="saved_model.pth", device="cpu")
    img = plt.imread("/home/mohammadmahdu/Downloads/FER-2013-4_Class/validation/angry/PrivateTest_1109992.jpg")

    fer.predict(img)