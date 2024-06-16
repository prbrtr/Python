# Import necessary libraries
import os  # This library provides functions for interacting with the operating system.
import torch  # This library provides tools for deep learning.
import torch.nn as nn  # This part of Torch helps in building and training neural networks.
import torch.nn.functional as F  # This provides various functions for building neural networks.
from torchvision.transforms import ToTensor  # This helps in transforming images into tensors.
from torchvision.datasets import ImageFolder  # This is used to load datasets of images.
import torchvision.transforms as T  # This provides tools for transforming images.
from torch.autograd import Variable  # This is used for automatic differentiation during training.
from werkzeug.utils import secure_filename  # This helps in securing file uploads in Flask.
from PIL import Image  # This library is used for opening, manipulating, and saving many different image file formats.
from flask import Flask, request, render_template  # Flask is a web application framework for Python.

# Define data directory, image size, and normalization statistics
data_dir ='./data'  # This is the directory where our dataset is stored.
image_size = 32  # This is the size to which we will resize our images.
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # These are normalization statistics for the images.

# Define image transformation
loader = T.Compose([
    T.Resize(image_size),  # This resizes the image to a specified size.
    T.CenterCrop(image_size),  # This crops the center of the image.
    T.ToTensor(),  # This converts the image to a PyTorch tensor.
    T.Normalize(*stats)  # This normalizes the image tensor using provided statistics.
])

# Load dataset
dataset = ImageFolder(data_dir+'/train', transform=ToTensor())  # This loads the dataset from the specified directory.

# Define a base class for image classification
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):  # This function calculates the loss during training.
        print(f"{batch}")
        images, labels = batch 
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        return loss
    
    def validation_step(self, batch):  # This function calculates loss and accuracy during validation.
        images, labels = batch 
        out = self(images)                    
        loss = F.cross_entropy(out, labels)   
        acc = accuracy(out, labels)          
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):  # This function calculates average loss and accuracy over a validation epoch.
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):  # This function prints the result at the end of each epoch.
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
# Define accuracy calculation function
def accuracy(outputs, labels):  # This function calculates the accuracy of the model's predictions.
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Define the CNN model for image classification
class CnnModel(ImageClassificationBase):  # This class defines a Convolutional Neural Network (CNN) model for image classification.
    def __init__(self):
        super().__init__()  # This calls the constructor of the superclass (ImageClassificationBase).
        
        # This defines the layers of the neural network.
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Convolutional layer with 3 input channels, 32 output channels, and a kernel size of 3x3 with padding of 1.
            nn.ReLU(),  # ReLU activation function to introduce non-linearity.
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Convolutional layer with 32 input channels, 64 output channels, and a kernel size of 3x3 with padding of 1.
            nn.ReLU(),  # ReLU activation function.
            nn.MaxPool2d(2, 2),  # Max pooling layer with a kernel size of 2x2 and a stride of 2.
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Convolutional layer with 64 input channels, 128 output channels, and a kernel size of 3x3 with padding of 1.
            nn.ReLU(),  # ReLU activation function.
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # Convolutional layer with 128 input channels, 128 output channels, and a kernel size of 3x3 with padding of 1.
            nn.ReLU(),  # ReLU activation function.
            nn.MaxPool2d(2, 2),  # Max pooling layer with a kernel size of 2x2 and a stride of 2.
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Convolutional layer with 128 input channels, 256 output channels, and a kernel size of 3x3 with padding of 1.
            nn.ReLU(),  # ReLU activation function.
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Convolutional layer with 256 input channels, 256 output channels, and a kernel size of 3x3 with padding of 1.
            nn.ReLU(),  # ReLU activation function.
            nn.MaxPool2d(2, 2),  # Max pooling layer with a kernel size of 2x2 and a stride of 2.
            
            nn.Flatten(),  # Flatten layer to convert the 3D tensor into a 1D tensor.
            nn.Linear(256*4*4, 512),  # Fully connected layer with 256*4*4 input features and 512 output features.
            nn.ReLU(),  # ReLU activation function.
            nn.Linear(512, 4)  # Fully connected layer with 512 input features and 4 output features (assuming 4 classes for classification).
        )
        
    def forward(self, xb):  # This function defines the forward pass of the model.
        return self.network(xb)

# Function to get the default device (NVDIA if available, else CPU)
def get_default_device():  # This function checks if a GPU is available and returns it, otherwise returns CPU.
    global cuda_arc
    if torch.cuda.is_available():
        cuda_arc=True
        return torch.device('cuda')
    else:
        cuda_arc=False
        return torch.device('cpu')

# Function to move data to the chosen device
def to_device(data, device):  # This function moves data to the specified device (CPU or GPU).
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Function to predict image class
def predict_image(img, model="any"):  # This function predicts the class of an image using the provided model.
    xb = to_device(img.unsqueeze(0), get_default_device())
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    print(dataset.classes[preds[0].item()])
    return dataset.classes[preds[0].item()]

# Function to load an image
def image_loader(image_name):  # This function loads and preprocesses an image.
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True) 
    if cuda_arc:
        return image.cuda() 
    else:
        return image.cpu() 

# Load the trained model
model = to_device(CnnModel(), get_default_device())  # This loads the trained model onto the appropriate device (CPU or GPU).
print(model)
if not cuda_arc:
    model.load_state_dict(torch.load(map_location=torch.device('cpu'),f='modelCottonDemo.pth'))  # This loads the model's weights.
else:
      model.load_state_dict(torch.load(f='modelCottonDemo.pth'))  # This loads the model's weights.




# Define a Flask app
app = Flask(__name__)

# Define routes
@app.route('/', methods=['GET'])  # This route renders the main page.
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])  # This route handles image uploads and predictions.
def upload():
    if request.method == 'POST':
        f = request.files['file']  # This gets the uploaded file.
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))  # This constructs the file
        f.save(file_path)  # This saves the uploaded file to a specified path.

        image = image_loader(file_path)  # This loads and preprocesses the uploaded image.

        pred = predict_image(image, model)  # This predicts the class of the uploaded image using the trained model.

        return pred  # This returns the predicted class label.
    return None

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
