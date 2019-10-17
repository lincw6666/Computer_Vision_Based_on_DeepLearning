# -*- coding: utf-8 -*-
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pandas as pd

"""
Load the data. Seperate it into training and validation set.
"""

# Define the transform apply on the dataset.
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
# Load the dataset.
hw1_dataset = datasets.ImageFolder(root='dataset/dataset/train',
                                   transform=data_transform)
data_classes = hw1_dataset.classes

# Split the dataset into training and testing set.
torch.manual_seed(1)
split_point = int(len(hw1_dataset) * 0.7)
indices = torch.randperm(len(hw1_dataset)).tolist()
dataset = torch.utils.data.Subset(hw1_dataset, indices[:split_point])
dataset_test = torch.utils.data.Subset(hw1_dataset, indices[split_point:])

# Define training and validation data loaders.
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=4)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4)

"""
Show the picture. We can know that whether we have loaded the data correctly or
not.
"""

for batched_data, label in data_loader_test:
    print(batched_data.size(), label.size())

    # Show image.
    npimg = batched_data[0].numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

    break

"""
Build a CNN model.
"""

device = torch.device('cuda') if torch.cuda.is_available()\
                              else torch.device('cpu')

# Get the pretrained model.
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# Modify the ouput layer to fit our task.
num_features = model.fc.in_features
# Our dataset has 13 classes.
num_classes = 13
model.fc = torch.nn.Linear(num_features, num_classes)
# move model to the right device
model.to(device)

# Construct a criterion and an optimizer.
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# And a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for @num_epochs epochs
num_epochs = 20
for epoch in range(num_epochs):
    # Train for one epoch, printing every 10 iterations
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    # update the learning rate
    lr_scheduler.step()

print('Finished Training')

"""
Save the parameters of the trained model.
"""

# Save result.
model_params_result_path = './hw1_net.pth'
torch.save(model.state_dict(), model_params_result_path)

"""
Create a new network to load the trained model.
"""

# Load Model.
net = torchvision.models.resnet18(pretrained=False)
net.fc = torch.nn.Linear(num_features, num_classes)
net.load_state_dict(torch.load(model_params_result_path))
net.eval()
net.to(device)

"""
Evaluation
"""

correct = 0
total = 0
with torch.no_grad():
    for data in data_loader_test:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test images: %d %%' % (
    len(data_loader_test), 100 * correct / total))

"""
Make prediction to non-labeld data.
"""


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def MyDataset(root_dir, transform):
    file_name = []
    images = []
    for root, _, fnames in sorted(os.walk(os.path.join(root_dir))):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            file_name.append(fname[:10])
            item = transform(default_loader(path))
            images.append(torch.unsqueeze(item, 0))
    return file_name, images

file_name, predict_dataset = MyDataset('dataset/dataset/test', data_transform)
prediction = []
with torch.no_grad():
    for data in predict_dataset:
        data = data.to(device)
        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)
        prediction.append(predicted.item())

"""
Export to a .csv file.
"""

df = pd.DataFrame({
    'id': file_name, 'label': [data_classes[x] for x in prediction]
})
df.to_csv('result.csv', index=False)
