# Pytorch node

## Utils
- Visualize image and labels
- - See [l6 CNN/cifar10_cnn_exercise.ipynb](https://github.com/kuThang/udacity-deep-learning/blob/master/l6%20CNN/cifar10_cnn_exercise.ipynb) or [l6 CNN/conv_visualization.ipynb](https://github.com/kuThang/udacity-deep-learning/blob/master/l6%20CNN/conv_visualization.ipynb) or [l6 CNN/maxpooling_visualization.ipynb](https://github.com/kuThang/udacity-deep-learning/blob/master/l6%20CNN/maxpooling_visualization.ipynb)
- **Visualize CNN**
- - See [visualize_CNN in Keras](https://github.com/kuThang/udacity-deep-learning/blob/master/l6%20CNN/visualize_CNN.ipynb)

## 1. Train, calculate loss and update
```
from torch import nn
import torch.nn.functional as F
model = nn.Sequential(nn.Linear(size, size),
        nn.ReLU(),
        nn.Linear(size, size),
        nn.ReLU())
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)   // define optimizer and parameters which should be updated
for i in range(epochs):
    for images, labels in trainloader:
        optimizer.zero_grad()       // reset optimizer for each batch
        images, labels = next(iter(...))
        fw = model.forward(images)  // calculate feedforward
        loss = criterion(logits, labels) // calcuate loss value
        loss.backward()             // calculate grad for each parameter
        optimizer.step()            // update parameters (only update parameters that are defined )

```

## 2. transfer learning
```
from torchvision import models, transforms
model = models.densenet121(pretrained = True)
for param in model.parameters():
    param.requires_grad = False   // freeze parameters
classifier = nn.Sequential([])
model.classifier = classifier
criterion = 
optimizer = 
for i in range (epochs):
    for images, labels in trainloader:
        // training
        optimizer.zero_grad()
        fw = model.forward(images)
        loss = criterion(fw, labels)
        loss.backward()
        optimizer.step()
    
    if step % test_every == 0:
        model.eval()    // switch model from training to evaluating
        for imgs, lbls in testloader:
            fw = model.forward(imgs)
            .
            .
            .
        model.train()   // switch model back to training
```

## 3. Create model class
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # linear layer (784 -> 1 hidden node)
        h1_node = 512
        h2_node = 512
        self.fc1 = nn.Linear(28 * 28, h1_node)
        self.fc2 = nn.Linear(h1_node, h2_node)
        self.fc3 = nn.Linear(h2_node, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```