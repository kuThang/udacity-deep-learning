# Pytorch node
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
