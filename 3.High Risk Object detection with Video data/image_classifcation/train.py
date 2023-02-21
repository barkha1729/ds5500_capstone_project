import torch.optim as optim
from model import Net
import torch.nn as nn
from dataset import create_train_dataset,create_train_loader
import torch
from tqdm import tqdm
from config import NUM_EPOCHS, DEVICE

train_dataset=create_train_dataset()
train_loader=create_train_loader(train_dataset)

net=Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
net=net.to(DEVICE)
print("Training on device",DEVICE)
for epoch in tqdm(range(NUM_EPOCHS)):  # loop over the dataset multiple times

    running_loss = 0.0
    prog_bar = tqdm(train_loader, total=len(train_loader))
    for i, data in enumerate(prog_bar):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels=torch.stack(list(labels), dim=0)
        inputs=torch.stack(list(inputs), dim=0)
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')