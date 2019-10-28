import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Trainer(object):
    def __init__(self, batch_size, dataloader, network):
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.network = network

    def train(self, epochs):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.network.parameters(), lr=0.0001, momentum=0.9)
        running_loss = 0.0

        for epoch in range(epochs):
            for i, sample in enumerate(self.dataloader, 0):
                uv = sample['uv']
                mano = sample['mano']

                optimizer.zero_grad()

                outputs = self.network(uv)
                loss = criterion(outputs, mano)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 100 == 99:
                    print("i: {}; loss: {}".format(i, running_loss / 100))
                    running_loss = 0.0

