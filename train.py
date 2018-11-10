import numpy as np
from utils import *
from Model.train_model import Model
import argparse
import math
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description="arguments")
parser.add_argument("--epochs", default=50, type=int,
                    help="The number of epochs to run.")
parser.add_argument("--lr", default=0.001, type=float,
                    help="The learning rate.")
args = parser.parse_args()


def compute_acc(model, data, total_labels):
    tic = time.time()
    corrects = 0
    bs = 32
    for i in range(int(math.ceil(data.shape[0]/bs))):
        start_idx = (i * bs) % data.shape[0]
        inputs = data[start_idx:start_idx+bs]
        labels = total_labels[start_idx:start_idx+bs]
        inputs = torch.from_numpy(inputs).float()
        labels = torch.from_numpy(labels).long()
        outputs = model(inputs)
        corrects += (outputs.cpu().detach().argmax(dim=1).numpy()==labels.cpu().numpy()).sum() 
    total_accuracy = corrects / len(total_labels)
    toc = time.time()
    print("Time consumed for computing total acc: ", (toc - tic) / 60, " mins.")
    return total_accuracy

def train_epoch(e, model, optimizer, criterion, train_data, train_labels, test_data, test_labels):
    batch_size = 32
    indices = np.random.permutation(train_data.shape[0])
    for iteration in range(int(math.ceil(train_data.shape[0] / batch_size))):
        start_idx = (iteration * batch_size) % train_data.shape[0]
        idx = indices[start_idx:start_idx+batch_size]
        inputs = train_data[idx]
        labels = train_labels[idx]
        inputs = torch.from_numpy(inputs).float()
        labels = torch.from_numpy(labels).long()

        outputs = model(inputs)
        train_loss = criterion(outputs, labels)

        train_loss.backward()

        with torch.no_grad():
            optimizer.step()
            optimizer.zero_grad()

        running_loss += train_loss.item()
        interval = 10
        if iteration % interval == 9:
            train_accuracy = (outputs.numpy()==labels.cpu().numpy()).mean() 
            print("Epoch: {} Iteration: {} Training loss: {}, Training accuracy: {}".format((e), (iteration), running_loss/interval, train_accuracy))
            running_loss = 0.0

    train_accuracy = compute_acc(model, train_data, train_labels)
    test_accuracy = compute_acc(model, test_data, test_labels)
    print("Epoch: {} Total Training accuracy: {} Test accuracy: {}".format(e, train_accuracy, test_accuracy))


def main():
    train_data, train_labels = load_data("../existing material/Data/train_imgs/")
    test_data, test_labels = load_data("../existing material/Data/test_imgs/")
    print(train_data.shape, train_labels.shape)
    print(test_data.shape, test_labels.shape)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()
    for e in range(args.epochs):
        train_epoch(e, model, optimizer, criterion, train_data, train_labels, test_data, test_labels)


if __name__ == "__main__":
    main()