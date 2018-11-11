import numpy as np
from utils import *
from Model.train_model import Model
import argparse
import math
import time
import torch
import torch.nn as nn
import pdb

parser = argparse.ArgumentParser(description="arguments")
parser.add_argument("--epochs", default=50, type=int,
                    help="The number of epochs to run.")
parser.add_argument("--lr", default=0.001, type=float,
                    help="The learning rate.")
args = parser.parse_args()
TRAIN_IMGS_PATH = "./Data/train_imgs_v0/"
TEST_IMGS_PATH = "./Data/test_imgs_v0/"
GPU = torch.cuda.is_available()


def compute_acc(model, data, total_labels):
    """
    Compute while train and test data accuracy.
    """
    tic = time.time()
    corrects = 0
    bs = 32
    for i in range(int(math.ceil(data.shape[0] / bs))):
        start_idx = (i * bs) % data.shape[0]
        inputs = data[start_idx : start_idx+bs]
        labels = total_labels[start_idx : start_idx + bs]
        inputs = torch.from_numpy(inputs).float()
        labels = torch.from_numpy(labels).long()
        if GPU:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        corrects += (outputs.cpu().detach().argmax(dim=1).numpy() == labels.cpu().numpy()).sum() 
    total_accuracy = corrects / len(total_labels)
    toc = time.time()
    print("Time consumed for computing total acc: ", (toc - tic) / 60, " mins.")
    return total_accuracy


def train_epoch(e, model, optimizer, criterion, train_data, train_labels, test_data, test_labels):
    batch_size = 32
    running_loss = 0.0
    indices = np.random.permutation(train_data.shape[0])
    if GPU:
        model = model.cuda()
    for iteration in range(int(math.ceil(train_data.shape[0] / batch_size))):
        start_idx = (iteration * batch_size) % train_data.shape[0]
        idx = indices[start_idx : start_idx+batch_size]
        inputs = train_data[idx]
        labels = train_labels[idx]
        inputs = torch.from_numpy(inputs).float()
        labels = torch.from_numpy(labels).long()
        if GPU:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        train_loss = criterion(outputs, labels)
        train_loss.backward()

        with torch.no_grad():
            optimizer.step()
            optimizer.zero_grad()

        running_loss += train_loss.item()
        interval = 10
        if iteration % interval == 9:
            outputs = outputs.cpu().detach().argmax(dim=1)
            #pdb.set_trace()
            train_accuracy = (outputs.numpy()==labels.cpu().numpy()).mean() 
            print("Epoch: {} Iteration: {} Training loss: {} Training accuracy: {}".format((e), (iteration), running_loss/interval, train_accuracy))
            running_loss = 0.0

    state = {
        'epoch': e,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, 'state_'+str(e)+'.pt')

    train_accuracy = compute_acc(model, train_data, train_labels)
    test_accuracy = compute_acc(model, test_data, test_labels)
    print("Epoch: {} Total Training accuracy: {} Test accuracy: {}".format(e, train_accuracy, test_accuracy))


def main():
    train_data, train_labels = load_data(TRAIN_IMGS_PATH)
    test_data, test_labels = load_data(TEST_IMGS_PATH)
    train_data = preprocess(train_data)
    print(train_data.shape, train_labels.shape)
    print(test_data.shape, test_labels.shape)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()
    for e in range(args.epochs):
        train_epoch(e, model, optimizer, criterion, train_data, train_labels, test_data, test_labels)

    #train_accuracy = compute_acc(model, train_data, train_labels)

if __name__ == "__main__":
    main()
