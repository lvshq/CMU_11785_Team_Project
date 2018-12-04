import numpy as np
from Utils.utils import *
from Utils.const import *
from Utils.eval_func import *
from Model.train_model import Model
from Model.resnet import resnet18
from Model.vgg import vgg16
import argparse
import math
import time
import torch
import torch.nn as nn
import pdb

parser = argparse.ArgumentParser(description="arguments")
parser.add_argument("--epochs", default=20, type=int,
                    help="The number of epochs to run.")
parser.add_argument("--lr", default=0.001, type=float,
                    help="The learning rate.")
parser.add_argument("--batch_size", default=64, type=float,
                    help="The batch size.")
parser.add_argument("--cuda", default=False, type=bool,
                    help="The GPU.")
parser.add_argument("--use_test_for_train", default=False, type=bool,
                    help="Use test data for training to facilite model validation.")
args = parser.parse_args()
# if torch.cuda.is_available():
#     args.cuda = True

net = 'vgg' # resnet, alexnet, vgg
pretrain = False
pretrain_states = 'state_resnet_nopretrain_9.pt'
if net == 'alexnet':
    logger = 'alexnet_pretrain' if pretrain == True else 'alexnet_nopretrain'
elif net == 'resnet':
    logger = 'resnet_pretrain' if pretrain == True else 'resnet_nopretrain'
elif net == 'vgg':
    logger = 'vgg_pretrain' if pretrain == True else 'vgg_nopretrain'

def compute_acc(model, data, total_labels):
    """
    Compute while train and test data accuracy.
    """
    tic = time.time()
    corrects = 0
    bs = args.batch_size
    for i in range(int(math.ceil(data.shape[0] / bs))):
        start_idx = (i * bs) % data.shape[0]
        inputs = data[start_idx : start_idx+bs]
        labels = total_labels[start_idx : start_idx + bs]
        inputs = torch.from_numpy(inputs).float()
        labels = torch.from_numpy(labels).long()
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        corrects += (outputs.cpu().detach().argmax(dim=1).numpy() == labels.cpu().numpy()).sum() 
    total_accuracy = corrects / len(total_labels)
    toc = time.time()
    print("Time consumed for computing total acc: ", (toc - tic) / 60, " mins.")
    return total_accuracy


def train_epoch(e, model, optimizer, criterion, train_data, train_labels, test_data, test_labels):
    tic = time.time()
    model.train()
    batch_size = args.batch_size
    running_loss = 0.0
    indices = np.random.permutation(train_data.shape[0])
    total_batch = int(math.ceil(train_data.shape[0] / batch_size))
    for iteration in range(total_batch):
        start_idx = (iteration * batch_size) % train_data.shape[0]
        idx = indices[start_idx : start_idx+batch_size]
        inputs = train_data[idx]
        labels = train_labels[idx]
        inputs = torch.from_numpy(inputs).float()
        labels = torch.from_numpy(labels).long()
        if args.cuda:
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
            toc = time.time()
            print("Epoch: {} Iteration: {:5.2f}% Time: {:5.2f}s Training loss: {:5.4f} Training accuracy: {:5.4f}".format(e, 100 * iteration/total_batch, (toc-tic)/interval, running_loss/interval, train_accuracy))
            running_loss = 0.0
            tic = time.time()

    state = {
        'epoch': e,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, 'state_' + logger + '_' + str(e) + '.pt')

    train_accuracy = compute_acc(model, train_data, train_labels)
    test_accuracy = compute_acc(model, test_data, test_labels)
    print("Epoch: {} Total Training accuracy: {:5.4f} Test accuracy: {:5.4f}".format(e, train_accuracy, test_accuracy))
    return train_accuracy, test_accuracy

def extract_embeddings(model, test_data):
    model.eval()
    batch_size = args.batch_size
    indices = np.arange(test_data.shape[0])
    total_batch = int(math.ceil(test_data.shape[0] / batch_size))
    embeddings = torch.tensor(())
    if args.cuda:
        embeddings = embeddings.cuda()
    for iteration in range(total_batch):
        start_idx = (iteration * batch_size) % test_data.shape[0]
        idx = indices[start_idx : start_idx+batch_size]
        inputs = test_data[idx]
        inputs = torch.from_numpy(inputs).float()
        if args.cuda:
            inputs = inputs.cuda()
        outputs = model(inputs)
        outputs = model.get_embedding()
        embeddings = torch.cat((embeddings, outputs), 0)
    print('The embeddings size: ', embeddings.size())
    return embeddings
    
def main():
    if args.use_test_for_train:
        train_data, train_labels = load_data(TEST_IMGS_PATH)
    else:
        train_data, train_labels = load_data(TRAIN_IMGS_PATH)
    test_data, test_labels = load_data(TEST_IMGS_PATH)
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)
    print(train_data.shape, train_labels.shape)
    print(test_data.shape, test_labels.shape)
    
    if net == 'alexnet':
        model = Model()
    elif net == 'resnet':
        model = resnet18()
    elif net == 'vgg':
        model = vgg16()
    
    if args.cuda:
        model = model.cuda()
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, threshold=0.01, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 0
    if pretrain:
        state = torch.load(pretrain_states)
        epochs = state['epoch'] + 1
        print(epochs)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        
    for e in range(epochs, args.epochs):
        train_accuracy, test_accuracy = train_epoch(e, model, optimizer, criterion, train_data, train_labels, test_data, test_labels)
        #embeddings = extract_embeddings(model, test_data)
        #score = eval(embeddings, test_labels, 100)
        #print('Score {:5.4f}'.format(score))
        scheduler.step(test_accuracy)
    

if __name__ == "__main__":
    main()
