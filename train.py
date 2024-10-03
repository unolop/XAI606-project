from __future__ import print_function, absolute_import
import argparse
import os
import shutil
import time
import sys
import wandb 

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms, models 
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm

sys.path.append('/workspace/XAI606-project')

# from model import AlexNetModel 
from eval import test, show_images_grid

parser = argparse.ArgumentParser(description='PyTorch Sketch Me That Shoe Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='Adm weight decay')
parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='basemodel', type=str,
                    help='name of experiment')



def to_scalar(vt):
    """Transform a length-1 pytorch Variable or Tensor to scalar.
    Suppose tx is a torch Tensor with shape tx.size() = torch.Size([1]),
    then npx = tx.cpu().numpy() has shape (1,), not 1."""
    if isinstance(vt, Variable):
        return vt.data.cpu().numpy().flatten()[0]
    if torch.is_tensor(vt):
        return vt.cpu().numpy().flatten()[0]
    raise TypeError('Input should be a variable or tensor')

def main():
    
    global args, best_acc
    best_acc = 0 
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    wandb.init(project="XAI606", config={
        "Model": args.name, 
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size, 
        "test_batch_size": args.test_batch_size 
    })

    print(args)

    ###### DataSet ######
    sketch_dir =  r"/workspace/XAI606-project/dataset/train_val/train"
    train_dataset = datasets.ImageFolder(
        sketch_dir,
        transform=transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )
    
    val_dir = r"/workspace/XAI606-project/dataset/train_val/val"
    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=True, **kwargs
    )

    ###### Model ######
    snet = models.alexnet(pretrained=True)
    snet.classifier[6] = nn.Linear(4096, 250) 
    print(snet)

    if args.cuda:
        snet.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_prec']
            snet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    ###### Criteria ######
    id_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(snet.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    n_parameters = sum([p.data.nelement() for p in snet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))
    best_val_acc =  0 

    for epoch in range(1, args.epochs + 1):
        
        trainloss, trainacc = train(train_loader, snet, id_criterion, optimizer, epoch) 
        adjust_learning_rate(optimizer, epoch)
        
        # evaluate on validation set
        valloss, valacc = test(val_loader, snet, id_criterion, epoch) 
        
        print(f"Epoch {epoch} | Train Loss: {trainloss:.3f}, Train Accuracy: {100 * trainacc:.2f}%, Val Loss {valloss:.3f}, Train Accuracy: {100 * valacc:.2f}%")

        wandb.log( { 
            'epoch': epoch,
            'trainloss': trainloss, 
            'valloss': valloss, 
            'trainacc' : trainacc, 
            'valacc' : valacc 
        })

        if valacc > best_val_acc:
            best_val_acc = valacc  # Update the best validation loss
            torch.save(snet.state_dict(), "./best_model.pth")
            print(f"Best model saved at epoch {epoch + 1} with validation acc: {valacc * 100:.2f} %")

    # Load the best model  & Test the best model  
    print("Loading best model for testing...")
    
    checkpoint = torch.load("./best_model.pth")
    snet.load_state_dict(checkpoint)  # Load the best model state
    _, acc = test(test_loader, snet, id_criterion, epoch)
    print(f'Test accuracy of the best model: {acc:.4f}')

    wandb.finish() 


def train(train_loader, snet, id_criterion, optimizer, epoch): 
    
    snet.train()
    running_loss = 0.0
    correct = 0 
    total = 0 
    
    for batch_indx, (input, target) in enumerate(train_loader):
        
        if batch_indx == 0 and epoch == 0: 
            # Visualize the training images in a grid
            show_images_grid(input[:8], target[:8].tolist()   , title='train')

        if args.cuda:
            input, target = input.cuda(), target.cuda()
        input, target = Variable(input), Variable(target)

        # compute output
        output = snet(input)
        loss = id_criterion(output, target)
        running_loss += loss.item() 

        optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm(snet.parameters(), 100.0)
        optimizer.step()

        predicted = torch.argmax(output, axis=1) 
        total += target.size(0)
        correct += (predicted == target).sum().item()

    acc = correct/total 
    loss = running_loss/len(train_loader)
     
    return loss, acc  


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "./runs/%s/" % (args.name)
    if not os.path.exists(directory):
        print('make directory')
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    print('')
    if is_best:
        shutil.copyfile(filename, './runs/%s/' % (args.name) + 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 10))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()