import torch 
import torchvision 
import numpy as np 
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models 



def show_images_grid(images, labels, title=None):
    # Make a grid of images (nrow defines the number of columns)
    # images = images[:8]
    nrow = 4 
    img_grid = torchvision.utils.make_grid(images, nrow=nrow)
    
    # Convert the tensor to a numpy array
    img_grid = img_grid / 2 + 0.5  # unnormalize (if necessary)
    np_img = img_grid.numpy()
    
    # Plot the images
    plt.figure(figsize=(8, 4))
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # Convert from Tensor image to numpy array
    if title:
        plt.title(title)
    plt.axis('off')  # Turn off axis
    # Save the figure
    plt.savefig(f'/workspace/XAI606-project/training_images_{title}.png', dpi=300)
    plt.close() 

def test(val_loader, snet, criterion, epoch):

    # switch to evaluate mode
    snet.eval()
    val_loss =  0 
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_indx, (input, target) in enumerate(val_loader):  
            
            if batch_indx ==0 and epoch == 0: 
                # Visualize the training images in a grid
                show_images_grid(input[:8], target[:8].tolist() , title='val')

            input, target = input.cuda(), target.cuda()
            # input, target = Variable(input), Variable(target)

            # compute output
            output = snet(input)
            loss = criterion(output, target)
            val_loss += loss.item()

            # Calculate accuracy
            predicted = torch.argmax(output, axis=1) 
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    print(f"Epoch {epoch } | Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct/total:.2f}%")

    val_loss = val_loss/len(val_loader) 
    acc = correct/total 
    return val_loss, acc 

    
if __name__ == '__main__':
    
    test_dir = r"/workspace/test" # /workspace/XAI606-project/dataset/train_val/test"
    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    )
    kwargs = {'num_workers': 8, 'pin_memory': True} #@ if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=True, **kwargs
    )

    checkpoint = torch.load("/workspace/XAI606-project/basemodel/best_model.pth")
    snet = models.alexnet(pretrained=True)
    id_criterion = nn.CrossEntropyLoss()
    snet.classifier[6] = nn.Linear(4096, 250)  
    snet.load_state_dict(checkpoint)  # Load the best model state
    snet = snet.cuda() 
    _, acc = test(test_loader, snet, id_criterion, 0)
    print(f'Test accuracy of the best model: {acc:.4f}') 