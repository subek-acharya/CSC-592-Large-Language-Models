import torch
from MyDataSet import MyDataset
import torchvision
import matplotlib.pyplot as plt

def get_train_loader(data_dir, batch_size, transform=None):
    dataset = MyDataset(data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    return data_loader

def get_test_loader(data_dir, batch_size, transform=None):
    dataset = MyDataset(data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return data_loader

def plot_images(images, labels): 
    # normalise=True below shifts [-1,1] to [0,1]
    img_grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)
    np_img = img_grid.numpy().transpose(1,2,0) # pytorch has the order, c,w,h
    # to be able to view an image, we need to change the order and 
    # put it in width, height, color order
    plt.imshow(np_img)
    plt.show()

