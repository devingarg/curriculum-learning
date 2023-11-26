import torch
from torchvision import datasets, transforms

# Define the transformations to apply to the Flowers102 data
transform_flowers102 = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image tensors
])

# Define the transformations to apply to the CIFAR-10 data
transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image tensors
])


def get_data_loaders(dataset, batch_size=16, num_workers=2, return_dataset=False):
    
    # Define the training and test datasets

    if dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)
        num_classes = 10
    
    elif dataset == "flowers102":
        train_dataset = datasets.Flowers102(root='./data', split="train", download=True, transform=transform_flowers102)
        test_dataset = datasets.Flowers102(root='./data', split="test", download=True, transform=transform_flowers102)
        num_classes = 102
    
    # Define the dataloaders to load the data in batches during training and testing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    if return_dataset:
        return (train_loader, test_loader), num_classes, (train_dataset, test_dataset) 
    else:
        return train_loader, test_loader, num_classes
