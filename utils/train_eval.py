import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter



def model_eval(model, dataloader, device):
    """
    Given a model and a test dataloader as input, returns the accuracy
    of the model on the data in that dataloader.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def train(model, train_loader, test_loader, num_epochs, config, device, writer):
    """ 
    Training loop for the baseline network without any curriculum i.e.,
    the training data is shuffled and presented to the network in batches.
    """
    optimizer = config["opt"]
    criterion = config["crit"]
    log_train_every = config["log_freq_tr"]
    log_test_every = config["log_freq_test"]
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)


            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % 100 == 0:
                print(f"[Epoch: {epoch + 1}, Batch: {i + 1}] Loss: {running_loss/100:.3f}")
                running_loss = 0.0

            # compute training & testing accuracy every couple of iterations        
            if (i+1) % log_train_every == 0:
                train_accuracy = model_eval(model, train_loader, device)

                # Log the loss
                writer.add_scalar('Loss/train', loss.cpu().item(), epoch * len(train_loader) + i)

                # Log the training accuracy
                writer.add_scalar('Accuracy/train', train_accuracy, epoch * len(train_loader) + i)

            if (i+1) % log_test_every == 0:
                test_accuracy = model_eval(model, test_loader, device)

                # Log the test accuracy
                writer.add_scalar('Accuracy/test', test_accuracy, epoch * len(train_loader) + i)

    writer.close()
    print("Training finished.")
    
    
def train_curriculum(model, dataset, train_dataset, test_loader, optimizer, criterion,
                     clustered_data, c_sizes, mode, num_epochs, data_config, device):
    """ 
    Training loop for the unsupervised curriculum i.e., training the model on one cluster
    at a time. 
    """
    
    num_clusters = len(c_sizes)
    
    if mode == "L2S":
        pass
    elif mode == "S2L":
        # Since c_size is sorted in the decreasing order, reverse it
        c_sizes = sorted(c_size, key=lambda x: x[1])
    else:
        print("Invalid mode")
    
    for c_idx, _ in c_sizes:
        
        # Set the directory for the TensorBoard logs
        log_dir = f"./logs/{dataset}/vgg16_{mode}_{num_epochs}_c{num_clusters}_{c_idx}"  
        writer = SummaryWriter(log_dir)

        # record the indices that are being used to train for a sanity check
        c_size_str = ', '.join(str(val) for val in c_sizes)
        writer.add_text("Cluster sizes: ", c_size_str)

        # create dataloader for the current cluster
        cluster_loader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=data_config["batch_size"],
                                                     num_workers=data_config["num_workers"], 
                                                     # this sampler, samples from the list of indices passed
                                                     sampler=SubsetRandomSampler(clustered_data[c_idx]),
                                                     drop_last=True)
        
        # Train the network on the current cluster for `num_epochs` epochs
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(cluster_loader):
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero out the gradients
                optimizer.zero_grad()

                # get predictions and compute loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # backpropagate
                loss.backward()
                optimizer.step()

                # track the loss
                running_loss += loss.item()

            # Log the train loss
            epoch_loss = running_loss/len(cluster_loader)
            writer.add_scalar('Loss/train', epoch_loss, epoch+1)
            running_loss = 0.0

            # Log the test accuracy
            test_accuracy = model_eval(model, test_loader, device)
            writer.add_scalar('Accuracy/test', test_accuracy, epoch+1)

        print(f"Training on cluster {c_idx} ({len(clustered_data[c_idx])/len(train_dataset)*100:.2f}% data) done", end=' ') 
        print(f"Test Acc: {test_accuracy:.3f}")

    writer.close()