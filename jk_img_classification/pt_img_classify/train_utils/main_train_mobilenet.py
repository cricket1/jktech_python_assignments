import shutil
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import time
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import os
from pt_img_classify.train_utils.common import device, num_classes, bs, num_epochs, visualise_dir, model_dir, \
    model_name, train_data_loader, valid_data_loader, train_data_size, valid_data_size, test_data_loader, \
    test_data_size


if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir)

if os.path.exists(visualise_dir):
    shutil.rmtree(visualise_dir)
os.makedirs(visualise_dir)

# Load pretrained mobilenet v2 Model
mobilenet_v2 = models.mobilenet_v2(pretrained=True)
mobilenet_v2 = mobilenet_v2.to(device)
for param in mobilenet_v2.parameters():
    param.requires_grad = False

# Change the final layer of mobilenetv2 Model for Transfer Learning
fc_inputs = mobilenet_v2.last_channel

mobilenet_v2.classifier = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes),  # Since 10 possible outputs
    nn.LogSoftmax(dim=1)  # For using NLLLoss()
)

# Convert model to be used on GPU
mobilenet_v2 = mobilenet_v2.to(device)


# Define Optimizer and Loss Function
loss_func = nn.NLLLoss()
optimizer = optim.Adam(mobilenet_v2.parameters())


def main():
    # Print the model to be trained
    summary(mobilenet_v2, input_size=(3, 224, 224), batch_size=bs, device=device)

    # Train the model for 25 epochs
    trained_model, history = train_and_validate(mobilenet_v2, loss_func, optimizer, num_epochs)

    torch.save(history, os.path.join(visualise_dir, 'vis_history.pt'))

    visualise(visualise_dir)
    # Test a particular model on a test image

    model = torch.load(os.path.join(model_dir, model_name + '_model_' + str(num_epochs - 1) + '.pt'))

    # predict(model, 'pixabay-test-animals/triceratops-954293_640.jpg')

    # Load Data from folders
    compute_test_set_accuracy(model, loss_func)

    print(' Training and testing complete')


def visualise(visualise_dir_):
    history = torch.load(os.path.join(visualise_dir_, 'vis_history.pt'))
    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(visualise_dir_, 'mask_loss_curve.png'))
    plt.show()
    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(visualise_dir_, 'mask_accuracy_curve.png'))
    plt.show()


def train_and_validate(model, loss_criterion, optimizer_, epochs=25):
    """
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer_: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)

    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    """

    # start = time.time()
    history = []
    # best_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        # Set to training mode
        model.train()

        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clean existing gradients
            optimizer_.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer_.step()

            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)

            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)

            # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                # print("Validation Batch number: {:03d}, Validation: Loss: {:.4f},
                # Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

        # Find average training loss and training accuracy
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        epoch_end = time.time()

        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, "
              "\n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))

        # Save if the model has best accuracy till now
        torch.save(model, os.path.join(model_dir, model_name + '_model_' + str(epoch) + '.pt'))

    return model, history


def compute_test_set_accuracy(model, loss_criterion):
    """
    Function to compute the accuracy on the test set
    Parameters
        :param model: Model to test
        :param loss_criterion: Loss Criterion to minimize
    """

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_acc = 0.0
    test_loss = 0.0

    # Validation - No gradient tracking needed
    with torch.no_grad():

        # Set to evaluation mode
        model.eval()

        # Validation loop
        for j, (inputs, labels) in enumerate(test_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            test_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc += acc.item() * inputs.size(0)

            print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

    # Find average test loss and test accuracy
    # avg_test_loss = test_loss/test_data_size
    avg_test_acc = test_acc/test_data_size

    print("Test accuracy : " + str(avg_test_acc))


if __name__ == '__main__':
    main()
