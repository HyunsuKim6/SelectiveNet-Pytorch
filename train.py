from __future__ import print_function, division

from torch.optim import lr_scheduler
import time
import copy
from SelectiveNet import *
from LoadingData import *


def train(model, dataloader, criterion, optimizer, scheduler, pkl_name, num_epochs=30):
    # device setting
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(torch.cuda.get_device_name(0))

    since = time.time()

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
        sel_risk = 0
        curr_coverage = 0

        # Iterate over data.
        for index, data in enumerate(dataloader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            pred_outputs, sel_outputs, aux_outputs = model(inputs)
            _, preds = torch.max(pred_outputs, 1)
            loss, sel_risk, curr_coverage = criterion(pred_outputs, sel_outputs, aux_outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if index % 100 == 0:
                print('Epoch {} [{}/{} ({:.0f}%)]:  Loss: {:.6f}  Selective risk: {:.6f}  Current coverage: {:.6f}'
                      .format(epoch, index * len(inputs), len(dataloader.dataset),
                              100. * index / len(dataloader), loss, sel_risk, curr_coverage))

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print('Epoch {} result:  Loss: {:.4f}  Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

        if epoch % 50 == 0:
            torch.save(model.state_dict(), pkl_name)

        print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print("A model was saved")
    print("Training is done")

    return model


if __name__ == "__main__":
    num_worker = 12
    pkl_name = 'checkpoints/SelectiveNet.pkl'

    model = SelectiveNet_vgg16_bn()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = OverAllLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # training
    batch_size = 128
    train_dataloader = load_data('train', batch_size, num_worker)
    best_model = train(model, train_dataloader, criterion, optimizer, scheduler, pkl_name, num_epochs=300)