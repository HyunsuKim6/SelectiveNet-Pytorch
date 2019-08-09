from __future__ import print_function, division

from torch.optim import lr_scheduler

import time
from tqdm import tqdm
from SelectiveNet import *
from LoadingData import *


def test(model, test_dataloader, batch_size, pkl_name):
    # device setting
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(torch.cuda.get_device_name(0))

    model.load_state_dict(torch.load(pkl_name))
    model.eval()

    since = time.time()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    total_classes_correct = 0.
    total_classes_total = 0.
    test_coverage = 0
    first_time_flag = 0

    with torch.no_grad():
        for data in tqdm(test_dataloader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            pred_outputs, sel_outputs, aux_outputs = model(inputs)
            _, predicted = pred_outputs.max(1)

            for i in range(batch_size):
                if sel_outputs[i].item() >= 0.5:
                    if labels[i].item() == predicted[i].item():
                        class_correct[labels[i].item()] += 1
                    class_total[labels[i].item()] += 1

            if first_time_flag == 0:
                total_sel_outputs = sel_outputs
                first_time_flag += 1

            else:
                total_sel_outputs = torch.cat((total_sel_outputs, sel_outputs), 0)

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        total_classes_correct += class_correct[i]
        total_classes_total += class_total[i]

    print('Total Accuracy : %2d %%' % (100 * total_classes_correct / total_classes_total))
    test_coverage = total_sel_outputs.mean()
    print('Test Coverage : %2d %%' % (100 * test_coverage))

    time_elapsed = time.time() - since

    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return


if __name__ == "__main__":
    num_worker = 12
    pkl_name = 'checkpoints/SelectiveNet.pkl' # for testing

    model = SelectiveNet_vgg16_bn()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = OverAllLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # testing
    batch_size = 100
    test_dataloader = load_data('test', batch_size, num_worker)
    test(model, test_dataloader, batch_size, pkl_name)