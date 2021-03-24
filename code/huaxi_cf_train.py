'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

import torch
from torch import nn
from torch import optim
import glob
import numpy as np
import logger
from huaxi_cf_dataset import MyDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from huaxi_cl.model import mobilenet_v2
from sklearn import metrics as mt
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0, 1"

def train(alexnet_model, train_loader, epoch, train_dict, logger,criterion,use_gpu):
    alexnet_model.train()
    losss = 0
    for iter, batch in enumerate(train_loader):
        torch.cuda.empty_cache()
        if use_gpu:
            inputs = Variable(batch[0].cuda())
            cf = Variable(batch[1].cuda())
            labels = Variable(batch[2].cuda())
        else:
            inputs, cf, labels = Variable(batch['0']), Variable(batch['1']), Variable(batch['2'])

        label_fla = labels.cpu().numpy()
        label_fla = label_fla.flatten()
        label_fla = label_fla[1::2]
        if np.sum(label_fla) < 3:
            continue
        optimizer.zero_grad()
        outputs = alexnet_model(inputs, cf)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losss = losss + loss.item()

        print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, iter, len(train_loader),
                    100. * iter / len(train_loader), losss / (iter+1)))



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001 * (0.1 ** (epoch // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def ODIR_Metrics(pred,target):
    th = 0.5

    gt = target.flatten()
    pr = pred.flatten()
    gt1 = gt[0::2]
    gt2 = gt[1::2]
    pr1 = pr[0::2]
    pr2 = pr[1::2]
    kappa = mt.cohen_kappa_score(gt, pr > th)
    print("1：auc值,", mt.roc_auc_score(gt1, pr1), 'acc:',mt.accuracy_score(gt1, pr1 > th))
    print("2：auc值,", mt.roc_auc_score(gt2, pr2), 'acc:',mt.accuracy_score(gt2, pr2 > th))
    #f1 = mt.f1_score(gt, pr > th, average='micro')
    auc = mt.roc_auc_score(gt2, pr2)
    f1 = mt.accuracy_score(gt, pr > th)
    final_score = (kappa+f1+auc) / 3.0
    return auc

def val_test(alexnet_model, val_loader):
    alexnet_model.eval()
    accs = 0

    val_loss = 0
    with torch.no_grad():
        p = []
        g = []
        for iter, batch in enumerate(val_loader):
            torch.cuda.empty_cache()
            if use_gpu:
                inputs = Variable(batch[0].cuda())
                cf = Variable(batch[1].cuda())
                labels = Variable(batch[2].cuda())
            else:
                inputs, cf, labels = Variable(batch['0']), Variable(batch['1']), Variable(batch['2'])
            outputs = alexnet_model(inputs, cf)
            loss = criterion(outputs, labels)
            outputs = torch.softmax(outputs, dim=1)
            outputs = outputs.data.cpu().numpy()
            labels = labels.cpu().numpy()
            for x, y in zip(outputs, labels):
                p.append(x)
                g.append(y)
            val_loss += loss.item()
        auc = ODIR_Metrics(np.array(p), np.array(g))
    val_loss /= len(val_loader)
    print('\nVal set: Average loss: {:.6f},auc: {:.6f}\n'.format(val_loss, auc))
    return auc




class WeightedMultilabel(torch.nn.Module):

    def __init__(self, weights: torch.Tensor):
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.weights = weights.unsqueeze()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets) * self.weights





if __name__ == "__main__":
    batch_size = 32
    epochs = 500
    lr = 0.001
    momentum = 0.99
    w_decay = 1e-6
    step_size = 50
    gamma = 0.5
    n_class = 2
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    train_path = '/media/stianci/360b04be-f526-4e91-aad3-85535b17af3e/3d_data/huaxi/train/*.npy'
    test_path = '/media/stianci/360b04be-f526-4e91-aad3-85535b17af3e/3d_data/huaxi/test/*.npy'
    train_path = glob.glob(train_path)

    leng = len(train_path)
    vail_path = train_path[:int(leng/5)]
    train_path = train_path[int(leng/5):]
    test_path = glob.glob(test_path)
    train_da = MyDataset(train_path, transform=True)
    test = MyDataset(test_path, transform=False)
    vail = MyDataset(vail_path, transform=False)
    train_loader = DataLoader(train_da, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(vail, batch_size=batch_size, shuffle=False, num_workers=2)

    print('model load...')
    model_dir = "/media/stianci/360b04be-f526-4e91-aad3-85535b17af3e/twx/pywork/pytorch/3D_class/mobilenet/huaxi/models_cf4"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    #model_path_best = os.path.join(model_dir, 'res50.pth')
    model_path = os.path.join(model_dir, 'mobilev2.pth')

    model = mobilenet_v2()
    # model = vgg16_bn()
    #model = resnet50(sample_input_D=16, sample_input_H=96, sample_input_W=96, num_seg_classes=2)
    #model = _3DCNN()

    if use_gpu:
        alexnet_model = model.cuda()
        alexnet_model = nn.DataParallel(alexnet_model, device_ids=num_gpu)
    # print(model)
    # exit()
    pos_weight = torch.FloatTensor([0.6, 3]).cuda() # 0.67
    #pos_weight = torch.FloatTensor([0.6, 3]).cuda()
    criterion = nn.BCELoss(weight=pos_weight)
    #criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #criterion = WeightedMultilabel(weights=pos_weight)
    #optimizer = optim.Adam(alexnet_model.parameters(), lr=lr, betas=(0.9, 0.99))
    optimizer = optim.SGD(alexnet_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    # create dir for score
    score_dir = os.path.join(model_dir, 'scores')
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)
    train_dict = {'loss': []}
    val_dict = {'loss': [], 'auc': []}
    logger = logger.Logger('/media/stianci/360b04be-f526-4e91-aad3-85535b17af3e/twx/pywork/pytorch/3D_class/mobilenet/huaxi/log_cf2')
    best_loss = 0
    for epoch in range(1, 5000 + 1):
        adjust_learning_rate(optimizer, epoch)
        #train(alexnet_model, train_loader, epoch, train_dict, logger, criterion, use_gpu)
        print("------------------------", epoch, '------------------------------')
        # print("------------------------", 'auc_vail', '------------------------------')
        # auc_vail = val_test(alexnet_model,  val_loader)
        print("------------------------", 'auc_test', '------------------------------')
        auc_test = val_test(alexnet_model,  val_loader)
        if epoch > 20:
            model_path = os.path.join(model_dir, str(auc_test)[:4] + '_train_mobilev2.pth')
            torch.save(alexnet_model, model_path)
