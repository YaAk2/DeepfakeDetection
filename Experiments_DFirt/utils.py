import numpy as np
import matplotlib.pyplot as plt
import json
import os
from os.path import join
import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
from facenet_pytorch import MTCNN
from data_utils import DFirt

conf_mat_labels = [1, 2, 3, 4, 5]
labels = ['Attribute Manipulation', 'Expression Swap', 'Entire Face Synthesis', 'Identity Swap', 'Real']

def plt_conf_mat(conf_mat):
    conf_mat_norm = (conf_mat.T/np.sum(conf_mat, axis=1)).T # normalize confusion matrix
    fig, ax = plt.subplots()
    ax.set_title('True vs Predicted')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    
    [ax.text(j, i, format(conf_mat_norm[i, j], '.2g'), ha="center", va="center", color="w") 
     for i in range(len(labels)) for j in range(len(labels))]
 
    ax.imshow(conf_mat_norm)
    
def plt_fp(fp):
    pass

# https://medium.com/datadriveninvestor/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
def saliency_map(model, im_path):
    t = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img = Image.open(im_path)
    im = t(img).unsqueeze(0).cuda()
    
    model.eval()
    im.requires_grad_()
    
    pred = model(im)
    pred_max_idx = pred.argmax()
    pred_max = pred[0, pred_max_idx]
    pred_max.backward()
    
    saliency, _ = torch.max(im.grad.data.abs(), dim=1)

    overlay = saliency[0].cpu()
    plt.imshow(img.resize((overlay.shape)))
    plt.imshow(overlay, alpha = 0.8)
    plt.axis('off')
    print('Type of manipulation: ', labels[pred_max_idx])
    plt.show()
    
def evaluation_DFFD(path, model, loss_func):
    test = DFirt(path)
    test_loader = test.data_loader(batch_size=128)
    model.eval()
    
    test_loss_history = []
    test_acc_history = []
    fp = []
    running_test_loss = 0.0
    running_test_acc = 0.0
    conf_mat = np.zeros((len(conf_mat_labels), len(conf_mat_labels)))
    for batch_idx, (X_test_batch, Y_test_batch) in enumerate(test_loader):
        if torch.cuda.is_available():
            X_test_batch, Y_test_batch = X_test_batch.cuda(), Y_test_batch.cuda()
        output_test = model(X_test_batch)
        test_loss = loss_func(output_test, Y_test_batch).cpu().detach()
        running_test_loss += test_loss.item()
        pred = np.argmax(output_test.cpu().detach().numpy(), axis=1)
        running_test_acc += np.sum(pred == Y_test_batch.cpu().detach().numpy()).item()/test_loader.batch_size

        # calculate confusion matrix 
        # and create a list of false positives (image, true label, predicted label)
        conf_mat += confusion_matrix(Y_test_batch.cpu().detach().numpy() + 1, pred + 1, labels=conf_mat_labels)
        fp_idx = np.where([pred != Y_test_batch.cpu().detach().numpy()])[1]
        [fp.append((torchvision.transforms.ToPILImage()(X_test_batch[i].cpu().detach()), 
                    labels[Y_test_batch[i].cpu().detach().item()], 
                    labels[pred[i]])) for i in fp_idx]
        
        del X_test_batch, Y_test_batch, output_test

                     
    print('TEST acc/loss: %.6f/%.6f' % (running_test_acc/len(test_loader), running_test_loss/len(test_loader)))
    print('\n')
    print('Confusion matrix: \n', conf_mat)
    print('\n')    
    print('FINISH.')
        
    return (conf_mat, fp)

'''
def store_fp(model):
    model.eval()
    
    path = '../DFirt/val/'
    dataset = ['attribute_manipulation', 'expression_swap', 'face_synthesis', 'identity_swap', 'real']
    
    if os.path.exists('FalsePrediction') == False:
        os.mkdir('FalsePrediction')
    for d in dataset:
        os.chdir('FalsePrediction')
        if os.path.exists(labels[dataset.index(d)]) == False:
            os.mkdir(labels[dataset.index(d)])
        os.chdir('..')
        for im_path in sorted(os.listdir(path + d)):
            img = Image.open(path + d + '/' + im_path)
            t = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
            im = t(img).unsqueeze(0).cuda()
            pred = model(im)
            pred_max_idx = pred.argmax()
            os.chdir('FalsePrediction')
            os.chdir(labels[dataset.index(d)])
            if os.path.exists(labels[pred_max_idx]) == False:
                os.mkdir(labels[pred_max_idx])
            os.chdir('..')
            os.chdir('..')
            if pred_max_idx != dataset.index(d):
                img.save('FalsePrediction/' + labels[dataset.index(d)] + '/' + labels[pred_max_idx] + '/' + im_path)    
'''    

def evaluation_ffpp(data, model, submission):
    model.eval()
    
    detector = MTCNN(image_size=256, post_process=False)
 
    f = open(submission, 'r+')
    im = json.load(f)
    f.close()    
    
    for im_path in sorted(os.listdir(data)):
        img = Image.open(join(data, im_path))
        box, probs = detector.detect(img)
        if probs[0]:
            b = np.absolute(np.floor(box[0]).astype(dtype=np.int))
            img = img.crop(b)
            t = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
            img = t(img).unsqueeze(0).cuda()
            pred = model(img)
            pred_max_idx = pred.argmax()

            if pred_max_idx == 4: # Real
                im[im_path] = 'real'
            else:
                im[im_path] = 'fake'
        
            f = open(submission, 'w+')
            f.write(json.dumps(im, indent=4))
            f.close()