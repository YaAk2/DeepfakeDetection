from random import shuffle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss as cross_entropy_loss
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

conf_mat_labels = [1, 2, 3, 4, 5]
labels = ['Attribute Manipulation', 'Expression Swap', 'Entire Face Synthesis', 'Identity Swap', 'Real']

class Solver(object):
    default_adam_args = {"lr": 1e-1,
                         "alpha": 0.99,
                         "eps": 1e-8,
                         "weight_decay": 0.0,
                         "momentum": 0.0,
                         "centered": False}

    def __init__(self, optim=torch.optim.RMSprop, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss(), reg=False):
                
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self._reset_histories()
        
        self.reg = reg
        if self.reg==True:
            loss_func = loss_func=torch.nn.CrossEntropyLoss(weight=torch.Tensor([1., 1., 1., 1., 5.]).cuda())
        

    def _reset_histories(self):
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        
        best_val_acc = 0.0
        model.train()
        for epoch in range(num_epochs):  
            ############
            # Training #
            ############       
            running_tr_loss = 0.0
            running_tr_acc = 0.0
            counter = 0
            for batch_idx, (X_tr_batch, Y_tr_batch) in enumerate(train_loader):
                counter+=1
                if torch.cuda.is_available():
                    X_tr_batch, Y_tr_batch = X_tr_batch.cuda(), Y_tr_batch.cuda()
                
                ## zero the parameter gradients
                optim.zero_grad()
                
                output_tr = model(X_tr_batch)
                train_loss = self.loss_func(output_tr, Y_tr_batch)
                
                # computes dloss/dw for every parameter w which has requires_grad=True
                train_loss.backward()
                
                # updates the parameters
                optim.step()
                
                # print statistics
                if log_nth == counter:
                    print("[Iteration %d/%d] TRAIN loss: %.6f" % (batch_idx + 1, iter_per_epoch, train_loss.item()), end="\r")
                    counter = 0
                
                running_tr_loss += train_loss.item()
                pred = np.argmax(output_tr.cpu().detach().numpy(), axis=1)
                running_tr_acc += np.sum(pred == Y_tr_batch.cpu().detach().numpy()).item()/train_loader.batch_size
                
                del X_tr_batch, Y_tr_batch, output_tr, train_loss
                
                    
            # save loss for plotting
            self.train_loss_history.append(running_tr_loss/len(train_loader))

            # save accuracy for plotting
            self.train_acc_history.append(running_tr_acc/len(train_loader))
            
            ##############
            # Validation #
            ##############
            model.eval()
            fp = []
            running_val_loss = 0.0
            running_val_acc = 0.0
            conf_mat = np.zeros((len(conf_mat_labels), len(conf_mat_labels)))
            for batch_idx, (X_val_batch, Y_val_batch) in enumerate(val_loader):
                if torch.cuda.is_available():
                    X_val_batch, Y_val_batch = X_val_batch.cuda(), Y_val_batch.cuda()
                output_val = model(X_val_batch)
                val_loss = self.loss_func(output_val, Y_val_batch).cpu().detach()
                running_val_loss += val_loss.item()
                pred = np.argmax(output_val.cpu().detach().numpy(), axis=1)
                running_val_acc += np.sum(pred == Y_val_batch.cpu().detach().numpy()).item()/val_loader.batch_size
                
                # calculate confusion matrix 
                # and create a list of false positives (image, true label, predicted label)
                if epoch == num_epochs-1:
                    conf_mat = conf_mat + confusion_matrix(Y_val_batch.cpu().detach().numpy() + 1, pred + 1, labels=conf_mat_labels)
                    fp_idx = np.where([pred != Y_val_batch.cpu().detach().numpy()])[1]
                    [fp.append((torchvision.transforms.ToPILImage()(X_val_batch[i].cpu().detach()), 
                                labels[Y_val_batch[i].cpu().detach().item()], 
                                labels[pred[i]])) for i in fp_idx]
                
                del X_val_batch, Y_val_batch, output_val
                
                
            # save loss for plotting
            self.val_loss_history.append(running_val_loss/len(val_loader))
            
            # save accuracy for plotting
            self.val_acc_history.append(running_val_acc/len(val_loader))
          
            
            print('[Epoch %d/%d] TRAIN acc/loss: %.6f/%.6f' % (epoch + 1, num_epochs, 
                                                               running_tr_acc/len(train_loader),
                                                               running_tr_loss/len(train_loader)))
            print('[Epoch %d/%d] VAL acc/loss: %.6f/%.6f' % (epoch + 1, num_epochs, 
                                                                running_val_acc/len(val_loader), 
                                                                running_val_loss/len(val_loader)))
            
            if epoch == num_epochs-1:
                print('\n')
                print('Confusion matrix: \n', conf_mat)
                print('\n')
            
            
            current_val_acc = running_val_acc/len(val_loader)
            if current_val_acc > best_val_acc:
                model.save()
                best_val_acc = current_val_acc     
                
        print('FINISH.')
        
        return (self.train_loss_history, self.train_acc_history, self.val_loss_history, self.val_acc_history, conf_mat, fp)