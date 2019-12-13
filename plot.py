"""
   plot real-time loss-acc curve and ROC curve
"""

import keras
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
         self.losses = {'batch': [], 'epoch':[]}
         self.accuracy = {'batch': [], 'epoch':[]}
         self.val_loss = {'batch':[],'epoch':[]}
         self.val_acc = {'batch':[],'epoch':[]}

    def on_batch_end(self,batch, logs={}):
          self.losses['batch'].append(logs.get('loss'))
          self.accuracy['batch'].append(logs.get('acc'))
          self.val_loss['batch'].append(logs.get('val_loss'))
          self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self,batch,logs={}):
           self.losses['epoch'].append(logs.get('loss'))
           self.accuracy['epoch'].append(logs.get('acc'))
           self.val_loss['epoch'].append(logs.get('val_loss'))
           self.val_acc['epoch'].append(logs.get('val_acc'))

    def  loss_plot(self,loss_type):
            iters=range(len(self.losses[loss_type]))
            plt.figure()
            plt.plot(iters,self.accuracy[loss_type],'r',label='train acc')
            plt.plot(iters,self.losses[loss_type],'g',label='train loss')
            if loss_type == 'epoch':
                plt.plot(iters, self.val_acc[loss_type],'b',label='val acc')
                plt.plot(iters, self.val_loss[loss_type],'k',label='val loss')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('acc-loss')
            plt.legend(loc="upper right")
            plt.savefig('./figure/acc-loss.png')

def plotROC(test,score):
    fpr,tpr,threshold = roc_curve(test, score)
    plt.figure()
    lw = 2
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='Independent dataset')
    # plt.plot(fpr, tpr, color='darkorange',lw=lw, label='P.syringae dataset')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.text(-0.1,1.1,'(a)',fontsize=15)
    plt.xlabel('False Positive Rate',fontsize=12)
    plt.ylabel('True Positive Rate',fontsize=12)
    # plt.tick_params(axis='both',labelsize=20)
    plt.title('Receiver operating characteristic curve',fontsize=12)
    plt.legend(loc="lower right")
    plt.savefig('./figure/ROC.png')
    
    