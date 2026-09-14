import numpy as np
import os
import scipy.io
import time
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection  import ParameterGrid
# "model" is imported inside GridSearch -- importing it here would be circular,
# because model.py imports this module.
import math

def LoadBatch(dataset):
    """
    Loads a single batch
    Parameters:
        dataset: the batch to be loaded
    Returns:
        data: images
        one_hot: labels in one-hot encode format
    """ 
    data_path = os.path.join('./Dataset/cifar-10-batches-mat/', dataset)
    mat = scipy.io.loadmat(data_path)
    data = mat['data'].T
    #obtain from the labels the one hot encoding
    labels = mat['labels']
    one_hot = np.zeros((10, labels.shape[0]))
    for i in range(labels.shape[0]):
        one_hot[labels[i], i] = 1
    # flatten labels to a vector
    labels = labels.flatten()
    return data, one_hot

def LoadData():
    """
    Loads all the dataset. 45000 images for training and 5000 for validation
    Returns:
        Xi: images for train/validation/test
        Yi: labels in one-hot encode format for train/validation/test
    """ 
    for i in range(1,6):
        if i==1:
            X,Y = LoadBatch('data_batch_1.mat')
            X_train,Y_train = X[:,:9000],Y[:,:9000]
            X_val,Y_val     = X[:,9000:],Y[:,9000:]
        elif 2<=i<=5:
            X,Y = LoadBatch('data_batch_'+str(i)+'.mat')
            X_train = np.concatenate((X_train, X[:,:9000]), axis=1)
            Y_train = np.concatenate((Y_train, Y[:,:9000]), axis=1)
            X_val = np.concatenate((X_val, X[:,9000:]), axis=1)
            Y_val = np.concatenate((Y_val, Y[:,9000:]), axis=1)
    X_test,Y_test = LoadBatch('test_batch.mat')
    return X_train,X_val,X_test,Y_train,Y_val,Y_test

def LoadBatch_1_2():
    """
    Loads the first two batches
    Returns:
        Xi: images for train/validation/test
        Yi: labels in one-hot encode format for train/validation/test
    """ 
    X_train,Y_train = LoadBatch('data_batch_1.mat')
    X_val,Y_val = LoadBatch('data_batch_2.mat')
    X_test,Y_test = LoadBatch('test_batch.mat')
    return X_train,X_val,X_test,Y_train,Y_val,Y_test

def zero_mean(X_train,X_val,X_test):
    """
    Zero mean and divide by std all the data row-wise with the info from train data
    Parameters:
        X_train,X_val,X_test: images from different sets
    Returns:
        X_train,X_val,X_test after the operations
    """ 
    mean = np.mean(X_train, axis=1)
    std = np.std(X_train, axis=1)
    mean = mean.reshape(mean.shape[0], 1)
    std = std.reshape(std.shape[0], 1)
    
    X_train = (X_train - mean) / std
    X_val =  (X_val  - mean) / std
    X_test =  (X_test  - mean) / std

    return X_train , X_val, X_test

def flip_augment_images(X,Y):
    """
    Given a set of images and their labels, rotate and augmentate them
    Parameters:
        X:images
        Y:labels
    Returns:
        data_aug: images augmented and rotated
        label_aug: labels of the new set of images
    """ 
    n = X.shape[-1]
    indices = []
    for i in range(n):
        k = random.randint(0, 1)
        if k == 0:
            indices.append(i)

    X = X.reshape((3,32,32,n))
    new_im = []
    new_lb = []
    for ind in indices:   
        new_im.append(np.fliplr(X[:,:,:,ind]))
        new_lb.append(Y[:,ind])

    aug = np.asarray(new_im).transpose(1,2,3,0)
    n1 = aug.shape[-1]
    data_aug = np.empty((X.shape[0],X.shape[1],X.shape[2],n+n1))
    data_aug[:,:,:,:n]=X
    data_aug[:,:,:,n:]=aug

    label_aug = np.empty((Y.shape[0],n+n1))
    label_aug[:,:n]=Y
    label_aug[:,n:]=np.asarray(new_lb).T
    data_aug = data_aug.reshape((3*32*32,n+n1))
    return data_aug, label_aug

def get_kfold_data(i, datasets, k=5):
    """
    Get a k-fold from a datset
    Parameters:
        i: number of fold to obtain
        dataset: whole dataset
        k: number of folds in total
    Returns:
        trainset: train set for fold i
        validset: validation set for fold i
    """ 
    fold_size = len(datasets) // k  
    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]
    return trainset, validset

def shuffle_dataset(dataset, seed):
    ''' Function that shuffles the datset'''
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    ''' Function that split the datset by a given ratio'''
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def load_state(timestr):
    ''' Load all the data from a folder with the parameter given timestr'''
    dir = './results/'+timestr+'/'
    W = np.load(dir+'W.npy') 
    cost_train = np.load(dir+'cost_train.npy') 
    cost_val = np.load(dir+'cost_val.npy') 
    loss_train = np.load(dir+'loss_train.npy') 
    loss_val = np.load(dir+'loss_val.npy') 
    # container = np.load('mat.npz')
    # data = [container[key] for key in container]
    return  W,cost_train,cost_val,loss_train,loss_val

def plot_prob(self,timestr,X,Y,W_star,b_star):
    ''' Choose correct and incorrect classified examples and for each set do a hist of the prob of the gt class, 
    which is, for each class, how many times the prob is 0.1,0.2,0.3,...'''
    dir = './results/'+timestr+'/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    probs = self.EvaluateClassifier(X,W_star,b_star)
    pred = np.argmax(probs, axis=0)
    gt = np.argmax(Y,axis=0)
    correct = (gt==pred)

    true_pred = []
    false_pred = []       
    for i in range(correct.shape[0]): 
        if correct[i]==True: 
            true_pred.append(probs[pred[i],i])
        else: 
            false_pred.append(probs[gt[i],i])

    plt.title('Correctly/Incorrectly classified samples')
    plt.xlabel("Probability of ground truth")
    plt.ylabel('Frequency')
    plt.hist(true_pred,70,alpha=0.6,label = 'Correct samples')
    plt.hist(false_pred,70,alpha=0.6,label = 'Incorrect samples')
    plt.legend()
    plt.savefig(dir+ 'prediction.png')
    plt.show()
    return None
 
def plot_data(data,title,xlabel,ylabel,dir=None,save=True,legend=False,label=None):
    ''' Function used to plot. Give the data in a list format '''
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend == True:
        for i in range(len(data)):
            plt.plot(data[i], label = label[i])
            plt.legend()
    else:
        for i in range(len(data)):
            plt.plot(data[i])

    if save==True:
        plt.savefig(dir)
    plt.show()
    return None


def save_state(history):
    ''' Plot the results of cost/loss, accuracy, lr rate and weights, Save them to a folder '''
    dir, W,b,Modelparams,loss_train,loss_val,cost_train,cost_val,train_acc,val_acc,test_acc,best_val_acc,etas = history
    # np.savez('data.npz', name1=W, name2=b,name3=cost_train,name4=cost_val,name5=loss_train,name6=loss_val,name7=test_acc,name8=val_acc)
  
    plot_data([etas],'Eta','t','value',dir+'lr.png')
    plot_data([train_acc,val_acc],'Accuracy: train and validation',\
              'Epochs','Accuracy',dir+'acc.png',legend=True,label=['train_acc','val_acc'])
    plot_data([cost_train,cost_val,loss_train,loss_val],'Cost, Loss: train and validation',\
              'Epochs','Cost and Loss',dir+'cost-loss.png',legend=True,label=['cost train','cost val','loss train','loss val'])

    """ Display the image for each label in W """
    fig, ax = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            im  = W[0][i*5+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    plt.savefig(dir+'weights.png')
    plt.show()
    return None

def GridSearch(param_grid,X_train,Y_train,X_val,Y_val,X_test,Y_test,optimizer,lr_scheduler):
    ''' Given a dictionary with a set of parameters and the dataset, do grid search '''
    grid = ParameterGrid(param_grid)
    root = time.strftime("%Y%m%d-%H%M%S")
    path ='./results/'+'grid-'+root+'/' 
    if not os.path.exists(path):
        os.makedirs(path)

    all_acc = [] # to find later the setting with highest acc
    for i,params in enumerate(grid):
        print('Fit number ',i,'/',len(grid))
        print('---------------------------------------')
        for key in params:
            print(key,':',params[key])  
        print('---------------------------------------')

        date = time.strftime("%Y%m%d-%H%M%S")
        dir = path + date+'/'
        if not os.path.exists(dir):
            os.makedirs(dir)

        D,K = X_train.shape[0], Y_train.shape[0]
        n_hidden=params['n_hidden']
        from model import model
        m = model(n_hidden,D,K)

        history = m.train_model(X_train,Y_train,X_val,Y_val,X_test,Y_test,optimizer,lr_scheduler,\
                                landa=params['landa'],dir=dir, batchnorm=params['batchnorm'])      
        save_state(history)
        best_val_acc = history[-2]
        all_acc.append(best_val_acc)
        # Extension: do plot of validation acc vs tuned hyperparameters (for each hyperparameter)
    
    best = np.argmax(np.array(all_acc))
    print('The best combination is the number',best,':',grid[best], 'with an accuracy of ',all_acc[best])
    with open(path+'info.txt', 'a') as f:
        f.write('The best combination is the setting '+str(best)+':'+str(grid[best])+ ' with an accuracy of '+str(all_acc[best]))
    return None  

class optimizer:
    def __init__(self,opt,beta1=.9,beta2=.999,Adam_eps=1e-8):
        self.opt = opt
        if self.opt == 'Adam':
            self.beta1,self.beta2,self.Adam_eps = beta1,beta2,Adam_eps
            self.append_gradients = True

    def update_grads(self,t,eta,grads,params):
        if self.opt == 'Adam':
            params = self.update_Adam(t,eta,grads,params)
        elif self.opt == 'SGD':
            params = self.update_SGD(eta,grads,params)
        return params

    def update_SGD(self,eta,grads,params):
        for j in range(len(params)):
            for k in range(len(params[j])):
                params[j][k] = params[j][k] - eta*grads[j][k]    
        return params
    
    def update_Adam(self,t,eta,grads,params):
        if self.append_gradients:
            self.append_gradients = False
            self.m,self.v,self.m_hat,self.v_hat = [],[],[],[]
            for j in range(len(params)):
                self.m.append([]); self.v.append([]); self.m_hat.append([]);  self.v_hat.append([])
                for k in range(len(params[j])):
                    self.m[j].append( self.beta1*0 + (1-self.beta1)*grads[j][k] )
                    self.v[j].append( self.beta2*0 + (1-self.beta2)*grads[j][k]**2 )
                    self.m_hat[j].append( self.m[j][k]/(1-self.beta1**t) ) 
                    self.v_hat[j].append( self.v[j][k]/(1-self.beta2**t) )
                    params[j][k] = params[j][k] - eta*self.m_hat[j][k]/(np.sqrt(self.v_hat[j][k])+self.Adam_eps)
        else:
            for j in range(len(params)):
                for k in range(len(params[j])):
                    self.m[j][k] = self.beta1*self.m[j][k] + (1-self.beta1)*grads[j][k]
                    self.v[j][k] = self.beta2*self.v[j][k] + (1-self.beta2)*grads[j][k]**2
                    self.m_hat[j][k] = self.m[j][k]/(1-self.beta1**t)
                    self.v_hat[j][k] = self.v[j][k]/(1-self.beta2**t)
                    params[j][k] = params[j][k] - eta*self.m_hat[j][k]/(np.sqrt(self.v_hat[j][k])+self.Adam_eps)
        return params

    def param_info(self):
        if self.opt == 'Adam':
            info = {'optimizer':self.opt,'beta1':self.beta1,'beta2':self.beta2,'Adam_eps':self.Adam_eps}
        elif self.opt == 'SGD':
            info = {'optimizer':self.opt}
        return info


class lr_scheduler:
    def __init__(self,lr_sch,eta=.001,decay=.1,decay_epochs_freq = 2,eta_min=1e-5,eta_max=.1,n_cycles=1):
        self.lr_sch = lr_sch
        if self.lr_sch == 'decay':
            self.eta = eta
            self.decay = decay
            self.decay_epochs_freq = decay_epochs_freq  
        elif self.lr_sch == 'cyclic':
            self.eta_min = eta_min
            self.eta_max = eta_max
            self.eta = eta_min
            self.n_cycles = n_cycles

    def ini_scheduler(self,n_samples,n_batch):
        if self.lr_sch == 'cyclic':
            ''' Every n_step we have a max or a min. 2 steps are a cycle 
            The number of epochs are according to the n_cycles '''
            self.l,self.k = 0,2
            self.n_samples,self.n_batch= n_samples,n_batch
            self.n_step = 5*45000/self.n_batch # math.floor(2*self.n_samples/self.n_batch) 
            self.epochs = math.ceil(self.n_cycles*2*self.n_step/(self.n_samples/self.n_batch)) 
        return self.eta,self.epochs

    def update_lr(self,epoch,batch_step,t):
        if self.lr_sch == 'decay':
            if epoch!=0 and batch_step==1:
                if (epoch%self.decay_epochs_freq==0):
                    self.eta = self.eta*self.decay
        elif self.lr_sch == 'cyclic':
            ''' t increases every batch and epoch: final_t=epochs*batch '''
            ''' Every n_step a min/max is reached and the lr start increasing/decreasing '''
            if (t)%(2*self.n_step) == 0:
                self.l +=1
                self.eta_max *= 0.8
            if (2*self.l*self.n_step)<=t<=((2*self.l+1)*self.n_step):
                self.eta = self.eta_min + (self.eta_max-self.eta_min)*(t-2*self.l*self.n_step)/(self.n_step)
            elif ((2*self.l+1)*self.n_step)<=t<=(2*(self.l+1)*self.n_step):
                self.eta = self.eta_max - (self.eta_max-self.eta_min)*(t-(2*self.l+1)*self.n_step)/(self.n_step)

        return self.eta
    
    def param_info(self):
        if self.lr_sch == 'decay':
            info = {'scheduler':self.lr_sch,'eta':self.eta,'decay':self.decay,'decay_epochs_freq':self.decay_epochs_freq}
        elif self.lr_sch == 'cyclic':
            info = {'scheduler':self.lr_sch,'eta_min':self.eta_min,'eta_max':self.eta_max,'n_cycles':self.n_cycles}
        return info
