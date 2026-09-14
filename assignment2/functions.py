import numpy as np
import pickle
import os
import scipy.io
import numpy.matlib
from tqdm import tqdm
import time
import os
import random
import matplotlib.pyplot as plt


def LoadBatch(train_set, validation_set, test_set):

    datasets = [train_set, validation_set, test_set]
    data_dir = './Dataset/cifar-10-batches-mat/'
    cont = 0
    for dataset in datasets:
        data_path = os.path.join(data_dir, dataset)
        mat = scipy.io.loadmat(data_path)
        # data = mat['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
        # data = np.transpose(data,(0, 2, 1, 3))
        data = mat['data'].T
        # print(data.T.shape)
        #obtain from the labels the one hot encoding
        labels = mat['labels']
        one_hot = np.zeros((10, labels.shape[0]))
        
        for i in range(labels.shape[0]):
            one_hot[labels[i], i] = 1

        # flatten labels to a vector
        labels = labels.flatten()
        # print('labels: ', labels.shape)
        # Create a dictionary that has the data and labels
        if cont == 0:
            train= {'data': data, 'labels':labels,'one_hot': one_hot}
        elif cont == 1:
            validation = {'data': data, 'labels': labels, 'one_hot': one_hot}
        else:
            test = {'data': data, 'labels': labels, 'one_hot': one_hot}

        cont +=1

    return train, validation, test

def LoadData():

    data_dir = './Dataset/cifar-10-batches-mat/'
    
    # Load the training data
    data_path = os.path.join(data_dir, 'data_batch_'+str(1)+'.mat')
    mat = scipy.io.loadmat(data_path)
    data = mat['data'].T
    labels = mat['labels']
    one_hot = np.zeros((10, labels.shape[0]))
    for i in range(labels.shape[0]):
        one_hot[labels[i], i] = 1
    labels = labels.flatten()

    for j in range(2,6):
        data_path = os.path.join(data_dir, 'data_batch_'+str(2)+'.mat')
        mat = scipy.io.loadmat(data_path)
        data = np.concatenate((data, mat['data'].T[:,:9000]), axis=1) 
        labels = np.concatenate((labels, mat['labels'].T.flatten()[:9000]))   
        one_hot = np.zeros((10, labels.shape[0]))
        for i in range(labels.shape[0]):
            one_hot[labels[i], i] = 1
        labels = labels.flatten()
    
    train= {'data': data, 'labels':labels,'one_hot': one_hot}
    
    # Load the validation data
    data_path = os.path.join(data_dir, 'data_batch_'+str(5)+'.mat')
    mat = scipy.io.loadmat(data_path)
    data = mat['data'].T[:,9000:]
    labels = mat['labels'][9000:]
    one_hot = np.zeros((10, labels.shape[0]))
    for i in range(labels.shape[0]):
        one_hot[labels[i], i] = 1
    labels = labels.flatten()

    validation= {'data': data, 'labels':labels,'one_hot': one_hot}

    # Load the test data
    data_path = os.path.join(data_dir, 'test_batch.mat')
    mat = scipy.io.loadmat(data_path)
    data = mat['data'].T
    labels = mat['labels']
    one_hot = np.zeros((10, labels.shape[0]))
    for i in range(labels.shape[0]):
        one_hot[labels[i], i] = 1
    labels = labels.flatten()

    test= {'data': data, 'labels':labels,'one_hot': one_hot}
    return train, validation, test

def zero_mean(X_train,X_val,X_test):
    """ Zero mean the data row-wise """
    mean = np.mean(X_train, axis=1)
    std = np.std(X_train, axis=1)
    mean = mean.reshape(mean.shape[0], 1)
    std = std.reshape(std.shape[0], 1)
    
    X_train = (X_train - mean) / std
    X_val =  (X_val  - mean) / std
    X_test =  (X_test  - mean) / std

    return X_train , X_val, X_test

def flip_images_2(X):
    n = X.shape[-1]
    indices = []
    for i in range(n):
        k = random.randint(0, 1)
        if k == 0:
            indices.append(i)

    X = X.reshape((3,32,32,n))
    for ind in indices:
        X[:,:,:,ind] =  np.fliplr(X[:,:,:,ind])
    X = X.reshape((3*32*32,n))
    return X

def flip_images(X,Y):
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
    # print('Y,',Y.shape)
    # print('new_lb',np.asarray(new_lb).shape)
    # print('label aug', label_aug.shape)

    # print('X,',X.shape)
    # print('aug',aug.shape)
    # print('dataaug',data_aug.shape)
    data_aug = data_aug.reshape((3*32*32,n+n1))
    # print('dataaug',data_aug.shape)
    return data_aug, label_aug

def get_kfold_data(i, datasets, k=5):
    
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
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


class model:
    def __init__(self,X_train,X_val,X_test,Y_train,Y_val,Y_test):

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.Y_train = Y_train
        self.Y_val = Y_val
        self.Y_test = Y_test

        self.D = self.X_train.shape[0]
        self.K = self.Y_train.shape[0]
        self.N = self.X_train.shape[1]

        self.landa = 0
        self.initialise_weights()



    def initialise_weights(self, sigma = 0.1*0.1, mu = 0):
        """ Initialise the weights and biases with gaussian random values"""
        self.W = np.random.randn(self.K, self.D) * sigma + mu
        self.b = np.random.randn(self.K, 1) * sigma + mu
        return self.W, self.b

    def softmax(self,S):
        """ Standard definition of the softmax function """
        return np.exp(S) / np.sum(np.exp(S), axis=0)

    def sigmoid(self,S):
        """ Standard definition of the softmax function """
        return np.exp(S) / (np.exp(S)+1)

    def EvaluateClassifier(self,X,W,b,activation = 'softmax'):
        """ Compute the scores of the classifier and applies the softmax function """
        S = np.matmul(W, X) + b
        if activation == 'softmax':
            P = self.softmax(S)
        else: 
            P = self.sigmoid(S)
        return P

    def Compute_Loss_Cost(self,X,Y,W,b,landa):
        """ Compute the cost function """
        # For Y,P, compute a mask and sum it first on the axis 0 and then on the axis 1
        P = self.EvaluateClassifier(X,W,b)
        loss = (-1/X.shape[1]) * np.sum(Y * np.log(P)) 
        # loss = (-1/X.shape[1]) * (1/self.K) * np.sum(Y * np.log(P)+(1-Y)*(np.log(1-P))) 
        cost = loss + landa* np.sum(W**2)
        return loss, cost

    def ComputeGradients(self,X,Y,W,b,landa):
        ''' Compute the gradients of the classifier'''
        grad_W = np.zeros(W.shape)
        grad_b = np.zeros((self.K, 1))
        dim = X.shape[1]

        '''Forward pass'''
        P = self.EvaluateClassifier(X,W,b)
        '''Backward pass'''
        G = -(Y-P)
        # G = (1/self.K)*((1-Y)*P - (1-P)*Y)  

        grad_W = (1/X.shape[1])*np.matmul(G, X.T) +2*landa*W
        # grad_b = np.sum(G, axis=1)[:, np.newaxis]
        grad_b = (1/X.shape[1])*np.sum(np.matmul(G, np.identity(X.shape[1])),axis = 1)
        return [grad_W, grad_b.reshape((self.K, 1))]

    def ComputeGradsNum(self,X, Y,W,b,landa, h=1e-6):
        """ Converted from matlab code """
        no 	= 	W.shape[0]
        d 	= 	X.shape[0]

        grad_W = np.zeros(W.shape)
        grad_b = np.zeros((no, 1))
        _,c = self.Compute_Loss_Cost(X,Y,W,b,landa)
        
        for i in range(len(b)):
            b_try = np.array(b)
            b_try[i] += h
            _,c2 = self.Compute_Loss_Cost(X,Y,W,b_try,landa)
            grad_b[i] = (c2-c) / h

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.array(W)
                W_try[i,j] += h
                _,c2 = self.Compute_Loss_Cost(X,Y,W_try,b,landa)
                grad_W[i,j] = (c2-c) / h
        return [grad_W, grad_b]

    def ComputeGradsNumSlow(self,X, Y, W, b, landa, h):
        """ Converted from matlab code """
        no 	= 	W.shape[0]
        d 	= 	X.shape[0]

        grad_W = np.zeros(W.shape);
        grad_b = np.zeros((no, 1));
        
        for i in range(len(b)):
            b_try = np.array(b)
            b_try[i] -= h
            c1 = ComputeCost(X, Y, W, b_try, landa)

            b_try = np.array(b)
            b_try[i] += h
            c2 = ComputeCost(X, Y, W, b_try, landa)

            grad_b[i] = (c2-c1) / (2*h)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.array(W)
                W_try[i,j] -= h
                c1 = ComputeCost(X, Y, W_try, b, landa)

                W_try = np.array(W)
                W_try[i,j] += h
                c2 = ComputeCost(X, Y, W_try, b, landa)

                grad_W[i,j] = (c2-c1) / (2*h)

        return [grad_W, grad_b]

    def check_grads(self):
        a = self.ComputeGradients(self.X_train[:,:5],self.Y_train[:,:5],self.W,self.b, 0)
        b = self.ComputeGradsNum(self.X_train[:,:5],self.Y_train[:,:5],self.W, self.b, 0)
        result = []
        for i in range(2):
            value = np.sum(np.abs(a[i]-b[i]))/(max(1e-6,np.sum(np.abs(a[i]))+np.sum(np.abs(b[i]))))
            result.append(value) 
        print('result: ', result[0], result[1])
        return None

    def ComputeAccuracy(self, X, Y, W, b):
        """ Compute the accuracy of the classifier """
        # Take the index with the maximum value
        score = self.EvaluateClassifier(X,W,b)
        pred = np.argmax(score, axis=0)
        gt = np.argmax(Y,axis=0)
        acc = (1/len(gt))*(np.sum(gt==pred))
        return acc

    def MiniBatchGD(self,X,Y,X_val,Y_val,X_test,Y_test,GDparams,W,b,landa,lr_epochs=30,lr_decay=False): 
        ''' Remember that the batches are not shuffled'''
        n_batch, eta, n_epochs = GDparams
        n = X.shape[1]
        cost_train, cost_val, loss_train, loss_val,test_acc,train_acc,val_acc = ([] for i in range(7))
        for i in tqdm(range(n_epochs)):
            print('epoch ',i,'/',n_epochs)

            '''Scheduled learning rate'''
            if i!=0:
                if (lr_decay==True) and (i%lr_epochs==0):
                    print('lr scaled:',eta/10)
                    eta = eta/10

            '''Divide in batches and compute gradients'''    
            for j in range(1,int(n/n_batch)):
                j_start = j * n_batch
                j_end = (j + 1) * n_batch
                Xbatch = X[:, j_start:j_end] 
                Ybatch = Y[:, j_start:j_end] 
                # Xbatch,Ybatch = flip_images(Xbatch,Ybatch)
                [grad_W, grad_b] = self.ComputeGradients(Xbatch,Ybatch,W,b,landa)
                W = W - eta*grad_W
                b = b - eta*grad_b

            loss_train_v, cost_train_v = self.Compute_Loss_Cost(X, Y, W, b, landa)
            loss_val_v, cost_val_v = self.Compute_Loss_Cost(X_val, Y_val, W, b, landa)
            cost_train.append(cost_train_v)
            cost_val.append(cost_val_v)
            loss_train.append(loss_train_v)
            loss_val.append(loss_val_v)
            test_acc.append(self.ComputeAccuracy(X_test, Y_test,W,b))
            train_acc.append(self.ComputeAccuracy(X, Y,W,b))
            val_acc.append(self.ComputeAccuracy(X_val, Y_val,W,b))

            print('Train acc, Val acc, Test acc|Train cost, val cost|Train loss, val loss| ',np.round(train_acc[-1],3),\
                np.round(val_acc[-1],3),np.round(test_acc[-1],3),'|',np.round(cost_train[-1],3),np.round(cost_val[-1],3),\
                    '|',np.round(loss_train[-1],3),np.round(loss_val[-1],3) )

        return W,b,loss_train,loss_val,cost_train,cost_val, test_acc,train_acc,val_acc 
    
    def train_model(self,n_batch=100,n_epochs=40,landa=0,eta=.1, lr_epochs = 30,lr_decay = False):

        timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
        GDparams=[n_batch,eta,n_epochs]
        
        W_star, b_star,loss_train,loss_val,cost_train,cost_val, test_acc,train_acc,val_acc = self.MiniBatchGD(self.X_train,self.Y_train,\
                                            self.X_val,self.Y_val,self.X_test,self.Y_test,GDparams,self.W,self.b,landa,lr_epochs=lr_epochs, lr_decay=lr_decay)
        
        self.save_state(timestr,W_star,b_star,cost_train,cost_val,loss_train,loss_val,GDparams,landa,test_acc,train_acc,val_acc)

        return W_star,b_star,cost_train,cost_val,loss_train,loss_val,GDparams,landa,test_acc,train_acc,val_acc

    def GridSearch(self, param_grid,X, Y,X_val,Y_val,X_test,Y_test, W, b,lr_epochs=5, lr_decay=False):
        from sklearn.model_selection  import ParameterGrid
        grid = ParameterGrid(param_grid)
        root = time.strftime("grid_%Y-%m-%d_%H-%M-%S")
        dir ='./results/'+root+'/' 
        if not os.path.exists(dir):
            os.makedirs(dir)
        all_acc = [] # to find later the setting with more acc

        for i,params in enumerate(grid):
            print('Fit number ',i,'/',len(grid))
            print('---------------------------------------')
            for key in params:
                print(key,':',params[key])  
            print('---------------------------------------')

            timestr = time.strftime("%Y-%m-%d_%H-%M-%S_grid_search")
            GDparams=[params['n_batch'],params['eta'] ,params['n_epochs']]

            W_star, b_star,loss_train,loss_val,cost_train,cost_val, test_acc,train_acc,val_acc = self.MiniBatchGD(X, Y,X_val,Y_val,X_test,Y_test,\
                             GDparams, W, b, params['landa'],lr_epochs, lr_decay)
            model_acc = self.ComputeAccuracy(X_test, Y_test,W_star, b_star)
            all_acc.append(model_acc)
            params.update({'accuracy':model_acc})
            self.save_state(root+'/'+timestr,W_star,b_star,cost_train,cost_val,loss_train,loss_val,GDparams,params['landa'],test_acc,train_acc,val_acc)

        best = np.argmax(np.array(all_acc))
        print('The best combination is:',grid[best], 'with an accuracy of ',all_acc[best])

        return None  

    def save_state(self,timestr, W,b,cost_train,cost_val,loss_train,loss_val,GDparams,landa,test_acc,train_acc,val_acc):
        ''' Save all the data to a folder'''
        [n_batch,eta,n_epochs]= GDparams
        dir = './results/'+timestr+'/'
        if not os.path.exists(dir):
            os.makedirs(dir)

        # np.save(dir+'W.npy', W) 
        # np.save(dir+'b.npy', b) 
        # np.save(dir+'cost_train.npy', cost_train) 
        # np.save(dir+'cost_val.npy', cost_val) 
        # np.save(dir+'loss_train.npy', loss_train) 
        # np.save(dir+'loss_val.npy', loss_val) 
        # np.save(dir+'test_acc.npy', test_acc) 
        # np.save(dir+'val_acc.npy', val_acc) 

        text = 'test acc '+str(test_acc[-1])+', lambda '+str(landa)+', n_batch '+ str(n_batch)+', eta '+str(eta)+', n_epochs '+str(n_epochs)
        with open(dir+'info.txt', 'w') as f:
            f.write(text)

        plots = [cost_train,cost_val,loss_train,loss_val]
        plot_name = ['cost train','cost val','loss train','loss val']
        plt.title('Cost, Loss: train and validation ')
        plt.xlabel('Epochs')
        plt.ylabel('Cost and Loss')
        for i,plot in enumerate(plots):       
            plt.plot(plot, label = plot_name[i])
        plt.legend()
        plt.savefig(dir+'cost-loss.png')
        plt.show()
        
        plt.title('Accuracy: train and validation ')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(train_acc, label = 'train_acc')
        plt.plot(val_acc, label = 'val_acc')
        plt.legend()
        plt.savefig(dir+'acc.png')
        plt.show()

        """ Display the image for each label in W """
        fig, ax = plt.subplots(2,5)
        for i in range(2):
            for j in range(5):
                im  = W[i*5+j,:].reshape(32,32,3, order='F')
                sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
                sim = sim.transpose(1,0,2)
                ax[i][j].imshow(sim, interpolation='nearest')
                ax[i][j].set_title("y="+str(5*i+j))
                ax[i][j].axis('off')
        plt.savefig(dir+'weights.png')
        plt.show()
        return None

    def load_state(timestr):
        ''' Load all the data from a folder'''
        dir = './results/'+timestr+'/'
        W = np.load(dir+'W.npy') 
        cost_train = np.load(dir+'cost_train.npy') 
        cost_val = np.load(dir+'cost_val.npy') 
        loss_train = np.load(dir+'loss_train.npy') 
        loss_val = np.load(dir+'loss_val.npy') 
        
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
        print('gt', gt.shape, gt)
        print('pred', pred.shape, pred)
        print('corr', correct.shape)
        print('acc',np.sum(correct)/len(correct))
        true_pred = []
        false_pred = []
        
        for i in range(correct.shape[0]): 
            # print(probs[pred[i],i],probs[gt[i],i])
            # print('probs',probs[:,i])
            # print('pred[i]',pred[i])
            # print('gt[i]',gt[i])
            if correct[i]==True: 
                true_pred.append(probs[pred[i],i])
            else: 
                false_pred.append(probs[gt[i],i])
        
        plt.title('Correctly/Incorrectly classified samples')
        plt.xlabel("Probability of ground truth")
        plt.ylabel('Frequency')
        plt.hist(true_pred,70,alpha=0.6,label = 'Correct samples')
        # plt.savefig(dir+'false_prediction.png')
        # plt.show()

        # plt.title('Inorrectly classified samples')
        # plt.xlabel("Probability of ground truth")
        # plt.ylabel('Frequency')
        plt.hist(false_pred,70,alpha=0.6,label = 'Incorrect samples')
        plt.legend()
        plt.savefig(dir+ 'prediction.png')
        plt.show()

        return None







 

