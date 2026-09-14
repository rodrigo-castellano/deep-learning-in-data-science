import numpy as np
import os
from tqdm import tqdm
import os

from utils import *

class model:
    def __init__(self,n_hidden, D,K):
        '''D: Dimension of the samples
        K: Number of classes
        n_hidden: list with the number of nodes of each layer '''
        self.D = D # Dimension of the samples
        self.K = K # Number of classes
        self.n_hidden = n_hidden # CHANGE NAME TO HIDDEN_LAYERS. SET N_HIDDEN= LEN()
        self.n_layers = len(n_hidden) +1
        self.S_mean, self.S_var = [], []
        self.av_mean, self.av_var = [],[]
        self.train_mean, self.train_var = [],[]


    def initialise_weights(self, sigma = 0.1*0.1, mu = 0, init='He'):
        """ Initialise the weights and biases with gaussian random values
        n_hidden: number of hidden layers                       [    hidden layers               ]
        n_layers: first layer + hidden layers                   [X - hidden layers               ]
        all_layers: n_layers including the last layer for output[X - hidden layers - output layer]
        n_layers =   n_hidden + 1
        all layers = n_layers + 1 = n_layers + 2
        len(W,b) = all_layers;   len(gamma,beta) = n_layers
        Parameters:                       
        Weights:                                
        """
        all_layers = [self.D] + self.n_hidden + [self.K]
        self.W,self.b = [],[]
        if (self.batchnorm):
            self.gamma, self.beta = [],[]

        for i in range(len(all_layers)-1): 
            self.b.append(np.zeros((all_layers[i+1], 1)))
            if init=='He':
                sigma = np.sqrt(2/all_layers[i])
                self.W.append( np.random.normal(0.0, sigma, (all_layers[i+1], all_layers[i])) )
            elif init=='Xavier':
                sigma = 1/np.sqrt(all_layers[i])
                self.W.append( np.random.normal(0.0, sigma, (all_layers[i+1], all_layers[i])) )
            elif init =='Normal': #sig=1e-1, 1e-3 and 1e-4
                sigma = 1e-4  # (all_layers[i])
                self.W.append(np.random.normal(0.0, sigma, (all_layers[i+1], all_layers[i])))
            elif init =='Random':  
                sigma = 1/np.sqrt(all_layers[i])
                self.W.append(np.random.randn(all_layers[i+1], all_layers[i]) * sigma + mu)
            if (self.batchnorm) and (i < len(all_layers)-2):
                self.gamma.append( np.ones(all_layers[i+1]).reshape(all_layers[i+1], 1) ) 
                self.beta.append( np.zeros((all_layers[i+1], 1)) )
        if self.batchnorm:
            return self.W,self.b, self.gamma, self.beta
        else: 
            return self.W,self.b

    def softmax(self,S):
        """ Standard definition of the softmax function """
        return np.exp(S) / np.sum(np.exp(S), axis=0)

    def sigmoid(self,S):
        """ Standard definition of the sigmoid function """
        return np.exp(S) / (np.exp(S)+1)

    def activation(self,A, activation='Relu'):
        if activation == 'Relu':
           out =  np.maximum(A,0)
        elif activation == 'LeakyRelu':
          out = np.maximum(0.1*A,A)
        elif activation == 'Elu':
            if A>0:
                out = A
            else: 
                out = 0.9*(np.exp(A)-1)
        elif activation == 'Swish':
            out = A*self.sigmoid(A)
        return out

    def Compute_Loss_Cost(self,X,Y,landa,mode):
        """ Compute the cost function """
        n = X.shape[1]
        out = self.EvaluateClassifier(X,mode)
        P = out[0]
        loss = (-1/n) * np.sum(Y * np.log(P)) 
        reg = 0
        # For L1 regularization: for each layer->np.sum(np.absolute(W[i]))
        for i in range(len(self.W)):
            reg += np.sum(self.W[i]**2) 
        cost = loss + landa*reg
        return loss, cost

    def compute_grads_num(self,X,Y,landa=0,h=1e-7):
        if self.batchnorm: 
            grad_W,grad_b,grad_gamma,grad_beta = [],[],[],[]
            grads = [grad_W,grad_b,grad_gamma,grad_beta]
            weights = [self.W,self.b,self.gamma,self.beta]
        else: 
            grad_W,grad_b = [],[]
            grads = [grad_W,grad_b]        
            weights = [self.W,self.b]

        for k in range(len(grads)): 
            for j in range(len(weights[k])):
                grads[k].append(np.zeros(weights[k][j].shape))
                for i in range(len(weights[k][j].flatten())):
                        old_par = weights[k][j].flat[i]
                        weights[k][j].flat[i] = old_par + h
                        _,c2 = self.Compute_Loss_Cost(X,Y,landa,mode='train')
                        weights[k][j].flat[i] = old_par - h
                        _,c3 = self.Compute_Loss_Cost(X,Y,landa,mode='train')
                        weights[k][j].flat[i] = old_par 
                        grads[k][j].flat[i] = (c2-c3)/(2*h)
        return grads

    def check_grads(self,X,Y,batchnorm=False):
        self.batchnorm = batchnorm
        self.initialise_weights(init=self.weight_ini)
        a = self.ComputeGradients(X,Y,0) 
        b = self.compute_grads_num(X,Y)
        for i in range(len(b)):
            for j in range(len(b[i])):
                diff = np.sum(np.abs(a[i][j]-b[i][j]))/(max(1e-6,np.sum(np.abs(a[i][j])+np.abs(b[i][j]))))
                print('diff',diff)
        return None

    def ComputeAccuracy(self, X, Y):
        """ Compute the accuracy of the classifier """
        out = self.EvaluateClassifier(X,mode='test')
        P = out[0]
        pred = np.argmax(P, axis=0)
        gt = np.argmax(Y,axis=0)
        acc = (1/len(gt))*(np.sum(gt==pred))
        return acc

    def BatchNormalize(self,S,mu,var):
        return np.diag(pow(var.flatten() + np.finfo(float).eps, -.5)) @ (S - mu)
    
    def Expected_value(self,X,X_mean):
        train_len = len(X_mean)
        return X_mean/train_len + (train_len/(train_len+1))*X

    def EvaluateClassifier(self,X,mode,function = 'softmax'):
        """ Compute the scores of the classifier and applies the softmax function """
        H = []
        H.append(np.copy(X))
        if self.batchnorm==False:
            for i in range(self.n_layers-1):
                S = np.matmul(self.W[i], H[i]) + self.b[i]
                H.append( self.activation(S) )
        else: 
            S_hat_list,S_list = [],[]
            for i in range(self.n_layers-1):

                if mode=='train':
                    S = np.matmul(self.W[i], H[i]) + self.b[i]
                    S_mean = np.mean(S,axis=1,keepdims=True)
                    S_var = np.var(S,axis=1,keepdims=True)

                    # To do adaptative BN, calculate the mean and var of the whole dataset
                    # by keeping track of every mean,var and doing the average always

                    if self.train_BN and len(self.train_mean)<self.n_layers-1:
                        self.train_mean.append(S_mean)
                        self.train_var.append(S_var)

                    if self.train_BN==False and len(self.train_mean)<self.n_layers-1:  # Initialise av_mean,av_var
                        self.av_mean.append(S_mean)
                        self.av_var.append(S_var)

                    else: 
                        if self.train_BN:
                            self.train_mean[i] = self.Expected_value(S_mean,self.train_mean[i])
                            self.train_var[i] = self.Expected_value(S_var,self.train_var[i])
                        else:
                            self.av_mean[i] =  0.9*self.av_mean[i]+(1-0.9)*S_mean 
                            self.av_var[i] = 0.9*self.av_var[i]+(1-0.9)*S_var 

                    if self.train_BN:
                        S_hat = self.BatchNormalize(S,self.train_mean[i],self.train_var[i])  
                    else:
                        S_hat = self.BatchNormalize(S,S_mean,S_var)

                    S_tilda = np.multiply(self.gamma[i],S_hat) +self.beta[i]
                    H.append( self.activation(S_tilda) )
                    S_hat_list.append(S_hat)
                    S_list.append(S)

                elif mode=='test':
                    S = np.matmul(self.W[i], H[i]) + self.b[i]
                    if self.train_BN:
                        S_hat = self.BatchNormalize(S,self.train_mean[i],self.train_var[i])
                    else:
                        S_hat = self.BatchNormalize(S,self.av_mean[i],self.av_var[i])
                    S_tilda = np.multiply(self.gamma[i],S_hat) +self.beta[i]
                    H.append( self.activation(S_tilda) )

        S = np.matmul(self.W[self.n_layers-1], H[self.n_layers-1]) + self.b[self.n_layers-1]
        if function == 'softmax':
            P = self.softmax(S)
        else: 
            P = self.sigmoid(S)

        if self.batchnorm==False:
            output = P,H
        elif self.batchnorm==True:
            output =  P,H,S_hat_list,S_list

        return output
        
    def BatchNormalizeBack(self,G,S,av_mean,av_var): 
        n = G.shape[1]
        ones_n = np.ones((n,1))
        sigma1 = np.power(av_var+1e-9,-.5)
        sigma2 = np.power(av_var+1e-9,-1.5)
        G1 = np.multiply(G, sigma1)
        G2 = np.multiply(G, sigma2)
        D = S - av_mean
        c = np.matmul(np.multiply(G2,D),ones_n)
        G = G1 - 1/n * np.matmul(G1, ones_n) - 1/n * np.multiply(D, c)
        return G
        
    def ComputeGradients(self,X,Y,landa):
        ''' Compute the gradients of the classifier'''
        n = X.shape[1]
        ones_n = np.ones((n,1))
        grad_W = []
        grad_b = []
        '''Forward pass'''
        output = self.EvaluateClassifier(X, mode='train') ###
        '''Backward pass'''
        if self.batchnorm == False:
            P,H = output
            G = -(Y-P)   
            for i in range(self.n_layers-1, -1, -1):
                grad_W.append( (1/n)*np.matmul(G, H[i].T) + 2*landa*self.W[i] ) 
                grad_b.append( (1/n)*np.matmul(G,ones_n) ) 
                if i>0:
                    G = np.matmul(self.W[i].T, G)
                    G[np.where(H[i]<=0)] = 0

        elif self.batchnorm == True:
            P,H,S_hat,S = output
            grad_gamma,grad_beta = [],[]
            G = -(Y-P)  

            grad_W.append( (1/n)*np.matmul(G, H[self.n_layers-1].T) + 2*landa*self.W[self.n_layers-1] )
            grad_b.append( (1/n)*np.matmul(G,ones_n) ) 
            G = np.matmul(self.W[self.n_layers-1].T, G) 
            G[np.where(H[self.n_layers-1]<=0)] = 0

            for i in range(self.n_layers-2, -1, -1):
                grad_gamma.append( (1/n)*np.matmul(np.multiply(G,S_hat[i]),ones_n) )  
                grad_beta.append( (1/n)*np.matmul(G,ones_n) )
                G = np.multiply(G,np.matmul(self.gamma[i],ones_n.T))  
                
                if self.train_BN:
                    G = self.BatchNormalizeBack(G,S[i],self.train_mean[i],self.train_var[i]) 
                else:
                    G = self.BatchNormalizeBack(G,S[i],self.av_mean[i],self.av_var[i]) 
                grad_W.append( (1/n)*np.matmul(G, H[i].T) + 2*landa*self.W[i] )
                grad_b.append( (1/n)*np.matmul(G,ones_n) ) 
                if i>0:
                    G = np.matmul(self.W[i].T, G)
                    G[np.where(H[i]<=0)] = 0

        grad_W.reverse()
        grad_b.reverse()
        if self.batchnorm:
            grad_gamma.reverse()
            grad_beta.reverse()
            output =  [grad_W, grad_b,grad_gamma,grad_beta]
        else:
            output =  [grad_W, grad_b]

        return output


    def MiniBatchGD(self,X,Y,X_val,Y_val,X_test,Y_test,optimizer,lr_scheduler,dir,Mparam): 
        ''' Remember that the batches are not shuffled'''
        n_samples = X.shape[1]
        self.cost_train,self.cost_val,self.loss_train,self.loss_val,self.train_acc,self.val_acc,self.etas = ([] for i in range(7))
        self.best_val_acc = 0

        n_epochs = Mparam['n_epochs']
        n_batch = Mparam['n_batch']
        landa = Mparam['landa']

        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir+'info.txt', 'a') as f:
            for key in Mparam:
                f.write(str(key) + ':' + str(Mparam[key])  +'\n')
            f.write('epoch| Train acc, Val acc| Train cost, val cost| Train loss, val loss|'+'\n')

        if lr_scheduler.lr_sch=='cyclic':
            eta,n_epochs = lr_scheduler.ini_scheduler(n_samples,n_batch)
        else: 
            eta = lr_scheduler.ini_scheduler(n_samples,n_batch)

        t = 1
        for epoch in tqdm(range(n_epochs )):
            print('epoch ',epoch+1,'/',n_epochs)

            p = np.random.permutation(X.shape[1])
            X = X[:,p]
            Y = Y[:,p]

            '''Divide in batches and compute gradients'''    
            for j in range(1,int(n_samples/n_batch)):
                j_start = j * n_batch
                j_end = (j + 1) * n_batch
                Xbatch = X[:, j_start:j_end] 
                Ybatch = Y[:, j_start:j_end] 
                # Xbatch,Ybatch = flip_augment_images(Xbatch,Ybatch)
                
                eta = lr_scheduler.update_lr(epoch,j,t)
                self.etas.append(eta)

                grads = self.ComputeGradients(Xbatch,Ybatch,landa)

                if self.batchnorm:
                    parameters = [self.W,self.b, self.gamma,self.beta] 
                    self.W,self.b,self.gamma,self.beta = optimizer.update_grads(t,eta,grads,parameters)
                else: 
                    parameters = [self.W,self.b]
                    [self.W,self.b] = optimizer.update_grads(t,eta,grads,parameters)

                t +=1

            loss_train_v, cost_train_v = self.Compute_Loss_Cost(X, Y, landa,mode='test')
            loss_val_v, cost_val_v = self.Compute_Loss_Cost(X_val, Y_val, landa,mode='test')
            self.cost_train.append(np.round(cost_train_v,3))
            self.cost_val.append(np.round(cost_val_v,3))
            self.loss_train.append(np.round(loss_train_v,3))
            self.loss_val.append(np.round(loss_val_v,3))
            self.train_acc.append(np.round(self.ComputeAccuracy(X, Y),3))
            self.val_acc.append(np.round(self.ComputeAccuracy(X_val, Y_val),3))
            if self.val_acc[-1] > self.best_val_acc:
                self.best_val_acc = self.val_acc[-1]
                best_epoch = epoch
                self.test_acc = np.round(self.ComputeAccuracy(X_test, Y_test),3)

            print('Train acc, Val acc|Train cost, val cost|Train loss, val loss| ',self.train_acc[-1],\
                self.val_acc[-1],'|',self.cost_train[-1],self.cost_val[-1],'|',self.loss_train[-1],self.loss_val[-1] )
            with open(dir+'info.txt', 'a') as f:
              f.write(str(epoch)+' '+str(self.train_acc[-1])+' '+str(self.val_acc[-1])+'|'+str(self.cost_train[-1])+' '+str(self.cost_val[-1])+'|'+\
                      str(self.loss_train[-1])+' '+str(self.loss_val[-1])+'\n')
          
        print('Best validation acc: ',self.best_val_acc,' in epoch ',best_epoch+1)
        print('Test acc in that epoch: ',self.test_acc)
        with open(dir+'info.txt', 'a') as f:
          f.write('Best validation acc: '+str(self.best_val_acc)+' in epoch '+str(best_epoch+1)+'\n')
          f.write('Test acc in that epoch: '+str(self.test_acc))

        return None
    

    def train_model(self,X_train,Y_train,X_val,Y_val,X_test,Y_test,optimizer,lr_scheduler,train_BN=False,\
                    n_epochs=20,n_batch=100,batchnorm = False,landa=0,dir='./results/tmp/',weight_ini='He'):

        lr_info = lr_scheduler.param_info()
        optimizer_info = optimizer.param_info()
        Modelparams = {'n_epochs' : n_epochs,
                      'n_batch' : n_batch,
                      'landa' : landa,
                      'batchnorm' : batchnorm,
                      'n_hidden': self.n_hidden,      
                      'weight_ini': weight_ini,   
                      'train_BN':train_BN,
                      }
        Modelparams.update(**lr_info,**optimizer_info)
        self.X_test = X_test
        self.Y_test = Y_test

        self.batchnorm = batchnorm
        self.train_BN = train_BN
        self.weight_ini = weight_ini
        self.initialise_weights(init=self.weight_ini)

        self.MiniBatchGD(X_train,Y_train,X_val,Y_val,X_test,Y_test,optimizer,lr_scheduler,dir,Modelparams)
        
        return dir,self.W,self.b,Modelparams,self.loss_train,self.loss_val,self.cost_train,self.cost_val,\
                                            self.train_acc,self.val_acc,self.test_acc,self.best_val_acc,self.etas