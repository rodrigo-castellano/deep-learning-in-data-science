from stat import ST_DEV
import numpy as np
import os
from tqdm import tqdm
import os
from utils import *
import time 
class model: 
    def __init__(self,m,K,ind_to_char,char_to_ind,seq_length=25):
        '''D: Dimension of the samples
        K: Number of classes
        n_hidden: list with the number of nodes of each layer '''
        self.K = K # Number of classes
        self.m = m # CHANGE NAME TO HIDDEN_LAYERS. SET N_HIDDEN= LEN()
        self.ind_to_char = ind_to_char
        self.char_to_ind = char_to_ind
        self.seq_length = seq_length


    def initialise_weights(self, sigma = 0.01, init = None):
        self.params = {}
        np.random.seed(0)
        self.params['W'] = np.random.normal(0.0, sigma, (self.m, self.m))
        np.random.seed(0)
        self.params['V'] = np.random.normal(0.0, sigma, (self.K, self.m))
        np.random.seed(0)
        self.params['U'] = np.random.normal(0.0, sigma, (self.m, self.K))
        self.params['b'] = np.zeros((self.m,1))
        self.params['c'] = np.zeros((self.K,1))
        return None

    def char2one_hot(self,chars,ind_to_char):
        ''' Parameters: 
                chars: number of characters to convert to one-hot encode, len N
                reference: set with all the unique elements: len K
            Returns:
                one-hot: matrix Kxn with the on-hot representation for each character from chars'''
        K = len(ind_to_char)
        N = len(chars)
        one_hot = np.zeros((K,N))

        for i in range(len(chars)):
            idx = ind_to_char[chars[i]]
            one_hot[idx,i] = 1
        return one_hot

    def one_hot2char(self,one_hot,char_to_ind):
        K,N= one_hot.shape[0],one_hot.shape[1]
        idx = np.where(one_hot.T[:,:]==1)[1] # I transpose it to get the indices by order easily
        char = ""
        for i in idx:
            char += char_to_ind[i]
        return char

    def softmax(self,S):
        """ Standard definition of the softmax function """
        return np.exp(S) / np.sum(np.exp(S), axis=0)

    def compute_grads_num(self,X,Y,h0,landa=0,h=1e-4):
        grad_W,grad_b,grad_V,grad_c,grad_U = [],[],[],[],[]
        grads = [grad_W,grad_b,grad_V,grad_c,grad_U] 
        weights = [[self.params['W']],[self.params['b']] ,[self.params['V']],[self.params['c']],[self.params['U']]]
        for k in range(len(grads)): 
            for j in range(len(weights[k])):
                grads[k].append(np.zeros(weights[k][j].shape))
                for i in range(len(weights[k][j].flatten())):
                    old_par = weights[k][j].flat[i]
                    weights[k][j].flat[i] = old_par + h
                    _,c2,_ = self.ComputeGradients(X,Y,h0)
                    weights[k][j].flat[i] = old_par - h
                    _,c3,_ = self.ComputeGradients(X,Y,h0)
                    weights[k][j].flat[i] = old_par 
                    grads[k][j].flat[i] = (c2-c3)/(2*h)

        for l in range(len(grads)):
            grads[l] = np.clip(grads[l], -5, 5)
        return grads

    def check_grads(self,X,Y):
        eps = np.finfo(np.float64).eps
        self.initialise_weights()
        h0 = np.zeros((self.m,1))
        a,_,_ = self.ComputeGradients(X,Y,h0)
        b = self.compute_grads_num(X,Y,h0)
        names = ['grad_W','grad_b','grad_V','grad_c','grad_U']
        for i in range(len(b)):
            print(names[i])
            b[i] = np.squeeze(b[i],axis=0) 
            diff = np.sum(np.abs(a[i]-b[i]))/(max(eps,np.sum(np.abs(a[i])+np.abs(b[i]))))
            print('diff',diff)
        return None

    def EvaluateRNN(self,X,Y,h0):
        """ Evaluate the RNN"""
        len_seq = X.shape[1]
        a = np.empty((len_seq,self.m,1))
        h = np.empty((len_seq+1,self.m,1))
        o = np.empty((len_seq,self.K,1))
        p = np.empty((len_seq,self.K,1))
        h[-1,:,:]=h0
        h_ = h0
        loss=0
        for t in range(len_seq):
            x = X[:,t]; x = x[:, np.newaxis]
            y = Y[:,t]; y = y[:, np.newaxis]
            a_ = self.params['W']@h_ + self.params['U']@x + self.params['b']
            h_ = np.tanh(a_)
            o_ = self.params['V']@h_ + self.params['c']
            p_ = self.softmax(o_)
            loss += -np.log(y.T@p_)
            a[t,:,:],h[t,:,:],o[t,:,:],p[t,:,:] = a_,h_,o_,p_

        #We always compare to the len_seq = 25. Therefore we divide the cost by (len_seq/25)
        batch_factor = self.seq_length/25
        return a,h,o,p,loss/(batch_factor)

    def ComputeGradients(self,X,Y,h0):
        ''' Compute the gradients of the classifier'''

        '''Forward pass'''
        len_seq = X.shape[1]
        a,h,o,p,loss = self.EvaluateRNN(X,Y,h0) 
        h_prev = h[-2,:,:]  # Becuase in h[-1] there is h0 -> h=[h1,...,h24,h0]

        '''Backward pass'''
        grad_a = np.empty((len_seq,self.m,1)) 
        grad_h = np.empty((len_seq+1,self.m,1))
        grad_o = np.empty((len_seq,self.K,1))
        grad_W,grad_U,grad_b,grad_c,grad_V = 0,0,0,0,0
         
        for t in range(len_seq-1, -1, -1):
            x = X[:,t]; x = x[:, np.newaxis]
            y = Y[:,t]; y = y[:, np.newaxis]

            grad_o[t,:,:] = -(y-p[t,:,:])
            grad_V += grad_o[t,:,:]@h[t,:,:].T 
            grad_c += grad_o[t,:,:] 
            if t==len_seq-1:
                grad_h[t,:,:] =  (grad_o[t,:,:].T@self.params['V']).T
            else:
                grad_h[t,:,:] =  (grad_o[t,:,:].T@self.params['V']).T + (grad_a[t+1,:,:].T@self.params['W'] ).T

            op = 1-np.tanh(a[t,:,:])**2
            grad_a[t,:,:] = (grad_h[t,:,:].T@np.diag( op.flatten() )).T
            grad_W += grad_a[t,:,:]@h[t-1,:,:].T 
            grad_b += grad_a[t,:,:]
            grad_U +=  grad_a[t,:,:]@x.T 
    
        grads = [grad_W,grad_b,grad_V,grad_c,grad_U]

        for g in range(len(grads)):
            grads[g] = np.clip(grads[g], -5, 5)
        return grads,loss,h_prev

    def Synthesize_seq(self,h0,x0,len_seq):
        """ Evaluate the classifier"""
        h_ = h0
        x = x0
        Y = np.zeros((self.K, len_seq))
        for t in range(len_seq):
            y = Y[:,t]; y = y[:, np.newaxis]
            a_ = self.params['W']@h_ + self.params['U']@x + self.params['b']
            h_ = np.tanh(a_)
            o_ = self.params['V']@h_ + self.params['c']
            p = self.softmax(o_)
            
            cp = np.cumsum(p)
            ixs = np.random.choice(self.K, p=p.flat) 
            x = np.zeros(x.shape)
            x[ixs] = 1
            Y[ixs,t] = 1    
        return Y


    def MiniBatchGD(self,data,optimizer,lr_scheduler,dir,Mparam): 
        ''' Assign constants '''
        n_samples = data.shape[1]
        self.loss_train,self.etas = ([] for i in range(2))
        self.best_loss_train = np.Inf
        n_epochs = Mparam['n_epochs']
        n_batch = Mparam['n_batch']*self.seq_length
        smooth_loss = 0
        h0 = np.zeros((self.m,1))

        ''' Write in a file '''
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir+'info.txt', 'a') as f:
            for key in Mparam:
                f.write(str(key) + ':' + str(Mparam[key])  +'\n')

        ''' Initialize lr'''
        if lr_scheduler.lr_sch=='cyclic':
            eta,n_epochs = lr_scheduler.ini_scheduler(n_samples,n_batch)
        else: 
            eta = lr_scheduler.ini_scheduler(n_samples,n_batch)

        t = 1
        print('n_epochs',n_epochs)
        print('n_samples',n_samples)
        print('n_batch',n_batch)
        print('n steps',int((n_samples)/n_batch))
        for epoch in tqdm(range(n_epochs)):
            print('epoch ',epoch+1,'/',n_epochs)
            ind = np.arange(int((n_samples)/n_batch))*n_batch
            np.random.shuffle(ind)
            ''' Divide the dataset in L=n_samples/n_batch  chunks. For each chunk reset h and then compute gradients 
            The chunks are not accesed in order, they are shuffled''' 
            ''' Save one chunk as valiation set'''
            X_val = data[:,ind[0]:n_batch+ind[0]]
            Y_val = data[:,ind[0]+1:n_batch+ind[0]+1]
            for j in range(1,int((n_samples)/n_batch)): 
                batch_start = ind[j]; batch_end = n_batch + ind[j]
                h_prev = np.copy(h0)

                for k in range(int(n_batch/self.seq_length)):
                    start = k*self.seq_length; end = k*self.seq_length + self.seq_length

                    if j == int((n_samples)/n_batch)-1:
                        X = data[:,(batch_start-1)+(start):(batch_start-1)+(end)]
                        Y = data[:,(batch_start)+(start):(batch_start)+(end)]
                    else: 
                        X = data[:,(batch_start)+(start):(batch_start)+(end)]
                        Y = data[:,(batch_start+1)+(start):(batch_start+1)+(end)]

                    ''' Compute the gradients, loss and h_prev'''
                    grads,loss,h_prev = self.ComputeGradients(X,Y,h_prev)
                    loss = np.round(loss.flatten()[0],4)
                    if smooth_loss == 0:
                        smooth_loss = loss
                    smooth_loss = np.round(.999*smooth_loss + .001*loss,5) 
                    self.loss_train.append(smooth_loss)

                    ''' Update the gradients'''
                    eta = lr_scheduler.update_lr(epoch,j,t)
                    self.etas.append(eta)
                    # The format to give to optimizer is [nparams][nlayers][grad[param,layer]]->nparams=5,nlayers=1
                    for i in range(len(grads)):
                        grads[i] = [grads[i]] 
                    parameters = [[self.params['W']],[self.params['b']],[self.params['V']],[self.params['c']],[self.params['U']]] 
                    parameters = optimizer.update_grads(t,eta,grads,parameters)
                    [[self.params['W']],[self.params['b']],[self.params['V']],[self.params['c']],[self.params['U']]]=parameters
                    

                    if (t%1000 == 0) or (t==1):
                        print('t:',t,', smooth loss:',smooth_loss)
                        with open(dir+'info.txt', 'a') as f:
                            f.write(str(epoch)+'|'+str(t)+'|'+str(smooth_loss)+'|\n')

                    if (t%10000 == 0) or (t==1):
                        x = X[:,0]; x = x[:, np.newaxis]
                        pred_seq = self.Synthesize_seq(h_prev,x,200)
                        char = self.one_hot2char(pred_seq,self.char_to_ind)
                        print('-'*30,'\n seq:\n',char,'\n','-'*30)

                        with open(dir+'info.txt', 'a') as f:
                            f.write('\n Epoch| t | smooth loss| '+'\n')
                            f.write(str(epoch)+'|'+str(t)+'|'+str(smooth_loss)+'|\n')
                            f.write('-'*30+'\n seq :\n'+char+'\n'+'-'*30+'\n')

                        ''' For when we obtain the best accuracy every 10000 steps'''
                        if self.loss_train[-1] < self.best_loss_train:
                            self.best_loss_train = self.loss_train[-1]
                            best_t = t
                            x = X[:,0]; x = x[:, np.newaxis]
                            pred_seq = self.Synthesize_seq(h_prev,x,1000)
                            char = self.one_hot2char(pred_seq,self.char_to_ind)
                            print('*'*30,'\n Lowest loss achieved. Seq of length 1000::\n',char,'\n','*'*30)
                            with open(dir+'info.txt', 'a') as f:    
                                f.write('*'*30+'\n Lowest loss achieved. Seq of length 1000::\n'+char+'\n'+'*'*30+'\n')
                    t +=1


        print('Best loss: ',self.best_loss_train,' in t=',best_t)
        with open(dir+'info.txt', 'a') as f:
          f.write('Best loss: '+str(self.best_loss_train)+' in t='+str(best_t))

        return None
    

    def train_model(self,data,optimizer,lr_scheduler,\
                    n_epochs=2,n_batch=100,landa=0,dir='./results/tmp/',weight_ini=None):

        lr_info = lr_scheduler.param_info()
        optimizer_info = optimizer.param_info()
        Modelparams = {'n_epochs' : n_epochs,
                      'n_batch' : n_batch,
                      'landa' : landa,
                      'n_hidden': self.m,      
                      'weight_ini': weight_ini,   
                      }
        Modelparams.update(**lr_info,**optimizer_info)

        self.weight_ini = weight_ini
        self.initialise_weights(init=self.weight_ini)

        self.MiniBatchGD(data,optimizer,lr_scheduler,dir,Modelparams)
        
        return dir,Modelparams,self.loss_train,self.best_loss_train,self.etas