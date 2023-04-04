import numpy as np
from matplotlib import pyplot as plt
import csv
import pandas as pd
from scipy.special import binom
import itertools
from scipy.optimize import minimize
import pickle as pkl


%matplotlib qt


# dividing data on training and test sets
def train_test_divide(X, y, train_per):
    ind = np.arange(X.shape[0])
    np.random.shuffle(ind)
    X = X[ind]
    y = y[ind]
    
    train_ind = np.int64(train_per * len(ind))
    
    X_train = X[:train_ind]
    X_test = X[train_ind:]
    y_train = y[:train_ind]
    y_test = y[train_ind:]
    
    return X_train, X_test, y_train, y_test



class LASSORegression():
    
    def __init__(self, degree, learning_rate, iterations):
        self.degree = degree
        self.lr = learning_rate
        self.iterations = iterations
    
    def transform(self, X):
        self.m, self.n = X.shape
        X_transform = np.ones((self.m,1))
        
        j = 0
        
        for j in range(1,self.degree+1):
            x_pow = np.power(X,j)
            X_transform = np.append(X_transform,x_pow,axis=1)
        
        # making combinations if there is more than one feature
        if self.n > 1:
            col1 = np.arange(self.n)
            for j in range(2,self.degree+1):
                comb = np.int64(list(itertools.combinations(col1,j)))
                #print(comb.shape)
                r = comb.shape[0]
                for i in range(r):
                    x_mul = X[:,comb[i,0]]
                    for k in range(1,len(comb[i,:])):
                        x_mul = np.multiply(x_mul,X[:,comb[i,k]])
                    X_transform = np.append(X_transform,x_mul.reshape(-1,1),axis=1)
        
        return X_transform
    
    def standardize(self, X):
        X[:,1:] = (X[:,1:] - np.mean(X[:,1:],axis=0)) / np.std(X[:,1:],axis=0)
        
        return X
    
    # cross-validation
    def fit(self, X, Y, K):
        self.X = X
        self.Y = Y
        self.m, self.n = self.X.shape
        
        # calculating number of features
        self.n_f = self.degree * binom(self.n,1)
        for i in range(2,self.degree+1):
            self.n_f += binom(self.n,i)
        self.n_f = np.int64(self.n_f)
        #self.Theta = np.zeros((self.n_f + 1)).reshape(-1,1)
        
        #lambdas = np.linspace(0,50000,4000) # tried, rough graphic
        #lambdas = np.logspace(-3,4,3000) # gives nice graphic :)
        #lambdas = np.array([0,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2,3])
        lambdas = np.linspace(1500,1500,1) # optimal lambda -> min mean value and ok std
        self.Theta_lam = []
        e_lam_tr = np.zeros((2,len(lambdas)))
        e_lam_val = np.zeros((2,len(lambdas)))
        
        n_fold = np.int64(X.shape[0]/K)
        
        X_transform = self.transform(self.X)
        X_standardize = self.standardize(X_transform)
        
        for l in range(len(lambdas)):
            lambda1 = lambdas[l]
            
            # metrics
            m_tr = np.zeros(K)
            m_val = np.zeros(K)
            
            for k in range(K):
                self.Theta = np.zeros((self.n_f + 1)).reshape(-1,1)
                Xval = []
                Xtr = []
                if k == 0:
                    Xtr = X_standardize[n_fold:,:]
                    Xval = X_standardize[k*n_fold:k*n_fold+n_fold,:]
                    ytr = self.Y[n_fold:]
                    yval = self.Y[k*n_fold:k*n_fold+n_fold]
                elif k == K - 1:
                    Xtr = X_standardize[0:self.m-n_fold+1,:]
                    Xval = X_standardize[(k*n_fold):,:]
                    ytr = self.Y[0:self.m-n_fold+1]
                    yval = self.Y[(k*n_fold):]
                else:
                    Xtr = np.append(X_standardize[0:k*n_fold,:],X_standardize[(k*n_fold+n_fold):,:],axis=0)
                    Xval = X_standardize[k*n_fold:(k*n_fold+n_fold),:]
                    ytr = np.append(self.Y[0:k*n_fold],self.Y[(k*n_fold+n_fold):],axis=0)
                    yval = self.Y[k*n_fold:(k*n_fold+n_fold)]
                    #print(k)
                # gradient descent method
                e_iter_tr = np.arange(self.iterations)
                e_iter_val = np.arange(self.iterations)
                for i in range(self.iterations):
                    #h_Theta = self.predict(self.X)
                    h_Theta = Xtr @ self.Theta
                    #print(h_Theta)
                    delta = h_Theta - ytr
                    
                    dR1 = np.sign(self.Theta)
                    dR1[0] = 0
                    #print(dR1)
                    self.Theta = self.Theta - self.lr * (2 * np.transpose(Xtr) @ delta + lambda1 * dR1)
                ########## for visualising error(iteration) on test/validation set
                #     y_tr = Xtr @ self.Theta
                #     y_val = Xval @ self.Theta
                #     e_iter_tr[i] = np.float64(1/Xtr.shape[0] * np.power(np.transpose(y_tr-ytr) @ (y_tr-ytr),0.5))
                #     e_iter_val[i] = np.float64(1/Xval.shape[0] * np.power(np.transpose(y_val-yval) @ (y_val-yval),0.5))
                
                # plt.figure()
                # plt.plot(np.arange(self.iterations), e_iter_tr, color='blue')
                # plt.plot(np.arange(self.iterations), e_iter_val, color='red')
                # red_br = 'fold {}'.format((k+1))
                # plt.title(red_br)
                # plt.xlabel('$l$ (iteration)')
                # plt.ylabel('e')
                # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                # plt.legend(['trainng','validation'])
                # plt.show()
                ##########
        ########## for visualising error(lambda) on test/validation set        
        #         y_tr = Xtr @ self.Theta
        #         y_val = Xval @ self.Theta
                
        #         m_tr[k] = 1/Xtr.shape[0] * np.power(np.transpose(y_tr-ytr) @ (y_tr-ytr),0.5)
        #         m_val[k] = 1/Xval.shape[0] * np.power(np.transpose(y_val-yval) @ (y_val-yval),0.5)
            
            
        #     e_lam_tr[0,l] = np.mean(m_tr)
        #     e_lam_tr[1,l] = np.std(m_tr)
        #     e_lam_val[0,l] = np.mean(m_val)
        #     e_lam_val[1,l] = np.std(m_val)
        
        # std_tr_u = e_lam_tr[0,:] + 1/2 * e_lam_tr[1,:]
        # std_tr_d = e_lam_tr[0,:] - 1/2 * e_lam_tr[1,:]
        # std_val_u = e_lam_val[0,:] + 1/2 * e_lam_val[1,:]
        # std_val_d = e_lam_val[0,:] - 1/2 * e_lam_val[1,:]
        # plt.figure()
        # #plt.scatter(X, Y, color='blue')
        # plt.plot(lambdas,e_lam_tr[0,:], color='blue')
        # plt.plot(lambdas, e_lam_val[0,:], color='red')
        # plt.plot(lambdas,std_tr_u,color='blue',linestyle='dashed')
        # plt.plot(lambdas,std_tr_d,color='blue',linestyle='dashed')
        # plt.plot(lambdas,std_val_u,color='red',linestyle='dashed')
        # plt.plot(lambdas,std_val_d,color='red',linestyle='dashed')
        # plt.title('Hyperparameter selection')
        # plt.legend(['training','validation'])
        # plt.xlabel('$\lambda$')
        # plt.ylabel('$e$')
        # #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # plt.show()
        ##########
        
        # theta_file_name = 'theta'
        # theta_data = open(theta_file,'wb')
        # pkl.dump(self.Theta, theta_data)
        # theta_data.close()
        
        return self.Theta
    
    def model(self, X):
        X_transform = self.transform(X)
        X_standardize = self.standardize(X_transform)
        
        return X_standardize @ self.Theta




if __name__ == '__main__' :
    
    # reading data
    data = pd.read_csv('data.csv',header=None)
    data1 = pd.DataFrame(data).to_numpy() # conversion from DataFrame to np type
    
    r,c = data1.shape
    
    X = np.zeros((r,c-1))
    X[:,0:c] = data1[:,0:c-1]
    y = np.zeros((r,1))
    y[:,0] = data1[:,c-1]
    
    
    ########## debugging ---> playing data
    # X = np.linspace(-5,5,100)
    # #y = np.power(X,5) -30 * np.power(X,3) - 50 * np.power(X,2) + np.random.normal(0,200,100)
    # y = np.power(X,2) + np.random.normal(0,2,100)
    # X.shape = (X.shape[0],1)
    # y.shape = (y.shape[0],1)
    ##########
    
    # making trainnig and test data
    X_train, X_test, y_train, y_test = train_test_divide(X, y, 0.9)
    # y_train1 = (y_train - np.mean(y_train)) / np.std(y_train)
    # y_test1 = (y_test - np.mean(y_test)) / np.std(y_test)
    # X_train = np.array([[1.], [2.], [3.], [4.], [5.], [6.], [7.]])
    # y_train = np.array([[200000], [120000], [100000], [80000], [110000], [150000], [200000]])
    model = LASSORegression(degree=2, learning_rate=0.0001, iterations=500)
    Theta = model.fit(X_train, y_train, 5)
    y_pred = model.model(X_test)
    
    e = np.power(1/X_test.shape[0] * np.transpose(y_pred-y_test) @ (y_pred-y_test),0.5)
    print(e)
    
    # theta_file_name = 'theta'
    # theta_data = open(theta_file_name,'wb')
    # pkl.dump(Theta, theta_data)
    # theta_data.close()
    
    ########## debugging ---> test on playing data
    # X_test1 = np.array(X_test[:,0])
    # X_train1 = np.array(X_train[:,0])
    # y_pred1 = np.array(y_pred[:,0])
    # y_test1 = np.array(y_test[:,0])
    # sort_ind = X_train1.argsort()
    # #X_test1 = X_test1[sort_ind]
    # X_train1 = X_train1[sort_ind]
    # y_pred1 = y_pred1[sort_ind]
    # #y_test1 = y_test1[sort_ind]
    
    # plt.figure()
    # plt.plot(X_train1,y_pred1,color='orange')
    # #plt.scatter(X_test1,y_test1,color='red')
    # plt.scatter(X,y,color='red')
    # plt.title('Процена')
    # plt.xlabel('$x$')
    # plt.ylabel('y')
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.legend(['тест'])
    # plt.show()
    ##########































