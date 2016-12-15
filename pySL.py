import numpy as np
from scipy import stats
from scipy.optimize import minimize


def softMax(f):
    # Y = [x-1 for x in Y]
    exp_f = np.exp(f)
    norm = np.sum(exp_f, axis = 1)
    prob =  (exp_f.T / norm).T
    return prob

class SL:
    def __init__(self, pred, Y, index = 5000):
        self.pred = pred
        self.Y = Y
        self.d = pred.shape[0]
        self.pred_val = self.pred[:,:5000,:]
        self.Y_val = Y[:5000]
        self.pred_test = self.pred[:,5000:,:]
        self.Y_test = Y[5000:]
        
    def loss(self, f, Y):
        f -= np.max(f)
        row_sum = np.sum(np.exp(f), axis = 1)

        n = len(Y)
        f_correct = f[range(n), Y]
        # print (np.exp(f_correct) / row_sum)
        loss = -np.mean(
        np.log(
          (np.exp(f_correct) / row_sum)
            )
          )
        return loss
    

        
    def nll_SL(self, w):
        
        for i in range(self.d):
            if i == 0:
                f_new = w[i] * self.pred_val[i, :, :]
            else:
                f_new = f_new + w[i] * self.pred_val[i, :, :]
        return self.loss(f_new, self.Y_val)
        
    def fit(self, constrained = False):
        self.w0 = np.array([1.0/ self.d] * self.d)
        # print self.w0 
    
        cons = ({'type': 'eq',
                   'fun' : lambda x: np.array(sum(x) - 1)})
            
        self.res_con = minimize(self.nll_SL, self.w0, bounds = ((0, None),) * self.d,
               constraints=cons,method='SLSQP')

        self.res_uncon = minimize(self.nll_SL, self.w0, bounds = ((None, None),) * self.d,
               # constraints=cons, method='SLSQP', options={'disp': True})
                method='SLSQP')
            
    def predict(self, method = 'SL', softmax = False, constrained = False):
        m = self.pred.shape[0]
        n = self.pred.shape[1]
        if method == 'SL':
            if constrained:
                w = self.res_con['x']
            else:
                w = self.res_uncon['x']
        elif method == 'naive':
            w = self.w0
        elif method == 'bayesian':
           
            w = []
            for i in range(m):
                tmp_pred = softMax(self.pred[i,:,:])
                curr_loglike = tmp_pred[range(5000), self.Y_val[0:5000]]
                # print curr_loglike
                curr_weight = np.sum(np.log(curr_loglike))
                w.append(curr_weight)
            w -= np.mean(w)
            
            
            if max(w) - sorted(w)[-2] > 50:
                ind = np.argmax(w)
                w = np.zeros_like(w)
                w[ind] = 1
            else:
                w = np.exp(w) / np.sum(np.exp(w))
            # print 'Bayesian weight is', w
            
        
        if method == 'SL' or softmax == False:    
            for i in range(self.d):
                if i == 0:
                    f_new = w[i] * self.pred_test[i, :, :]
                else:
                    f_new = f_new + w[i] * self.pred_test[i, :, :]
        else:
            for i in range(self.d):
                if i == 0:
                    f_new = w[i] * softMax(self.pred_test[i, :, :])
                else:
                    f_new = f_new + w[i] * softMax(self.pred_test[i, :, :])
            
            
        return f_new
            
        
    def majority(self):
        for i in range(self.d):
            if i == 0:
                res = np.argmax(self.pred_test[i, :, :], axis = 1)
            else:
                res = np.vstack((res, np.argmax(self.pred_test[i, :, :], axis = 1)))
        major = stats.mode(res)
        return major[0]
            
        
