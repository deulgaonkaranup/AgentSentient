import numpy
from agents import Agent
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

class Agent_sentient(Agent):
    
    def choose_the_best_classifier(self, X_train, y_train, X_val, y_val):
        
        Domain       = numpy.zeros(3)
        LogLoss      = numpy.zeros(3)
        ZeroOneError = numpy.zeros(3)
        prob         = numpy.zeros(3)
        
        '''Initialise the variables with default values'''
        lr = LogisticRegression()
        bnb = BernoulliNB()
        svc = SVC()
        svc.kernel = 'poly'
        svc.degree = 4
        svc.probability = True
        svc.random_state = 0
        
        '''Train all 3 classifiers on the training data'''
        
        svc.fit(X_train, y_train)
        bnb.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        
        '''Three different kinds of tests on validation set for individual classifier'''
        value = 1000
        epsilon = 1e-10
        ll = numpy.zeros(3);
        error = numpy.zeros(3);
        for i in range(len(y_val)):
            prob[0] = bnb.predict_proba(X_val[i])[0][0]
            prob[1] = lr.predict_proba(X_val[i])[0][0]
            prob[2] = svc.predict_proba(X_val[i])[0][0]
            
            '''Domain relevant test, Results stored in Domain array size 3. 0 - BNB, 1 - LR, 2 - SVC'''
            
            for pt in range(10):
                price = ((2*pt+1)*value)/(20)
                if(value*prob[0] > price):
                    Domain[0] -= price
                    if y_val[i] == 'Excellent':
                        Domain[0] += value
                if(value*prob[1] > price):
                    Domain[1] -= price
                    if y_val[i] == 'Excellent':
                        Domain[1] += value
                if(value*prob[2] > price):
                    Domain[2] -= price
                    if y_val[i] == 'Excellent':
                        Domain[2] += value
            
            '''Log-Loss Error Check, Results stored in ll array size 3. 0 - BNB, 1 - LR, 2 - SVC'''            
            
            if y_val[i] == 'Excellent':
                '''This block for Excellent'''
                ll[0] += -(numpy.log(prob[0]+epsilon))
                ll[1] += -(numpy.log(prob[1]+epsilon))
                ll[2] += -(numpy.log(prob[2]+epsilon))
                
                '''0/1 Error Check, Results stored in error array size 3. 0 - BNB, 1 - LR, 2 - SVC'''
                
                if prob[0] < 0.5:
                    error[0] += 1
                if prob[1] < 0.5:
                    error[1] += 1
                if prob[2] < 0.5:
                    error[2] += 1
            else:
                '''This block for Trash'''
                ll[0] += -(numpy.log(1-prob[0]+epsilon))
                ll[1] += -(numpy.log(1-prob[1]+epsilon))
                ll[2] += -(numpy.log(1-prob[2]+epsilon))
                
                '''0/1 Error Check, Results stored in error array size 3. 0 - BNB, 1 - LR, 2 - SVC'''
                
                if prob[0] >= 0.5:
                    error[0] += 1
                if prob[1] >= 0.5:
                    error[1] += 1
                if prob[2] >= 0.5:
                    error[2] += 1
        
        count = numpy.zeros(3)
        
        '''Final Decision, Choose the best among the three which beats the maximum of the three criteria compared to other agents'''
        
        if Domain[0] > Domain[1] and Domain[0] > Domain[2]:
                count[0] = count[0] + 1
        elif Domain[1] > Domain[0] and Domain[1] > Domain[2]:
                count[1] = count[1] + 1
        elif Domain[2] > Domain[1] and Domain[2] > Domain[0]:
              count[2] = count[2] + 1
        
        if ll[0] < ll[1] and ll[0] < ll[2]:
                count[0] = count[0] + 1
        elif ll[1] < ll[0] and ll[1] < ll[2]:
                count[1] = count[1] + 1
        elif ll[2] < ll[1] and ll[2] < ll[0]:
              count[2] = count[2] + 1
        
        if error[0] < error[1] and error[0] < error[2]:
                count[0] = count[0] + 1
        elif error[1] < error[0] and error[1] < error[2]:
                count[1] = count[1] + 1
        elif error[2] < error[1] and error[2] < error[0]:
              count[2] = count[2] + 1
        
        if count[0] > count[1] and count[0] > count[2]:
            return BernoulliNB()
        elif count[1] > count[0] and count[1] > count[2]:
            return LogisticRegression()
        elif count[2] > count[1] and count[2] > count[0]:
            svc = SVC()
            svc.kernel = 'poly'
            svc.degree = 4
            svc.probability = True
            svc.random_state = 0
            return svc
