# -*- coding: utf-8 -*-
import numpy
from agents import Agent

class Agent_sentient(Agent):
    
    def calc_probability(self,array,value):
        excellent = 0
        for item in array:
            if item == 'Excellent':
                excellent = excellent + 1
        
        return excellent
    
    def train(self, X, y):
        
        '''Identifying number of Features and Products'''
        self.number_of_features = len(X[0])
        self.number_of_products = len(y)   
        
        '''Initializing Variables. feature_prob_distrib is a 2D array which stores the probability of feature/condition_of_prod
        rows represent the feature and column 0 and 1 represent the Condition Excellent and Trash respectively'''
        self.prob_excellent = 0.0
        self.feature_prob_distrib = numpy.zeros((self.number_of_features,2))
        
        '''Calculate variable P(Excellent) using y array'''
        no_of_excellent = self.calc_probability(y, 'Excellent')
        self.prob_excellent = float(no_of_excellent) / len(y)
        
        '''Using the array X and y, calculate probability and store it in feature_prob_distrib array
           First count and then divide by total to get individual probability'''
        for i in range(len(X)):
            product_cond = y[i]
            for j in range(len(X[0])):
                if X[i][j] == True and product_cond == 'Excellent':
                    self.feature_prob_distrib[j][0] = self.feature_prob_distrib[j][0] + 1
                elif X[i][j] == True and product_cond == 'Trash':
                    self.feature_prob_distrib[j][1] = self.feature_prob_distrib[j][1] + 1
        
        for i in range(len(self.feature_prob_distrib)):
                self.feature_prob_distrib[i][0] = ( float(self.feature_prob_distrib[i][0]) / no_of_excellent )
                self.feature_prob_distrib[i][1] = ( float(self.feature_prob_distrib[i][1]) / ( len(y) - no_of_excellent ) )
        
    def predict_prob_of_excellent(self, x):
        
        '''Initializing the variables'''
        excellent_cummulative = 1
        trash_cummulative = 1
        
        '''This method uses the array calculated in the train method to get the P(Excellent/features[]).
        Equation for 5 features looks like :
        P(Excellent / F1,F2,F3,F4,F5) = [ P(F1,F2,F3,F4,F5 / Excellent) * P(Excellent) ]  divided_by P(F1,F2,F3,F4,F5)
        
        This logic calculates: 
        P(F1,F2,F3,F4,F5 / Excellent ) = P(F1 / Excellent) * P(F2/ Excellent) * P(F3/ Excellent) * P(F4/ Excellent) * P(F5 / Excellent)
        and P(F1,F2,F3,F4,F5 / Trash) = P(F1 / Trash) * P(F2/ Trash) * P(F3/ Trash) * P(F4/ Trash) * P(F5 / Trash)'''
        for i in range(len(x)):
            if x[i] == True:
                excellent_cummulative = excellent_cummulative * self.feature_prob_distrib[i][0]
                trash_cummulative = trash_cummulative * self.feature_prob_distrib[i][1]
            else:
                excellent_cummulative = excellent_cummulative * ( 1 - self.feature_prob_distrib[i][0] )
                trash_cummulative = trash_cummulative * ( 1 - self.feature_prob_distrib[i][1] )
        
        '''P(F1,F2,F3,F4,F5) = P(F1,F2,F3,F4,F5 / Excellent ) * P(Excellent) + P(F1,F2,F3,F4,F5 / Trash) * (1 – P(Excellent))'''        
        denominator = 0.0
        denominator = ( ( excellent_cummulative * self.prob_excellent ) + ( trash_cummulative * ( 1 - self.prob_excellent ) ) )
        
        '''Calculate: P(Excellent / F1,F2,F3,F4,F5) = [ P(F1,F2,F3,F4,F5 / Excellent) * P(Excellent) ]  / P(F1,F2,F3,F4,F5)'''
        if denominator != 0:
            return ( float( self.prob_excellent * excellent_cummulative ) ) / denominator
