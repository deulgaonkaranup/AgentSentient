'''
The agent base class as well as a baseline agent.
'''

from abc import abstractmethod


class Agent(object):
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return "Agent_" + self.name
    
    def will_buy(self, value, price, prob):
        """Given a value, price, and prob of Excellence,
        return True if you want to buy it; False otherwise.
        The rational agent. Do NOT change or override this."""
        return value*prob > price

    @abstractmethod
    def train(self, X, y):
        """Train the agent to learn a function that
        can predict, probabilistically, whether
        a product is Excellent or Trash.
        Override this method.

        Parameters:
        -----------
        X: A matrix (2D numpy array) where rows correspond to
           products and columns correspond to feature values of
           those products.
        y: A 1D numpy array where each entry corresponds to
           whether a product is Excellent or Trash. The ith
           entry in y corresponds to the ith row in X.
        """

    @abstractmethod
    def predict_prob_of_excellent(self, x):
        """Given a single product, predict and return
        the probability of the product being Excellent.
        Override this method.

        Parameters:
        -----------
        x: A 1D numpy array that corresponds to a single product.        
        """

class FixedProbAgent(Agent):
    """No matter what the product is, it believes the
    probability of being Excellent is a fixed value."""

    def __init__(self, name, fixed_prob):
        super(FixedProbAgent, self).__init__(name)
        self.fixed_prob = fixed_prob

    def train(self, X, y):
        """Simply ignore everything; do nothing."""
        pass

    def predict_prob_of_excellent(self, x):
        """Simply ignore x and return the fixed probability value
        no matter what."""
        return self.fixed_prob

