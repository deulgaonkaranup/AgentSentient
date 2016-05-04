# Agent - Suggestions for buying a product in an e-commerce domain

Given a set of product features the task of the agent is to predict of the product could be purchased.

The agent outputs a probability of product being excellent by looking at a set of features. The Agent is an <a>ensemble</a> of different classification model:

1. Multinomial-Naive Bayes
2. Logistic Regression
3. Support Vector Machine

Agent, given a training set trains all these models and pass these models through different evaluation methods:
1. precision
2. recall
3. accuracy
4. doamin Evaluation - (In this case as we were dealing with buying products by optimizing cost we added money as one of criteria for domain evaluation)

Agent pics the one that performs best on the training set for classification task.
