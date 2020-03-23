from __future__ import division
import numpy as np
import pylab as plt
from linearclassifier import linear_predict, perceptron_update, plot_predictions, log_reg_train
from scipy.io import loadmat


# In[ ]:


# load cardio data from MATLAB data file

variables = dict()
loadmat('processedCardio.mat', variables)

train_labels = variables['trainLabels'].ravel() - 1 # the original MATLAB data was 1-indexed
test_labels = variables['testLabels'].ravel() - 1
train_data = variables['trainData']
test_data = variables['testData']

# get data dimensions and sizes
num_dim, num_train = train_data.shape
_, num_test = test_data.shape

classes = np.unique(train_labels)
num_classes = len(classes)


# In[ ]:


plt.hist(train_labels)
plt.xlabel('Class')
plt.ylabel('Number of Examples')
plt.show()


# In[ ]:


# Perceptron experiment

epochs = 40
lambda_val = 1

model = { 'weights': np.zeros((num_dim, num_classes)) }
params = {'lambda': lambda_val}
train_accuracy = np.zeros(epochs)
test_accuracy = np.zeros(epochs)

for epoch in range(epochs):
    # first measure training and testing accuracy
    predictions = linear_predict(train_data, model)
    train_accuracy[epoch] = np.sum(predictions == train_labels) / num_train

    predictions = linear_predict(test_data, model)
    test_accuracy[epoch] = np.sum(predictions == test_labels) / num_test

    # run perceptron training
    mistakes = 0
    for i in range(num_train):
        correct = perceptron_update(train_data[:, i], model, params, train_labels[i])
        
        if not correct:
            mistakes += 1
    
    print("Finished epoch %d with %d mistakes." % (epoch, mistakes))


# In[ ]:


# Plot results of perceptron

train_line = plt.plot(range(epochs), train_accuracy, label="Training")
test_line = plt.plot(range(epochs), test_accuracy, label="Testing")
plt.title('Cardiotocography Data')
plt.xlabel('Iteration')
plt.ylabel('Perceptron Accuracy')
plt.legend()

plt.show()

print("Train Accuracy: %f" % train_accuracy[epochs-1])
print("Test Accuracy: %f" % test_accuracy[epochs-1])


# In[ ]:


# Logistic regression gradient check

# first check if the gradient and objective function are consistent with each other
_ = log_reg_train(train_data, train_labels, 
              {'weights': np.random.randn(num_dim * num_classes)}, check_gradient=True)


# In[ ]:


# Logistic regression experiment

model = {'weights': np.zeros((num_dim, num_classes))}
    
model = log_reg_train(train_data, train_labels, model)
        
train_predictions = linear_predict(train_data, model)
train_accuracy = np.sum(train_predictions == train_labels) / num_train

test_predictions = linear_predict(test_data, model)
test_accuracy = np.sum(test_predictions == test_labels) / num_test
    
print("Train Accuracy: %f" % train_accuracy)
print("Test Accuracy: %f" % test_accuracy)


# In[ ]:





# In[ ]:




