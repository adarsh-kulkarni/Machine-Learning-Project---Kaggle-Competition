import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics

units = [5,10,50,100,200]
tanh = [0.70535,0.75422,0.77343,0.77504,0.77283]
sigmoid = [0.70012,0.75623,0.77393, 0.77450,0.77122]

plt.plot(units,tanh,'-r^',label='Tanh activation function')
#plt.show()
plt.plot(units,sigmoid,'-bo',label='Sigmoid activation function')
#plt.show()
#plt.plot(features,rfe,'-gx',label='RFE-SVM')
plt.legend(loc='lower right',prop={'size':10})
plt.title('Accuracy of Neural Network Classifier as function of number of hidden layers')
plt.xlabel('Number of hidden units-->')
plt.ylabel('Accuracy ---->')
#plt.xscale('log')
plt.show()


estimators = [10,50,100,200,300,400,500,1000]
accuracy=[0.70103,0.74165,0.75171,0.75282,0.75191,0.75654,0.75613,0.75654]
plt.plot(estimators,accuracy,'-bo')
#plt.show()
#plt.plot(features,ksub_l,'-bo',label='L1-SVM using subsamples')
#plt.show()
#plt.plot(features,rfe_svm,'-gx',label='RFE-SVM')
plt.legend(loc='lower right',prop={'size':10})
plt.title('Accuracy of the Random Forest Classifier as function of number of estimators')
plt.xlabel('Number of estimators ---->')
plt.ylabel('Accuracy ---->')
#plt.xscale('log')
plt.show()

estimators = [10,50,100,300,500]
accuracy=[0.69522,0.72798,0.73321,0.73390,0.73382]
plt.plot(estimators,accuracy,'-bo')
#plt.show()
#plt.plot(features,ksub_l,'-bo',label='L1-SVM using subsamples')
#plt.show()
#plt.plot(features,rfe_svm,'-gx',label='RFE-SVM')
plt.legend(loc='lower right',prop={'size':10})
plt.title('Accuracy of the Gradient Boosting Classifier')
plt.xlabel('Number of estimators ---->')
plt.ylabel('Accuracy ---->')
#plt.xscale('log')
plt.show()

hidden=[5,10,100,200,300]
accuracy=[0.70143,0.74588,0.76488,0.77051,0.76317]
plt.plot(hidden,accuracy,'-bo')
plt.xlabel('Number of hidden units of both the layers ---->')
plt.ylabel('Accuracy---->')
plt.title('Accuracy as a function of number of hidden units')
plt.show()

