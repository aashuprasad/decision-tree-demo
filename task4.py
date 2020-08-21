# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:42:38 2020

@author: aashu
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
SIZE=50
STEP=.02


"""
First step is to load the iris data set into variables x and y where x contains
the data (4 columns) and y contains the target.
"""
a=load_iris()
x=a['data']
y=a['target']


"""
Split the data into train and validation set so that we can see the performance of
our model on “previously unseen” instances
"""
train,test,train_lab,test_lab=train_test_split(x,y,test_size=.20,random_state=22)

"""
Sepal length,Sepal width, Petal length, Petal width are its features and the targets
are setosa (0), Iris virginica(1) and Iris versicolor(2)
"""
def plot(train,train_lab):
    for index,row in enumerate(train):
        if train_lab[index]==0:
            c='r'
            marker='>'
            q=plt.scatter(train[index,0],train[index,1],c=c,marker=marker,s=SIZE)
        elif train_lab[index]==1:
            c='b'
            marker='^'
            w=plt.scatter(train[index,0],train[index,1],c=c,marker=marker,s=SIZE)
        else:
            c='g'
            marker='<'
            e=plt.scatter(train[index,0],train[index,1],c=c,marker=marker,s=SIZE)
            plt.legend((q,w,e),('setosa','virginica','versicolor'),scatterpoints=1)




"""
Visualize the data set to see how the points are spread in space.
"""
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus



"""
This is the plot we obtain by plotting the first 2 feature points 
(of sepal length and width)
"""
plot(train,train_lab)
plt.show()
clf=DecisionTreeClassifier()
clf.fit(train,train_lab)


"""
We now get the tree-type structure for the same using the dot image from sklearn.
"""
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, feature_names=a.feature_names,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())



output=clf.predict(test)
print(accuracy_score(test_lab,output))
pca=PCA(n_components=2,whiten=True)
new_train=pca.fit_transform(train)
mean=new_train.mean(axis=0)
std=new_train.std(axis=0)
new_train=(new_train-mean)/std
clf.fit(new_train,train_lab)

plot(new_train,train_lab)
plt.show()
x_min,x_max=new_train[:,0].min()-1,new_train[:,0].max()+1
y_min,y_max=new_train[:,1].min()-1,new_train[:,1].max()+1


"""
This is the PCA plot of the data.We can see clearly the rectangular decision
boundary learned by our classifier. If a point falls in the blue surface it will be
classified as setosa(0) and so on.
"""
xx,yy=np.meshgrid(np.arange(x_min,x_max,STEP),np.arange(y_min,y_max,STEP))
Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)
cs=plt.contourf(xx,yy,Z,cmap=plt.cm.Paired)
plot(new_train,train_lab)
plt.show()




