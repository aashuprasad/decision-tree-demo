# decision-tree
In this post we will deal with Decision Tree Classifier(DTC). DTC can be simply assumed as a series of if-else questions asked at each decision node. For example if I need to know a given object is a car or not, I will ask a series of questions which can be interpreted as if-else statements. I ask, if the object had 4 tyres. If yes then does it have a box-like structure around it. If yes then does it have a steering-wheel and gear box. If answer to all of these decision questions is yes probably the object is car. Else it may not be a car.

This is a very small example of a decision tree. This is used only for illustration purpose and I hope it drives the message that DTC is nothing but a series of if-else situations at each decision node.

Now we will look into the iris data set and try to implement decision tree classifier algorithm on the same. By implement, most the classifiers have been coded and integrated in scikit sklearn package. We will all we need by using sklearn
