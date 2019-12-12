from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris_dataset = load_iris()
irisTrainDatas, irisTestDatas, irisTrainLabels, irisTestLabels= train_test_split(iris_dataset.data, iris_dataset.target,random_state=0)
DecisionTree = DecisionTreeClassifier()
DecisionTree.fit(irisTrainDatas, irisTrainLabels)
result= DecisionTree.predict(irisTestDatas)
print(result)
print(irisTestLabels)

dcx= DecisionTree.score(irisTestDatas,irisTestLabels)
print('Do chinh xac cua bo phan loai la: ',dcx)
print(irisTrainDatas)