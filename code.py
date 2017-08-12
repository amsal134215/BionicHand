import csv
import numpy
import pywt

training_data_filename = 'train.txt'
test_data_filename = 'test.txt'

training_data = []
training_data_labels = []

test_data = []
test_data_labels = []

_data_data = []
data_data_ = []

with open(training_data_filename, newline='') as training_csvfile:
    training_data_file = csv.reader(training_csvfile, delimiter=',', quotechar='|')
    for row in training_data_file:
        if len(row) > 0:
            t_var = row[0:-1]
            _data = [float(i) for i in t_var]
            _label = row[-1]
            training_data_labels.append(_label)

            i=0
            while i < 10:
                _data_data.append(_data[i:i+10])
                data_data_ += _data[i:i+10]
                i+=1;

            training_data.append(_data)
            _data_data = None
            _data_data = []
            data_data_ = None
            data_data_ = []

with open(test_data_filename, newline='') as test_csvfile:
    test_data_file = csv.reader(test_csvfile, delimiter=',', quotechar='|')
    for row in test_data_file:
        if len(row) > 0:
            t_var = row[0:-1]
            _data = [float(i) for i in t_var]
            _label = row[-1]
            test_data_labels.append(_label)

            i = 0
            while i < 10:
                _data_data.append(_data[i:i+10])
                data_data_ += _data[i:i + 10]
                i += 1;

            test_data.append(_data)
            _data_data = None
            _data_data = []
            data_data_ = None
            data_data_ = []

#Fourier Transform
#training_data = numpy.fft.fft2(training_data)
#training_data = numpy.real(training_data)
#test_data = numpy.fft.fft2(test_data)
#test_data = numpy.real(test_data)


training_data_arr = numpy.array(training_data)
print("Training Data Shape = ", training_data_arr.shape)
test_data_arr = numpy.array(test_data)
print("Test Data Shape = ", test_data_arr.shape)

#Core Code

clf = []
pred = []
result = []

from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


#clf.append(tree.DecisionTreeClassifier(min_samples_split=20,min_samples_leaf=10))
#clf.append(RandomForestClassifier(n_estimators=100,min_samples_split=10))
clf.append(ExtraTreesClassifier(n_estimators=100,min_samples_split=10))
#clf.append(GradientBoostingClassifier(n_estimators=100,min_samples_split=10))
#clf.append(svm.SVC(kernel = 'linear'))



index = 0
while index < len(clf):
    clf[index] = clf[index].fit(training_data, training_data_labels)
    pred.append(clf[index].predict(test_data))
    #accuracy.append(accuracy_score(pred[index], test_data_labels))
    index+=1;


#clf = clf.fit(training_data, training_data_labels)
#pred = clf.predict(test_data)

print ("Training Data =",len(training_data))
print ("Test Data =",len(test_data))
#print (pred)
#print ("Accuracy =",accuracy_score(pred, test_data_labels))

index = 0;
while index < len(pred[0]):

    gvote = 0
    ivote = 0
    for _pred in pred:

        if _pred[index] == 'index':
            ivote += 1
        else:
            gvote += 1

    if ivote > gvote:
        result.append("index")
    else:
        result.append("group")

    index+=1

print ("Accuracy =",accuracy_score(result, test_data_labels))

exit()

fresult = []
ftest_data_labels = []

r_i_count=0
r_g_count=0
t_i_count=0
t_g_count=0

while i < len(result):
    j=0
    while i < len(result) and j < 5:
        if result[i] == 'index':
            r_i_count+=1
        else:
            r_g_count+=1

        if test_data_labels[i] == 'index':
            t_i_count += 1
        else:
            t_g_count += 1

        j+=1
        i+=1

    if r_i_count<r_g_count:
        fresult.append("group")
    else:
        fresult.append("index")


    if t_i_count<t_g_count:
        ftest_data_labels.append("group")
    else:
        ftest_data_labels.append("index")

    r_i_count = 0
    r_g_count = 0
    t_i_count = 0
    t_g_count = 0

print("Accuracy =", accuracy_score(fresult, ftest_data_labels))
print(fresult)
print(ftest_data_labels)
