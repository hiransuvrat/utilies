import numpy as np
import utilities as util
import sklearn.linear_model as linear
import sklearn.ensemble as ensemble
from sklearn import cross_validation
import pandas as pd
import sys
import sklearn.feature_selection as fs
import sklearn.svm as svm
import analysis as an
import matplotlib.pyplot as plt
import sklearn.cluster as kmeans
from sklearn import preprocessing
import pandas as pd
from numpy.lib import recfunctions

def clusterData(data, fields, removeFeatures= True, enc = 0, clust = 0, isTrain = True, numberCluster = 20):
    #Scale the data
    min_max_scaler = preprocessing.MinMaxScaler()
    scaledData = min_max_scaler.fit_transform(data[fields].tolist())
    #scaler = preprocessing.StandardScaler().fit(data[fields].tolist())
    #scaledData = scaler.transform(data[fields].tolist())

    #If training then fit the data first
    if isTrain:
        clust = kmeans.KMeans(n_clusters = numberCluster)
        clust.fit(scaledData)
        
    #Get predicted clusters
    getCluster = clust.predict(scaledData)
    
    blankArr = np.zeros((len(getCluster),1))
    blankArr[:, 0] = getCluster
    
    if isTrain:
        enc = preprocessing.OneHotEncoder()
        enc.fit(blankArr)

    #Do dummy encoding
    cluster = enc.transform(blankArr).toarray()

    #Give names to the cluster
    newColumns = ['1'] * len(cluster[0])
    for i in range(len(cluster[0])):
        newColumns[i] = 'cluster_' + str(i)
        data = recfunctions.append_fields(data, newColumns[i], cluster[:, i])
    
    if removeFeatures:
        names = []
        for field in fields:
            names = list(data.dtype.names)
            if field in names:
                names.remove(field)
        data = data[names]

    return data, clust, enc, newColumns

headersTest = np.genfromtxt(sys.argv[2], delimiter=',', names=True)

#Columns to be picked from training file
pickTrain = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI','Depth','Ca','P','pH','SOC','Sand']

data = np.genfromtxt(sys.argv[1], names=True, delimiter=',') #, usecols=(pickTrain))

#Column to be picked from test file
#pickTest = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
pickTest = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
test = np.genfromtxt(sys.argv[2], names=True, delimiter=',')#, usecols=(pickTest))

ids = np.genfromtxt(sys.argv[2], dtype=str, skip_header=1, delimiter=',', usecols=0)

#Features to train model on
#featuresList = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
featuresList = np.array(headersTest.dtype.names)
featuresList = featuresList[2:-5]
clusterFields = ['TMAP']
#Keep a copy of train file for later use
data1 = np.copy(data)
test1 = np.copy(test)

#Dependent/Target variables
targets = ['Ca','P','pH','SOC','Sand']
#targets = ['P']

#Prepare empty result
df = pd.DataFrame({"PIDN": ids, "Ca": test['PIDN'], "P": test['PIDN'], "pH": test['PIDN'], "SOC": test['PIDN'], "Sand": test['PIDN']})
testa = 0
             
for target in targets:
    data = np.copy(data1)
    test = np.copy(test1)
    #bs = cross_validation.Bootstrap(len(data), random_state=0)
    
             
    #Prepare data for training
    
    delList = np.array([])
    clf = 0
    print target

    '''
    for i in range(len(data)):
        if data[target][i] > (data[target].mean() + 2*data[target].std()) or data[target][i] < (data[target].mean() - 2*data[target].std()):
            delList = np.append(delList, i)
            print (data[target].mean() - 1*data[target].std()), data[target].std()
    '''

    #clf = linear.BayesianRidge(normalize=True, verbose=True, tol=.01)
    clf = svm.SVR(C=10000.0, kernel='rbf', degree=1)
    data = np.delete(data, delList, 0)
    data, testa, features, fillVal = util.prepDataTrain(data, target, featuresList, False, 30, False, True, 'mean', False, 'set')
    #data, clust, enc, newCol = clusterData(data, clusterFields, True)
    #testa, clust, enc, newCol = clusterData(testa, pickTest, True, enc, clust, False)
    #features = np.concatenate((features, newCol))
    
    #sel = fs.SelectKBest(fs.f_regression, k=2000)
    #data = np.array(sel.fit_transform(data[features].tolist(), data1[target]))
    #print data.shape

    #Use/tune your predictor
 
    #scores = cross_validation.cross_val_score(clf, data[features].tolist(), data[target], cv=5, scoring='mean_squared_error')
    #scores = np.array(cross_validation.cross_val_score(clf, data[features].tolist(), data[target], cv=5, scoring='mean_squared_error'))
    #print (-1 * scores), (-1  * scores.sum()/5)
    #continue
    clf.fit(data[features].tolist(), data[target])

    #Prepare test data
    #testa, clust, enc, newCol = clusterData(testa, clusterFields, True, enc, clust, False)
    test = util.prepDataTest(test, features, True, fillVal, False, 'set')

    #Get predictions
    pred = clf.predict(test[features].tolist())
    #pred1 = clf.predict(testa[features].tolist())
    #pred = clf.predict(data[features].tolist())
    #an.plotData(data[target], pred)

    #Store results
    df[target] = pred
    continue

    print (((((testa[target] - pred1) ** 2).sum())/len(pred1))**.5)
    abse = pred - data[target]
    absp = pred1 - testa[target]
    abse = abse[abse < 1]
    absp = absp[absp < 1]
    abse = abse[abse > -1]
    absp = absp[absp > -1]

    plt.hist(abse, bins=100, color='blue')
    plt.hist(absp, bins=100, color='red')
    plt.show()

#np.savetxt("testActual.csv", testa, delimiter=",", fmt='%.4e')
df.to_csv("predictions.csv", index=False, cols=["PIDN","Ca","P","pH","SOC","Sand"])


