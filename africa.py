import numpy as np
import utilities as util
import sklearn.linear_model as linear
import sklearn.ensemble as ensemble
from sklearn import cross_validation
import pandas as pd


featuresList = ['PIDN','BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI','Depth','Ca','P','pH','SOC','Sand']
data = np.genfromtxt('../training.csv', names=True, delimiter=',', usecols=(featuresList))

featuresList = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
test = np.genfromtxt('../sorted_test.csv', names=True, delimiter=',', usecols=(featuresList))

data1 = np.copy(data)
targets = ['Depth','Ca','P','pH','SOC','Sand']

for target in targets:
    data, testa, features, fillVal = util.prepDataTrain(data1, target, featuresList, False, 10, False, True, 'mean', False, 'set')

    print 'Data preped'
    
    clf = ensemble.GradientBoostingRegressor(n_estimators=20)
    clf.fit(data[features].tolist(), data[target])

    test = util.prepDataTest(test, featuresList, True, fillVal, False, 'set')
    pred = clf.predict(test[features].tolist())
    print 'predicted'
    np.savetxt("pred_%s.csv" % (target), pred)




