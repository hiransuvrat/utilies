import numpy as np
import utilities as util
import sklearn.linear_model as linear
import sklearn.ensemble as ensemble
from sklearn import cross_validation
import pandas as pd

data = np.genfromtxt('../training.csv', names=True, delimiter=',')
test = np.genfromtxt('../test.csv', names=True, delimiter=',')

featuresList = ['DER_mass_MMC','DER_mass_transverse_met_lep','DER_mass_vis','DER_pt_h','DER_deltaeta_jet_jet','DER_mass_jet_jet','DER_prodeta_jet_jet','DER_deltar_tau_lep','DER_pt_tot','DER_sum_pt','DER_pt_ratio_lep_tau','DER_met_phi_centrality','DER_lep_eta_centrality','PRI_tau_pt','PRI_tau_eta','PRI_tau_phi','PRI_lep_pt','PRI_lep_eta','PRI_lep_phi','PRI_met','PRI_met_phi','PRI_met_sumet','PRI_jet_num','PRI_jet_leading_pt','PRI_jet_leading_eta','PRI_jet_leading_phi','PRI_jet_subleading_pt','PRI_jet_subleading_eta','PRI_jet_subleading_phi','PRI_jet_all_pt']


var = 'DER_mass_MMC'
cutoffUpper = data[var].mean() + 2*data[var].std()
cutoffLower = data[var].mean() - 2*data[var].std()
print data.shape
data = data[:, (data[var] < cutoffUpper)]
print data.shape
data = data[:, (data[var] > cutoffLower)]
print data.shape
#data = data[data[var] < (data[var].mean() + 2*data[var].std()) and data[var] > (data[var].mean() - 2*data[var].std())]
'''
for i in range(len(data)):
    if data[var][i] > (data[var].mean() + 2*data[var].std()) or data[var][i] < (data[var].mean() - 2*data[var].std()):
        delList = np.append(delList, i)
    data = np.delete(data, delList, 0)
'''

data, testa, features, fillVal = util.prepDataTrain(data, 'Label', featuresList, False, 20, True, True, 'mean', False, 'set')

print 'Data preped'

#clf = ensemble.GradientBoostingClassifier(n_estimators=20,min_samples_leaf=40,max_features=15,verbose=1)
clf = ensemble.RandomForestClassifier(n_estimators=25,min_samples_leaf=60,max_features=15,verbose=1, n_jobs=3)
#clf = linear.LogisticRegression()
#clf = linear.BayesianRidge()

clf.fit(data[features].tolist(), data['Label'])
#scores = cross_validation.cross_val_score(clf, data[features].tolist(), data['Label'], cv=3, scoring='f1')
#print scores
#exit()
#print clf.score(testa[features].tolist(), testa['Label'])
for i in range(len(clf.feature_importances_)):
    print i, featuresList[i], clf.feature_importances_[i] 

print 'fitted'

ids = test['EventId'].astype(int)
X_test = util.prepDataTest(test, featuresList, True, fillVal, False, 'set')

#data = pd.read_csv("../test.csv")
#X_test = data.values[:, 1:]



d = clf.predict_proba(X_test.tolist())[:,1]
#e = clf1.predict_proba(X_test.tolist())[:, 1]
#f = clf2.predict_proba(X_test.tolist())[:, 1]

pcut = np.percentile(d, 85)
r = np.argsort(d) + 1
p = np.empty(len(X_test), dtype=np.object)
#print p, d, pcut
p[d > pcut] = 's'
p[d <= pcut] = 'b'

df = pd.DataFrame({"EventId": ids, "RankOrder": r, "Class": p})
df.to_csv("predictions.csv", index=False, cols=["EventId", "RankOrder", "Class"])
