import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

baseDeDados = pd.read_csv('svbr.csv', delimiter=';')
X = baseDeDados.iloc[:,:].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:,1:3])
X = imputer.transform(X[:,1:3]).astype(str)
X = np.insert(X, 0,baseDeDados.iloc[:,0].values, axis=1)

print(X)
