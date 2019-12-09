import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
head = pd.read_csv('C:/Users/student/Desktop/dsdp.csv')
x=head.iloc[:,:-1].values
y=head.iloc[:,3].values
print(x)
print(y)

imp = Imputer(missing_values=np.nan, strategy='mean')
imp=imp.fit(x[:,1:3])
x[:,1:3]=imp.transform(x[:,1:3])
print(x)
