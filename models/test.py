#from ModelFactory import ModelFactory
import numpy as np

from models.LSTMModel import LSTMModel

#mk = ModelFactory()
#mk.load_models()

"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.SimpleNNModel import SimpleNNModel

iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SimpleNNModel()
model.define_model(4,10,1)
model.train(X_train, y_train, params={'learning_rate': 0.01, 'epochs': 5000})
predictions = model.predict(X_test)
predictions = np.round(predictions)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
from sklearn.metrics import classification_report
report = classification_report(y_test, predictions)
print(report)
"""
"""
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from models.LogisticRegressionModel import LogisticRegressionModel

data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1],
    'target': [0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

# 分离特征和目标列
X = df.drop('target', axis=1).values  # 特征列
y = df['target'].values # 目标列

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegressionModel()
model.train(X_train, y_train, params={'C': 1.0, 'max_iter': 200})
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
"""
import pandas as pd
data_raw = pd.read_excel('../datasets/2019.xlsx')
data_load = data_raw.iloc[:,1:].values
input_sequences = data_load[:,:-1]
output_sequences = data_load[:,-1]#.flatten().tolist()

model = LSTMModel()
model.train(input_sequences, output_sequences, params={'learning_rate': 0.01, 'epochs': 5000, 'input_size': 7,'output_size': 1 })
print("finshed training")
