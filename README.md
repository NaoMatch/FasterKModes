# FasterKModes
Faster Implementation of KModes and KPrototypes

# How to use

インストール
```
git clone https://github.com/NaoMatch/FasterKModes
pip install requirements.txt
```

実行
```python
print("*"*100)
print("*"*100)
print("*"*100)
print("Kmodes")
import numpy as np
from FasterKModes import FasterKModes
from FasterKPrototypes import FasterKPrototypes

N = 1100 # 行数
K = 31 # 列数
C = 8 # クラスタ数

X = np.random.randint(0, 256, (N, K)).astype(np.uint8)
X_train = X[:1000,:]
X_test = X[1000:,:]

# Kmodes
fKModes = FasterKModes(n_clusters=C, init="random", n_init=10)
fKModes.fit(X_train)
labels = fKModes.predict(X_test)

# Kprototypes
categorical_features = [1, 3, 5, 7, 9]
fKProto = FasterKPrototypes(n_clusters=C, init="random", n_init=10)
fKProto.fit(X_train, categorical=categorical_features)
labels = fKProto.predict(X_test)
```
