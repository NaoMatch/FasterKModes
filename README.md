# FasterKModes
　Kmodes と KPrototypes を実装した。

　この実装では、KModes および KPrototypes のより高速な実装を目的としており、精度などは既存実装と特に変わりはない。特に、処理時間のかかる距離行列計算と最頻値計算の処理を C コードで実装することにより高速化を図った。また現状、計算できる距離の種類はカテゴリ特徴量に対してはHamming距離、数値特徴量に対してはEuclid距離のみである。数値特徴量はGEMMによる実装で対処している。

　初期化、カテゴリ/数値特徴量の距離行列計算はUser Defined Functionを設定することができる。

# インストール方法
```bash
pip install FasterKModes
```

# How to use

## FasterKModes
```python
# ライブラリ内で動的に生成したファイルの削除
from FasterKModes import delete_all_generated_so
delete_all_generated_so()
```

```python
import numpy as np
from FasterKModes import FasterKModes

# データ作成
N = 1100      # 総データ数
K = 31        # 特徴量数
C = 8         # クラスタ数
X = np.random.randint(0, 256, (N, K)).astype(np.uint8)
X_train = X[:1000, :]
X_test  = X[1000:, :]

# FasterKModes の実行（random 初期化）
kmodes = FasterKModes(n_clusters=C, init="random", n_init=10)
kmodes.fit(X_train)
labels = kmodes.predict(X_test)

print("KModes クラスタリング結果:", labels)
```

## FasterKPrototypes
```python
import numpy as np
from FasterKModes import FasterKPrototypes

# データ作成
N = 1100      # 総データ数
K = 31        # 特徴量数
C = 8         # クラスタ数
X = np.random.randint(0, 256, (N, K)).astype(np.uint8)
X_train = X[:1000, :]
X_test  = X[1000:, :]

# カテゴリカル特徴量のインデックス例（例：2列目、4列目、6列目、8列目、10列目）
categorical_features = [1, 3, 5, 7, 9]

# FasterKPrototypes の実行（random 初期化）
kproto = FasterKPrototypes(n_clusters=C, init="random", n_init=10)
kproto.fit(X_train, categorical=categorical_features)
labels = kproto.predict(X_test)

print("KPrototypes クラスタリング結果:", labels)
```

## FasterKModes: カスタム初期化・カスタム距離計算の例
```python
import numpy as np
from FasterKModes import FasterKModes

# カスタム初期化関数の例
def custom_init(X, n_clusters):
    # ランダムに n_clusters 個のサンプルを初期セントロイドとして選択
    indices = np.random.choice(len(X), n_clusters, replace=False)
    return X[indices, :]

def custom_caetgorical_measure(x_cat, c_cat):
    # 変数名は固定
    # x_cat は np.array かつ, x_cat.ndim = 1
    # c_cat は np.array かつ, c_cat.ndim = 1
    return np.sum(x_cat != c_cat)

# データ作成
N = 1100
K = 31
C = 8
X = np.random.randint(0, 256, (N, K)).astype(np.uint8)
X_train = X[:1000, :]
X_test  = X[1000:, :]

# カスタム初期化を利用した KModes の実行
kmodes_custom = FasterKModes(
        n_clusters = C, 
        n_init = 10, 
        init = custom_init, 
        categorical_measure = custom_caetgorical_measure
    )
kmodes_custom.fit(X_train)
labels_custom = kmodes_custom.predict(X_test)

print("カスタム初期化 KModes 結果:", labels_custom)
```

## FasterKPrototypes: カスタム初期化・カスタム距離計算の例
```python
import numpy as np
from FasterKModes import FasterKPrototypes

# カスタム初期化関数の例
def custom_init(Xcat, Xnum, n_clusters):
    # ランダムに n_clusters 個のサンプルを初期セントロイドとして選択
    indices = np.random.choice(len(Xcat), n_clusters, replace=False)
    return Xcat[indices, :], Xnum[indices, :]

def custom_caetgorical_measure(x_cat, c_cat):
    # 変数名は固定
    # x_cat は np.array かつ, x_cat.ndim = 1
    # c_cat は np.array かつ, c_cat.ndim = 1
    return np.sum(x_cat != c_cat)

def custom_numerical_measure(x_num, c_num):
    # 変数名は固定
    # x_num は np.array かつ, x_num.ndim = 1
    # c_num は np.array かつ, c_num.ndim = 1
    return np.linalg.norm(x_num - c_num)

# データ作成
N = 1100
K = 31
C = 8
X = np.random.randint(0, 256, (N, K)).astype(np.uint8)
X_train = X[:1000, :]
X_test  = X[1000:, :]

# カテゴリカル特徴量のインデックス例（例：2列目、4列目、6列目、8列目、10列目）
categorical_features = [1, 3, 5, 7, 9]

# カスタム初期化を利用した kproto の実行
kproto_custom = FasterKPrototypes(
        n_clusters = C, 
        n_init = 10, 
        init = custom_init, 
        categorical_measure = custom_caetgorical_measure, 
        numerical_measure = custom_numerical_measure
    )
kproto_custom.fit(X_train, categorical=categorical_features)
labels_custom = kmodes_custom.predict(X_test)

print("カスタム初期化 KPrototypes 結果:", labels_custom)
```
