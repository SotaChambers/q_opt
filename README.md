# Ref
[Link1](https://leadinge.co.jp/rd/2021/06/21/966/)



# D-Wave: Procedure
1. [D-Wave Leap](https://cloud.dwavesys.com/leap/login/?next=/leap/)のアカウント作成
2. DashboradでAPI Tokenを控えておく
3. [D-Wave Ocean SDK](https://github.com/dwavesystems/dwave-ocean-sdk)をインストール
Ocean SDKは，D-Waveの量子コンピューターをプログラミングするためのオープンソースのPythonツールセット
```
$ pip install dwave-ocean-sdk
```
# 巡回セールスマン問題
## 定数と変数
- $Q_{i,j}$
出発都市 $i$ から都市 $j$ までの移動距離
- $x_{i,k}$
都市 $i$ を時刻 $k$ に訪れるか訪れないかのバイナリ変数

## 目的関数
QUBO行列を以下で表す．
```math
\sum_{i,j}\sum_{k=1}^N Q_{i,j}x_{i,k}x_{j,k+1}
```

## 制約条件
### 各都市に1回は訪問しなければならない
全ての時刻 $k$ である1都市 $i$ について考えた時，$x$ は1でなければならない
```math
\forall i:  \sum_{k=1}^Nx_{i, k} = 1
```
D-Waveマシン用に定式化する
```math
\lambda\sum_{i=1}^N(\sum_{k=1}^Nx_{i, k} - 1)^2
```

### 1度に訪れる都市は1つでなければならない
ある時刻 $k$ で全ての都市について考えた時，$x$ は1でなければならない
```math
\forall k: \sum_{i=1}^Nx_{i, k} = 1
```
D-Waveマシン用に定式化する
```math
\lambda\sum_{k=1}^N(\sum_{i=1}^Nx_{i, k} - 1)^2
```

## コスト関数
最小化したい関数は，目的関数 + 制約条件1 + 制約条件2
```math
\sum_{i,j}\sum_{k=1}^N Q_{i,j}x_{i,k}x_{j,k+1} + \lambda\sum_{i=1}^N(\sum_{k=1}^Nx_{i, k} - 1)^2 + \lambda\sum_{k=1}^N(\sum_{i=1}^Nx_{i, k} - 1)^2
```

# 実装
## 各種インストール
[pyqubo](https://pyqubo.readthedocs.io/en/latest/)をインストールする(変数の定義に必要)<br>
:warning: dwave-ocean-sdkに含まれているぽい
```
$ pip install pyqubo
```
`numpy`, `pandas`, `pyyaml`もインストール

## 実行
```
$ bash scripts/run.sh
```
- output1

```txt
[([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], -9900., 1, 0.  )
 ([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], -9890., 1, 0.  )
 ([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], -9870., 1, 0.  )
 ([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], -7880., 1, 0.  )
 ([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], -9890., 1, 0.04)
 ([0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], -7900., 1, 0.04)
 ([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], -6910., 1, 0.  )
 ([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], -7830., 1, 0.04)
 ([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0], -6860., 1, 0.04)
 ([0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], -4840., 1, 0.  )]
```