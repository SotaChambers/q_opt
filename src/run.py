import pandas as pd
import numpy as np
from pyqubo import Array, Constraint, Placeholder
from dwave.system import DWaveSampler, EmbeddingComposite
from openjij import SQASampler

from utils.utils import load_yaml

# --------- PARAMS ---------- #
select_solver = "openjij"
N = 5
df = pd.read_csv("data/distance.csv", index_col=0)
Q = np.array(df)

credential_path = "config/d-wave_credential.yml"
credential_cfg = load_yaml(credential_path)
token = credential_cfg["TOKEN"]
endpoint = credential_cfg["ENDPOINT"]
# --------- /PARAMS ---------- #

# 変数の定義
x = Array.create("x", shape=(N, N), vartype="BINARY")

# 目的関数の定義(%Nで繰り返す)
cost = 0
for i in range(N):
    for j in range(N):
        for k in range(N):
            cost += Q[i][j] * x[i][k] * x[j][(k+1)%N]

# 制約条件1: 各都市に1回は訪問しなければならない
constr_1 = 0
for i in range(N):
    constr_1 += (np.sum(x[i]) - 1) ** 2

# 制約条件2: 1度に訪れる都市は1つでなければならない
constr_2 = 0
for k in range(N):
    constr_2 += (np.sum(x.T[k]) - 1) ** 2

# コスト関数
# Tips: Placeholder("lam")はpenalty係数λで，コスト関数をコンパイルした後もハイパーパラメータとして変更できるようにしている
# Tips: penalty係数を変えることで制約条件の影響力を調整できる
# Tips: Constraintは制約条件を守っているかチェックする機能を持っている
cost_func = cost \
            + \
            Placeholder("lam") * Constraint(constr_1, label="constr_1") \
            + \
            Placeholder("lam") * Constraint(constr_2, label="constr_2")

model = cost_func.compile()

# ハイパーパラメータをコスト関数に渡して辞書形式のQUBOを生成
feed_dict = {"lam": 1000.0}
qubo, offset = model.to_qubo(feed_dict=feed_dict)

# ソルバーの定義
dw_sampler = DWaveSampler(
    solver="DW_2000Q_6",
    token=token,
    endpoint=endpoint
)

# 問題の埋め込み
if select_solver == "openjij":
    sampler = SQASampler()
elif select_solver == "d-wave":
    sampler = EmbeddingComposite(dw_sampler)
else:
    raise NotImplementedError("only openjij and d-wave are supported")
# 実行
sampleset = sampler.sample_qubo(qubo, num_reads=10)

print(type(sampleset.record))
print(sampleset.record)