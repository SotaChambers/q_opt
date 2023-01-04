import pandas as pd
import numpy as np
from pyqubo import Array

# --------- PARAMS ---------- #

N = 5
df = pd.read_csv("data/distance.csv", index_col=0)
Q = np.array(df)

# --------- /PARAMS ---------- #

# 変数の定義
x = Array.create("x", shape=(N, N), vartype="BINARY")

# 目的関数の定義
cost = 0
for i in range(N):
    for j in range(N):
        for k in range(N):
            Q[i][j] * x[i][k] * x[j][k+1]