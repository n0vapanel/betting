import pandas as pd
import numpy as np
import os
base = os.path.dirname(os.path.abspath(__file__))
md = pd.read_csv(os.path.join(base, "raw_mdabbert", "ufc-master.csv"))
md["date"] = pd.to_datetime(md["date"])

fighters = ["Islam Makhachev", "Alex Pereira", "Jon Jones", "Kamaru Usman", "Charles Oliveira",
            "Leon Edwards", "Cory Sandhagen", "Ciryl Gane"]
for f in fighters:
    sub = md[(md["R_fighter"] == f) | (md["B_fighter"] == f)].sort_values("date")
    if len(sub) == 0:
        print(f, ": no rows"); continue
    ranks = []
    for _, r in sub.iterrows():
        if r["R_fighter"] == f:
            ranks.append((r["date"].date(), r["R_match_weightclass_rank"], r["R_age"], r["weight_class"]))
        else:
            ranks.append((r["date"].date(), r["B_match_weightclass_rank"], r["B_age"], r["weight_class"]))
    print(f, "->", ranks)
