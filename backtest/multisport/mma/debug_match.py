import pandas as pd
import numpy as np
import os
base = os.path.dirname(os.path.abspath(__file__))

raw = pd.read_csv(os.path.join(base, "raw", "raw_total_fight_data.csv"), sep=";")
raw.columns = [c.strip() for c in raw.columns]
raw["date"] = pd.to_datetime(raw["date"])
print("rajeevw date dtype:", raw["date"].dtype, raw["date"].head(3).tolist())
print("rajeevw fighters sample:", raw["R_fighter"].head(5).tolist(), raw["B_fighter"].head(5).tolist())

md = pd.read_csv(os.path.join(base, "raw_mdabbert", "ufc-master.csv"))
md["date"] = pd.to_datetime(md["date"])
print("md date dtype:", md["date"].dtype, md["date"].head(3).tolist())
print("md fighters sample:", md["R_fighter"].head(5).tolist(), md["B_fighter"].head(5).tolist())

r_names = set(pd.concat([raw["R_fighter"], raw["B_fighter"]]).str.strip().dropna().unique())
m_names = set(pd.concat([md["R_fighter"], md["B_fighter"]]).str.strip().dropna().unique())
print("name overlap:", len(r_names & m_names), "of", len(r_names))
print("overlap sample:", sorted(r_names & m_names)[:10])
r_dates = set(raw["date"].dt.normalize())
m_dates = set(md["date"].dt.normalize())
print("date overlap:", len(r_dates & m_dates))
