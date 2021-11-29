import pandas as pd

services = ["s1", "s2", "s3"]
iterables = [services, ["lawyer", "rating", "time"]]

data = [
        ["pepo", 0.8, 10, "Diego", 0.4, 8, "Jose", 0.9, 25],
        ["joaco", 0.4, 15, "Maria", 0.1, 20]
]

index = pd.MultiIndex.from_product(iterables)
df = pd.DataFrame(data=data, columns = index)
df.loc[2, "s1"] = ["jac", 1, 0]
print(df)