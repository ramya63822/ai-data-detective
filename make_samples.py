"""Generate three deliberately different sample datasets used for demos and tests."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "sample_data"


def sales(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2023-01-01", "2026-10-03", freq="D")
    n = 14000
    d = rng.choice(days, n)
    d = pd.to_datetime(d)
    month = d.month.to_numpy()
    season = 1 + 0.35 * np.isin(month, [11, 12]) - 0.15 * np.isin(month, [1, 2])
    keep = rng.random(n) < (season / season.max())
    d = d[keep]
    n = len(d)
    region = rng.choice(["North", "South", "East", "West"], n, p=[0.24, 0.22, 0.20, 0.34])
    category = rng.choice(["Laptops", "Phones", "Accessories", "Tablets", "Monitors"], n, p=[0.2, 0.3, 0.3, 0.1, 0.1])
    segment = rng.choice(["Consumer", "Corporate", "Small Business"], n, p=[0.5, 0.3, 0.2])
    base = {"Laptops": 1100, "Phones": 650, "Accessories": 60, "Tablets": 420, "Monitors": 280}
    unit_price = np.array([base[c] for c in category]) * rng.lognormal(0, 0.12, n)
    qty = rng.integers(1, 6, n)
    discount = rng.choice([0, 0.05, 0.1, 0.15, 0.2], n, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    revenue = qty * unit_price * (1 - discount)
    # planted business event: West / Laptops collapse in Sep 2026
    sep26 = (d >= "2026-09-01") & (d < "2026-10-01")
    west_laptop = (region == "West") & (category == "Laptops")
    west = region == "West"
    revenue = np.where(sep26 & west & (rng.random(n) < 0.75), revenue * 0.45, revenue)
    revenue = np.where(sep26 & west_laptop & (rng.random(n) < 0.7), revenue * 0.4, revenue)
    customers = np.array([f"CUST-{i:04d}" for i in rng.integers(1, 420, n)])
    df = pd.DataFrame({
        "OrderID": [f"SO-{100000 + i}" for i in range(n)],
        "OrderDate": d.strftime("%Y-%m-%d"),
        "Region": region, "ProductCategory": category, "CustomerSegment": segment,
        "Customer": customers, "Quantity": qty, "UnitPrice": unit_price.round(2),
        "Discount": discount, "Revenue": revenue.round(2),
    })
    # planted VIP customers
    vip = df.sample(40, random_state=1).index
    df.loc[vip, "Customer"] = "CUST-0001"
    df.loc[vip, "Revenue"] = df.loc[vip, "Revenue"] * 6
    # data problems: extreme order, refund, missing, duplicates
    df.loc[df.sample(3, random_state=2).index, "Revenue"] = [185000.0, 240000.0, 99000.0]
    df.loc[df.sample(1, random_state=3).index, "Revenue"] = -1250.0
    df.loc[df.sample(int(n * 0.03), random_state=4).index, "Region"] = np.nan
    df.loc[df.sample(int(n * 0.05), random_state=5).index, "Discount"] = np.nan
    df = pd.concat([df, df.sample(25, random_state=6)], ignore_index=True)
    return df.sort_values("OrderDate").reset_index(drop=True)


def housing(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 1460
    hoods = {"Downtown": 1.35, "Riverside": 1.15, "Oak Park": 1.05, "Lakeview": 1.2, "Maple Grove": 0.95,
             "Old Town": 0.9, "Eastfield": 0.8, "Industrial Edge": 0.65}
    hood = rng.choice(list(hoods), n, p=[0.08, 0.12, 0.18, 0.1, 0.18, 0.14, 0.12, 0.08])
    qual = np.clip(rng.normal(6, 1.4, n).round(), 1, 10).astype(int)
    area = np.clip(rng.normal(1500, 480, n), 500, 4500).round()
    lot = np.clip(rng.lognormal(9.1, 0.4, n), 1500, 60000).round()
    built = rng.integers(1925, 2010, n)
    bedrooms = np.clip((area / 600 + rng.normal(0, 0.7, n)).round(), 1, 6).astype(int)
    garage = rng.choice([0, 1, 2, 3], n, p=[0.06, 0.3, 0.52, 0.12])
    price = (45000 + area * 62 + qual * 11000 + garage * 6500 + (built - 1925) * 180) * np.array([hoods[h] for h in hood])
    price = price * rng.lognormal(0, 0.08, n)
    df = pd.DataFrame({
        "Id": np.arange(1, n + 1), "Neighborhood": hood,
        "BldgType": rng.choice(["Single Family", "Townhouse", "Duplex"], n, p=[0.8, 0.14, 0.06]),
        "OverallQual": qual, "GrLivArea": area, "LotArea": lot, "YearBuilt": built,
        "Bedrooms": bedrooms, "GarageCars": garage,
        "YrSold": rng.choice([2006, 2007, 2008, 2009, 2010], n), "MoSold": rng.integers(1, 13, n),
        "SalePrice": price.round(0),
    })
    df.loc[df.sample(4, random_state=1).index, "SalePrice"] = [755000, 12000, 690000, 18500]
    df.loc[df.sample(int(n * 0.06), random_state=2).index, "GarageCars"] = np.nan
    return df


def hr(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 1200
    dept = rng.choice(["Sales", "Engineering", "HR", "Operations", "Finance"], n, p=[0.3, 0.3, 0.08, 0.22, 0.1])
    role = rng.choice(["Associate", "Senior", "Lead", "Manager"], n, p=[0.4, 0.3, 0.2, 0.1])
    age = np.clip(rng.normal(37, 9, n), 20, 62).round().astype(int)
    tenure = np.clip(rng.gamma(2.2, 2.6, n), 0, 30).round(1)
    level_pay = np.array([{"Associate": 0, "Senior": 18000, "Lead": 36000, "Manager": 60000}[r] for r in role])
    income = 28000 + level_pay + tenure * 900 + rng.normal(0, 6000, n)
    overtime = rng.choice(["Yes", "No"], n, p=[0.28, 0.72])
    sat = rng.integers(1, 5, n)
    p = 0.08 + 0.18 * (overtime == "Yes") + 0.1 * (sat <= 2) + 0.06 * (dept == "Sales") - 0.04 * (tenure > 8)
    attr = np.where(rng.random(n) < np.clip(p, 0.02, 0.9), "Yes", "No")
    return pd.DataFrame({
        "EmployeeID": np.arange(5001, 5001 + n), "Department": dept, "JobLevel": role, "Age": age,
        "MonthlyIncome": income.round(0), "YearsAtCompany": tenure, "OverTime": overtime,
        "SatisfactionScore": sat, "Attrition": attr,
    })


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    sales().to_csv(OUT / "retail_sales.csv", index=False)
    housing().to_csv(OUT / "house_prices.csv", index=False)
    hr().to_csv(OUT / "employee_attrition.csv", index=False)
    print("written:", sorted(p.name for p in OUT.iterdir()))
