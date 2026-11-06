import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
TOTAL_ROWS = 1_000_000
OUTPUT_DIR = "synthetic_dataset"
rng = np.random.default_rng(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CUSTOMERS = 100_000
N_PRODUCTS = 5_000
AVG_ITEMS_PER_ORDER = 2

START_DATE = pd.Timestamp("2010-01-01")
END_DATE = pd.Timestamp("2025-12-31")

# ============================================================
# GEOGRAPHY DIMENSION
# ============================================================
state_region_map = {
    "California": "West",
    "Texas": "South",
    "Florida": "South",
    "New York": "Northeast",
    "Illinois": "Midwest",
    "Pennsylvania": "Northeast",
    "Ohio": "Midwest",
    "Georgia": "South",
    "North Carolina": "South",
    "Michigan": "Midwest",
    "New Jersey": "Northeast",
    "Virginia": "South",
    "Washington": "West",
    "Arizona": "West",
    "Massachusetts": "Northeast"
}

STATES = list(state_region_map.keys())

geography = pd.DataFrame({
    "state": list(state_region_map.keys()),
    "region": list(state_region_map.values())
})

# ============================================================
# CUSTOMERS
# ============================================================
signup_dates = (
    START_DATE + pd.to_timedelta(
        rng.integers(0, (END_DATE - START_DATE).days + 1, size=N_CUSTOMERS),
        unit="D"
    )
)

customers = pd.DataFrame({
    "customer_id": np.arange(1, N_CUSTOMERS + 1),
    "customer_unique_id": [f"CUST-{i:07d}" for i in range(1, N_CUSTOMERS + 1)],
    "state": rng.choice(STATES, size=N_CUSTOMERS),
    "signup_date": signup_dates,
    "customer_segment": rng.choice([
        "Consumer", "Corporate", "Small Business"
    ], size=N_CUSTOMERS, p=[0.75, 0.15, 0.10]),
    "acquisition_channel": rng.choice([
        "Organic Search", "Paid Search", "Email Marketing",
        "Social Media", "Referral", "Affiliate"
    ], size=N_CUSTOMERS)
})

# ============================================================
# PRODUCTS
# ============================================================
categories = {
    "Electronics": ["Smartphones", "Laptops", "Headphones", "Accessories", "Cameras"],
    "Home": ["Furniture", "Kitchenware", "Decor", "Bedding", "Storage"],
    "Fashion": ["Clothing", "Shoes", "Bags", "Watches", "Accessories"],
    "Sports": ["Fitness", "Outdoor", "Cycling", "Team Sports", "Camping"],
    "Beauty": ["Skincare", "Haircare", "Makeup", "Fragrances", "Personal Care"],
    "Books": ["Fiction", "Business", "Technology", "Education", "Children"]
}

product_ids = np.arange(1, N_PRODUCTS + 1)
product_categories = rng.choice(list(categories.keys()), size=N_PRODUCTS)

products = pd.DataFrame({
    "product_id": product_ids,
    "category": product_categories,
    "subcategory": [rng.choice(categories[c]) for c in product_categories]
})

costs = pd.DataFrame({
    "product_id": product_ids,
    "unit_cost": np.round(rng.lognormal(mean=3.0, sigma=0.8, size=N_PRODUCTS), 2)
})

promotions = pd.DataFrame({
    "promotion_id": ["PROMO10", "PROMO15", "BLACKFRIDAY", "SUMMERSALE"],
    "discount_pct": [0.10, 0.15, 0.25, 0.20],
    "campaign_type": ["Email", "Affiliate", "Seasonal", "Seasonal"]
})

# ============================================================
# SIZE CALCULATION
# ============================================================
fixed_rows = len(customers) + len(products) + len(costs) + len(geography) + len(promotions)
available_rows = TOTAL_ROWS - fixed_rows
N_ORDERS = int(available_rows / (1 + AVG_ITEMS_PER_ORDER + 1 + 0.10))

while True:
    item_counts = rng.choice([1,2,3,4], size=N_ORDERS, p=[0.20,0.45,0.25,0.10])
    N_ITEMS = int(item_counts.sum())
    total = fixed_rows + N_ORDERS + N_ITEMS + N_ORDERS + int(round(N_ORDERS*0.10))
    if total <= TOTAL_ROWS:
        break
    N_ORDERS -= max(1,(total-TOTAL_ROWS)//5)

# ============================================================
# ORDERS
# ============================================================
order_ids = np.arange(1, N_ORDERS + 1)
order_dates = START_DATE + pd.to_timedelta(
    rng.integers(0, (END_DATE - START_DATE).days + 1, size=N_ORDERS),
    unit="D"
)

statuses = rng.choice(
    ["Delivered", "Shipped", "Processing", "Canceled"],
    size=N_ORDERS,
    p=[0.77, 0.10, 0.08, 0.05]
)

sales_channel = rng.choice(
    ["Website", "Mobile App", "Marketplace", "Physical Store"],
    size=N_ORDERS,
    p=[0.45,0.30,0.20,0.05]
)

promised_delivery_date=[]
delivered_date=[]
canceled_at=[]

for st, dt in zip(statuses, order_dates):
    promised = dt + pd.Timedelta(days=int(rng.integers(3,15)))
    promised_delivery_date.append(promised)

    if st == "Delivered":
        delivered_date.append(dt + pd.Timedelta(days=int(rng.integers(2,18))))
        canceled_at.append(pd.NaT)
    elif st == "Canceled":
        delivered_date.append(pd.NaT)
        canceled_at.append(dt + pd.Timedelta(days=int(rng.integers(0,3))))
    else:
        delivered_date.append(pd.NaT)
        canceled_at.append(pd.NaT)

orders = pd.DataFrame({
    "order_id": order_ids,
    "customer_id": rng.integers(1, N_CUSTOMERS+1, size=N_ORDERS),
    "order_date": order_dates,
    "status": statuses,
    "promised_delivery_date": promised_delivery_date,
    "delivered_date": delivered_date,
    "canceled_at": canceled_at,
    "sales_channel": sales_channel
})

# ============================================================
# ORDER ITEMS
# ============================================================
item_order_ids = np.repeat(order_ids, item_counts)
actual_items = len(item_order_ids)

order_items = pd.DataFrame({
    "order_id": item_order_ids,
    "product_id": rng.integers(1, N_PRODUCTS+1, size=actual_items),
    "quantity": rng.integers(1,6,size=actual_items),
    "price": np.round(rng.lognormal(mean=3.3,sigma=0.7,size=actual_items),2),
    "freight": np.round(rng.uniform(2,35,size=actual_items),2)
})

promo_options=[None,"PROMO10","PROMO15","BLACKFRIDAY","SUMMERSALE"]
promotion_id = rng.choice(promo_options,size=actual_items,p=[0.70,0.10,0.08,0.07,0.05])

promo_pct = {
    None:0.0,
    "PROMO10":0.10,
    "PROMO15":0.15,
    "BLACKFRIDAY":0.25,
    "SUMMERSALE":0.20
}

order_items["promotion_id"] = promotion_id
order_items["unit_cost"] = order_items["product_id"].map(costs.set_index("product_id")["unit_cost"])
order_items["gross_merchandise_value"] = order_items["quantity"] * order_items["price"]
order_items["discount"] = np.round(
    order_items["gross_merchandise_value"] * np.vectorize(promo_pct.get)(promotion_id),2
)
order_items["freight_value"] = order_items["freight"]
order_items["line_revenue"] = order_items["gross_merchandise_value"] - order_items["discount"] + order_items["freight_value"]
order_items["line_cost"] = order_items["quantity"] * order_items["unit_cost"]

shipping_cost = order_items.groupby("order_id")["freight"].sum().reset_index(name="shipping_cost")
orders = orders.merge(shipping_cost, on="order_id", how="left")

# ============================================================
# PAYMENTS
# ============================================================
order_totals = order_items.groupby("order_id")["line_revenue"].sum()
payments = pd.DataFrame({
    "order_id": order_ids,
    "payment_value": np.round(order_totals.reindex(order_ids).to_numpy(),2)
})

# ============================================================
# RETURNS
# ============================================================
eligible = orders.loc[orders.status == "Delivered", "order_id"].to_numpy()
N_RETURNS = min(int(round(N_ORDERS * 0.10)), len(eligible))
returned = rng.choice(eligible, size=N_RETURNS, replace=False)

payment_map = payments.set_index("order_id")["payment_value"]
date_map = orders.set_index("order_id")["order_date"]

returns = pd.DataFrame({
    "order_id": returned,
    "return_date": [date_map.loc[o] + pd.Timedelta(days=int(rng.integers(1,61))) for o in returned],
    "refund_value": np.round([payment_map.loc[o] * rng.uniform(0.25,1.0) for o in returned],2),
    "return_reason": rng.choice([
        "Damaged Product","Wrong Product","Customer Dissatisfaction",
        "Late Delivery","Product Defect"
    ], size=N_RETURNS, p=[0.30,0.15,0.20,0.15,0.20]),
    "return_status": rng.choice([
        "Approved","Pending","Rejected"
    ], size=N_RETURNS, p=[0.85,0.10,0.05]),
    "returned_quantity": rng.integers(1,5,size=N_RETURNS)
})

orders.loc[orders.order_id.isin(returned), "status"] = "Returned"

# ============================================================
# FORMAT DATES
# ============================================================
for col in ["order_date","promised_delivery_date","delivered_date","canceled_at"]:
    orders[col] = pd.to_datetime(orders[col]).dt.strftime("%Y-%m-%d")

customers["signup_date"] = pd.to_datetime(customers["signup_date"]).dt.strftime("%Y-%m-%d")
returns["return_date"] = pd.to_datetime(returns["return_date"]).dt.strftime("%Y-%m-%d")

# ============================================================
# EXPORT
# ============================================================
tables = {
    "customers": customers,
    "products": products,
    "costs": costs,
    "geography": geography,
    "promotions": promotions,
    "orders": orders,
    "order_items": order_items,
    "payments": payments,
    "returns": returns
}

for name, df in tables.items():
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"{name}.csv: {len(df):,} rows")

print("Dataset generation completed.")
print("Total rows:", sum(len(df) for df in tables.values()))
print("All files exported successfully.")
