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

# Dimension sizes
N_CUSTOMERS = 100_000
N_PRODUCTS = 5_000

# Fact assumptions
AVG_ITEMS_PER_ORDER = 2

# Period
START_DATE = pd.Timestamp("2010-01-01")
END_DATE = pd.Timestamp("2025-12-31")

# ============================================================
# DIMENSION TABLE: CUSTOMERS
# ============================================================

customers = pd.DataFrame({
    "customer_id": np.arange(1, N_CUSTOMERS + 1),
    "customer_unique_id": [
        f"CUST-{i:07d}"
        for i in range(1, N_CUSTOMERS + 1)
    ],
    "state": rng.choice(
        [
            "California",
            "Texas",
            "Florida",
            "New York",
            "Illinois",
            "Pennsylvania",
            "Ohio",
            "Georgia",
            "North Carolina",
            "Michigan",
            "New Jersey",
            "Virginia",
            "Washington",
            "Arizona",
            "Massachusetts"
        ],
        size=N_CUSTOMERS
    )
})

# ============================================================
# DIMENSION TABLE: PRODUCTS
# ============================================================

categories = {
    "Electronics": [
        "Smartphones",
        "Laptops",
        "Headphones",
        "Accessories",
        "Cameras"
    ],
    "Home": [
        "Furniture",
        "Kitchenware",
        "Decor",
        "Bedding",
        "Storage"
    ],
    "Fashion": [
        "Clothing",
        "Shoes",
        "Bags",
        "Watches",
        "Accessories"
    ],
    "Sports": [
        "Fitness",
        "Outdoor",
        "Cycling",
        "Team Sports",
        "Camping"
    ],
    "Beauty": [
        "Skincare",
        "Haircare",
        "Makeup",
        "Fragrances",
        "Personal Care"
    ],
    "Books": [
        "Fiction",
        "Business",
        "Technology",
        "Education",
        "Children"
    ]
}

product_ids = np.arange(1, N_PRODUCTS + 1)

product_categories = rng.choice(
    list(categories.keys()),
    size=N_PRODUCTS
)

products = pd.DataFrame({
    "product_id": product_ids,
    "category": product_categories,
    "subcategory": [
        rng.choice(categories[c])
        for c in product_categories
    ]
})

# ============================================================
# DIMENSION TABLE: COSTS
# ============================================================

costs = pd.DataFrame({
    "product_id": product_ids,
    "unit_cost": np.round(
        rng.lognormal(
            mean=3.0,
            sigma=0.8,
            size=N_PRODUCTS
        ),
        2
    )
})

# ============================================================
# DETERMINE FACT TABLE SIZES
# ============================================================

fixed_rows = (
    N_CUSTOMERS +
    len(products) +
    len(costs)
)

available_rows = TOTAL_ROWS - fixed_rows

N_ORDERS = int(
    available_rows /
    (
        1 +          # orders
        AVG_ITEMS_PER_ORDER +
        1 +          # payments
        0.10         # returns
    )
)

while True:

    item_counts = rng.choice(
        [1, 2, 3, 4],
        size=N_ORDERS,
        p=[0.20, 0.45, 0.25, 0.10]
    )

    N_ITEMS = int(item_counts.sum())

    estimated_returns = int(
        round(N_ORDERS * 0.10)
    )

    total = (
        len(customers) +
        len(products) +
        len(costs) +
        N_ORDERS +
        N_ITEMS +
        N_ORDERS +
        estimated_returns
    )

    if total <= TOTAL_ROWS:
        break

    reduction = max(
        1,
        (total - TOTAL_ROWS) // 5
    )

    N_ORDERS -= reduction

# ============================================================
# FACT TABLE: ORDERS
# ============================================================

order_ids = np.arange(
    1,
    N_ORDERS + 1
)

order_dates = (
    START_DATE
    + pd.to_timedelta(
        rng.integers(
            0,
            (END_DATE - START_DATE).days + 1,
            size=N_ORDERS
        ),
        unit="D"
    )
)

statuses = rng.choice(
    [
        "Delivered",
        "Shipped",
        "Processing",
        "Canceled"
    ],
    size=N_ORDERS,
    p=[
        0.77,
        0.10,
        0.08,
        0.05
    ]
)

orders = pd.DataFrame({
    "order_id": order_ids,
    "customer_id": rng.integers(
        1,
        N_CUSTOMERS + 1,
        size=N_ORDERS
    ),
    "order_date": order_dates,
    "status": statuses
})

# ============================================================
# FACT TABLE: ORDER ITEMS
# ============================================================

item_order_ids = np.repeat(
    order_ids,
    item_counts
)

actual_items = len(item_order_ids)

order_items = pd.DataFrame({
    "order_id": item_order_ids,
    "product_id": rng.integers(
        1,
        N_PRODUCTS + 1,
        size=actual_items
    ),
    "quantity": rng.integers(
        1,
        6,
        size=actual_items
    ),
    "price": np.round(
        rng.lognormal(
            mean=3.3,
            sigma=0.7,
            size=actual_items
        ),
        2
    ),
    "freight": np.round(
        rng.uniform(
            2,
            35,
            size=actual_items
        ),
        2
    )
})

# ============================================================
# ENRICH ORDER ITEMS
# ============================================================

order_items = order_items.merge(
    costs,
    on="product_id",
    how="left"
)

order_items["gross_merchandise_value"] = (
    order_items["quantity"] *
    order_items["price"]
)

order_items["freight_value"] = (
    order_items["freight"]
)

order_items["line_revenue"] = (
    order_items["gross_merchandise_value"] +
    order_items["freight_value"]
)

order_items["line_cost"] = (
    order_items["quantity"] *
    order_items["unit_cost"]
)

# ============================================================
# FACT TABLE: PAYMENTS
# ============================================================

order_totals = (
    order_items
    .groupby("order_id")["line_revenue"]
    .sum()
)

payments = pd.DataFrame({
    "order_id": order_ids,
    "payment_value": np.round(
        order_totals.reindex(order_ids).to_numpy(),
        2
    )
})

# ============================================================
# FACT TABLE: RETURNS
# ============================================================

eligible_orders = orders.loc[
    orders["status"] == "Delivered",
    "order_id"
].to_numpy()

N_RETURNS = min(
    int(round(N_ORDERS * 0.10)),
    len(eligible_orders)
)

returned_order_ids = rng.choice(
    eligible_orders,
    size=N_RETURNS,
    replace=False
)

payment_map = (
    payments
    .set_index("order_id")["payment_value"]
)

order_date_map = (
    orders
    .set_index("order_id")["order_date"]
)

return_dates = []

for order_id in returned_order_ids:

    base_date = order_date_map.loc[order_id]

    return_dates.append(
        base_date +
        pd.Timedelta(
            days=int(
                rng.integers(1, 61)
            )
        )
    )

returns = pd.DataFrame({
    "order_id": returned_order_ids,
    "return_date": return_dates,
    "refund_value": np.round(
        [
            payment_map.loc[o]
            * rng.uniform(0.25, 1.0)
            for o in returned_order_ids
        ],
        2
    )
})

# ============================================================
# SYNCHRONIZE RETURNS WITH ORDER STATUS
# ============================================================

orders.loc[
    orders["order_id"].isin(
        returns["order_id"]
    ),
    "status"
] = "Returned"

# ============================================================
# DATE FORMATTING
# ============================================================

orders["order_date"] = (
    pd.to_datetime(
        orders["order_date"]
    ).dt.strftime("%Y-%m-%d")
)

returns["return_date"] = (
    pd.to_datetime(
        returns["return_date"]
    ).dt.strftime("%Y-%m-%d")
)

# ============================================================
# EXPORT CSV FILES
# ============================================================

tables = {
    "customers": customers,
    "products": products,
    "costs": costs,
    "orders": orders,
    "order_items": order_items,
    "payments": payments,
    "returns": returns
}

for table_name, dataframe in tables.items():

    file_path = os.path.join(
        OUTPUT_DIR,
        f"{table_name}.csv"
    )

    dataframe.to_csv(
        file_path,
        index=False
    )

    print(
        f"{table_name}.csv: "
        f"{len(dataframe):,} rows"
    )

# ============================================================
# SUMMARY
# ============================================================

total_generated = sum(
    len(df)
    for df in tables.values()
)

print("\nDataset generation completed.")
print(
    f"Total rows generated: "
    f"{total_generated:,}"
)

print(
    f"Maximum row budget: "
    f"{TOTAL_ROWS:,}"
)

print(
    f"Utilization: "
    f"{(total_generated / TOTAL_ROWS):.2%}"
)

print(
    f"Output directory: "
    f"{OUTPUT_DIR}"
)

# ============================================================
# VALIDATIONS
# ============================================================

assert orders["order_id"].is_unique
assert customers["customer_id"].is_unique
assert products["product_id"].is_unique
assert costs["product_id"].is_unique
assert payments["order_id"].is_unique

assert order_items["order_id"].isin(
    orders["order_id"]
).all()

assert order_items["product_id"].isin(
    products["product_id"]
).all()

assert orders["customer_id"].isin(
    customers["customer_id"]
).all()

assert payments["order_id"].isin(
    orders["order_id"]
).all()

assert returns["order_id"].isin(
    orders["order_id"]
).all()

assert costs["product_id"].isin(
    products["product_id"]
).all()

returned_orders = set(
    orders.loc[
        orders["status"] == "Returned",
        "order_id"
    ]
)

assert returned_orders == set(
    returns["order_id"]
)

assert (
    order_items["line_revenue"] >= 0
).all()

assert (
    order_items["line_cost"] >= 0
).all()

payment_lookup = (
    payments
    .set_index("order_id")["payment_value"]
)

assert (
    returns["refund_value"]
    <= returns["order_id"].map(
        payment_lookup
    )
).all()

print(
    "All relationship checks passed."
)
