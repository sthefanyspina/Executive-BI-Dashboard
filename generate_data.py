
import os
import numpy as np
import pandas as pd

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
TOTAL_ROWS = 1_000_000
OUTPUT_DIR = "synthetic_dataset"

rng = np.random.default_rng(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dimension sizes
N_CUSTOMERS = 100_000
N_PRODUCTS = 5_000

# Order assumptions
START_DATE = pd.Timestamp("2010-01-01")
END_DATE = pd.Timestamp("2025-12-31")

AVG_ITEMS_PER_ORDER = 2

# Return assumptions
RETURN_RATE = 0.10

# ============================================================
# 2. REFERENCE DATA
# ============================================================

STATES = [
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
]

# Commercial regions
STATE_REGION = {
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

geolocation = pd.DataFrame({
    "state": list(STATE_REGION.keys()),
    "region": list(STATE_REGION.values())
})

# Product categories
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

CUSTOMER_SEGMENTS = [
    "New",
    "Regular",
    "Loyal",
    "VIP"
]

ACQUISITION_CHANNELS = [
    "Organic Search",
    "Paid Search",
    "Social Media",
    "Email Marketing",
    "Referral",
    "Direct"
]

SALES_CHANNELS = [
    "Website",
    "Mobile App",
    "Marketplace",
    "Store"
]

ORDER_STATUSES = [
    "Delivered",
    "Shipped",
    "Processing",
    "Canceled"
]

RETURN_REASONS = [
    "Damaged Product",
    "Wrong Product",
    "Customer Dissatisfaction",
    "Late Delivery",
    "Product Defect"
]

# ============================================================
# 3. DIMENSION TABLE: CUSTOMERS
# ============================================================

customer_ids = np.arange(1, N_CUSTOMERS + 1)

customer_signup_dates = (
    pd.Timestamp("2008-01-01")
    + pd.to_timedelta(
        rng.integers(
            0,
            (END_DATE - pd.Timestamp("2008-01-01")).days + 1,
            size=N_CUSTOMERS
        ),
        unit="D"
    )
)

customers = pd.DataFrame({
    "customer_id": customer_ids,

    "customer_unique_id": [
        f"CUST-{i:07d}"
        for i in customer_ids
    ],

    "state": rng.choice(
        STATES,
        size=N_CUSTOMERS
    ),

    "signup_date": customer_signup_dates,

    "customer_segment": rng.choice(
        CUSTOMER_SEGMENTS,
        size=N_CUSTOMERS,
        p=[0.30, 0.40, 0.22, 0.08]
    ),

    "acquisition_channel": rng.choice(
        ACQUISITION_CHANNELS,
        size=N_CUSTOMERS,
        p=[0.25, 0.20, 0.18, 0.12, 0.10, 0.15]
    )
})

# ============================================================
# 4. DIMENSION TABLE: PRODUCTS
# ============================================================

product_ids = np.arange(1, N_PRODUCTS + 1)

product_categories = rng.choice(
    list(categories.keys()),
    size=N_PRODUCTS
)

products = pd.DataFrame({
    "product_id": product_ids,

    "category": product_categories,

    "subcategory": [
        rng.choice(categories[category])
        for category in product_categories
    ]
})

# ============================================================
# 5. DIMENSION TABLE: COSTS
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
# 6. DETERMINE FACT TABLE SIZES
# ============================================================

FIXED_ROWS = (
    len(customers)
    + len(products)
    + len(costs)
    + len(geolocation)
)

# Expected rows per order:
# 1 order + 2.25 items + 1 payment + 0.10 return
EXPECTED_ROWS_PER_ORDER = (
    1 + 2.25 + 1 + RETURN_RATE
)

N_ORDERS = int(
    (TOTAL_ROWS - FIXED_ROWS)
    / EXPECTED_ROWS_PER_ORDER
)

# Recalculate until all tables fit the row budget.
while True:

    item_counts = rng.choice(
        [1, 2, 3, 4],
        size=N_ORDERS,
        p=[0.20, 0.45, 0.25, 0.10]
    )

    N_ITEMS = int(item_counts.sum())

    estimated_returns = int(
        round(N_ORDERS * RETURN_RATE)
    )

    estimated_total = (
        FIXED_ROWS
        + N_ORDERS
        + N_ITEMS
        + N_ORDERS
        + estimated_returns
    )

    if estimated_total <= TOTAL_ROWS:
        break

    reduction = max(
        1,
        int(np.ceil(
            (estimated_total - TOTAL_ROWS) / 5
        ))
    )

    N_ORDERS -= reduction

# ============================================================
# 7. FACT TABLE: ORDERS
# ============================================================

order_ids = np.arange(1, N_ORDERS + 1)

# Assign customers
order_customer_ids = rng.integers(
    1,
    N_CUSTOMERS + 1,
    size=N_ORDERS
)

# Generate order dates
random_order_dates = (
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

# Ensure orders happen after customer registration
signup_map = customers.set_index(
    "customer_id"
)["signup_date"]

assigned_signup_dates = (
    signup_map.reindex(order_customer_ids)
    .reset_index(drop=True)
)

order_dates = pd.DatetimeIndex(
    pd.concat(
        [
            pd.Series(random_order_dates),
            assigned_signup_dates
        ],
        axis=1
    ).max(axis=1)
)

# Order statuses
statuses = rng.choice(
    ORDER_STATUSES,
    size=N_ORDERS,
    p=[0.77, 0.10, 0.08, 0.05]
)

# Promised delivery: 3 to 10 days after order
promised_delivery_dates = (
    order_dates
    + pd.to_timedelta(
        rng.integers(3, 11, size=N_ORDERS),
        unit="D"
    )
)

# Actual delivery dates
delivery_days = rng.integers(
    2,
    15,
    size=N_ORDERS
)

actual_delivery_dates = (
    order_dates
    + pd.to_timedelta(delivery_days, unit="D")
)

# Only delivered and returned orders have delivery dates
delivered_dates = pd.Series(
    pd.NaT,
    index=np.arange(N_ORDERS),
    dtype="datetime64[ns]"
)

delivered_mask = np.isin(
    statuses,
    ["Delivered"]
)

delivered_dates.loc[delivered_mask] = (
    actual_delivery_dates[delivered_mask]
)

# Cancellation dates
canceled_dates = pd.Series(
    pd.NaT,
    index=np.arange(N_ORDERS),
    dtype="datetime64[ns]"
)

canceled_mask = statuses == "Canceled"

canceled_dates.loc[canceled_mask] = (
    order_dates[canceled_mask]
    + pd.to_timedelta(
        rng.integers(
            0, 4,
            size=canceled_mask.sum()
        ),
        unit="D"
    )
)

orders = pd.DataFrame({
    "order_id": order_ids,

    "customer_id": order_customer_ids,

    "order_date": order_dates,

    "status": statuses,

    "promised_delivery_date": promised_delivery_dates,

    "delivered_date": delivered_dates.to_numpy(),

    "canceled_at": canceled_dates.to_numpy(),

    # Operational shipping expense
    "shipping_cost": np.round(
        rng.uniform(
            5,
            50,
            size=N_ORDERS
        ),
        2
    ),

    "sales_channel": rng.choice(
        SALES_CHANNELS,
        size=N_ORDERS,
        p=[0.40, 0.30, 0.20, 0.10]
    )
})

# ============================================================
# 8. FACT TABLE: ORDER ITEMS
# ============================================================

item_order_ids = np.repeat(
    order_ids,
    item_counts
)

actual_items = len(item_order_ids)

quantities = rng.integers(
    1,
    6,
    size=actual_items
)

prices = np.round(
    rng.lognormal(
        mean=3.3,
        sigma=0.7,
        size=actual_items
    ),
    2
)

# Gross merchandise value before discounts
gross_merchandise_value = (
    quantities * prices
)

# Discount rates
discount_rates = rng.choice(
    [0.00, 0.05, 0.10, 0.15, 0.20, 0.30],
    size=actual_items,
    p=[0.45, 0.15, 0.15, 0.12, 0.08, 0.05]
)

discount_values = np.round(
    gross_merchandise_value * discount_rates,
    2
)

# Promotion IDs: NULL means no promotion
promotion_ids = np.where(
    discount_rates > 0,
    rng.choice(
        [
            "PROMO-001",
            "PROMO-002",
            "PROMO-003",
            "PROMO-004",
            "PROMO-005"
        ],
        size=actual_items
    ),
    None
)

# Freight charged to customer per item
freight = np.round(
    rng.uniform(
        2,
        35,
        size=actual_items
    ),
    2
)

order_items = pd.DataFrame({
    "order_id": item_order_ids,

    "product_id": rng.integers(
        1,
        N_PRODUCTS + 1,
        size=actual_items
    ),

    "quantity": quantities,

    "price": prices,

    "freight": freight,

    "discount": discount_values,

    "promotion_id": promotion_ids
})

# ============================================================
# 9. ENRICH ORDER ITEMS
# ============================================================

order_items = order_items.merge(
    costs,
    on="product_id",
    how="left",
    validate="many_to_one"
)

# Gross sales before discount and freight
order_items["gross_merchandise_value"] = (
    order_items["quantity"]
    * order_items["price"]
)

# Discount amount
order_items["discount"] = np.round(
    order_items["discount"],
    2
)

# Freight revenue
order_items["freight_value"] = (
    order_items["freight"]
)

# Net revenue after discount plus freight
order_items["line_revenue"] = np.round(
    order_items["gross_merchandise_value"]
    - order_items["discount"]
    + order_items["freight_value"],
    2
)

# Product cost
order_items["line_cost"] = np.round(
    order_items["quantity"]
    * order_items["unit_cost"],
    2
)

# Final column organization
order_items = order_items[
    [
        "order_id",
        "product_id",
        "quantity",
        "price",
        "freight",
        "unit_cost",
        "gross_merchandise_value",
        "discount",
        "promotion_id",
        "freight_value",
        "line_revenue",
        "line_cost"
    ]
]

# ============================================================
# 10. FACT TABLE: PAYMENTS
# ============================================================

order_totals = (
    order_items
    .groupby("order_id")["line_revenue"]
    .sum()
)

payment_values = (
    order_totals
    .reindex(order_ids)
    .to_numpy()
)

# Canceled orders have no completed payment
payment_values[canceled_mask] = 0

payments = pd.DataFrame({
    "order_id": order_ids,

    "payment_value": np.round(
        payment_values,
        2
    )
})

# ============================================================
# 11. FACT TABLE: RETURNS
# ============================================================

eligible_orders = orders.loc[
    orders["status"] == "Delivered",
    "order_id"
].to_numpy()

N_RETURNS = min(
    int(round(N_ORDERS * RETURN_RATE)),
    len(eligible_orders)
)

returned_order_ids = rng.choice(
    eligible_orders,
    size=N_RETURNS,
    replace=False
)

# Aggregate purchased quantities by order
order_quantities = (
    order_items
    .groupby("order_id")["quantity"]
    .sum()
)

# Maps
payment_map = payments.set_index(
    "order_id"
)["payment_value"]

delivery_date_map = orders.set_index(
    "order_id"
)["delivered_date"]

# Generate returned quantities
returned_quantities = []

for order_id in returned_order_ids:

    purchased_quantity = int(
        order_quantities.loc[order_id]
    )

    returned_quantities.append(
        int(
            rng.integers(
                1,
                purchased_quantity + 1
            )
        )
    )

returned_quantities = np.array(
    returned_quantities
)

# Generate return dates after delivery
return_dates = []

for order_id in returned_order_ids:

    base_date = delivery_date_map.loc[order_id]

    return_dates.append(
        base_date
        + pd.Timedelta(
            days=int(rng.integers(1, 61))
        )
    )

# Calculate refunds proportionally to returned quantity
refund_values = []

for order_id, returned_quantity in zip(
    returned_order_ids,
    returned_quantities
):

    purchased_quantity = int(
        order_quantities.loc[order_id]
    )

    payment_value = float(
        payment_map.loc[order_id]
    )

    refund = (
        payment_value
        * (returned_quantity / purchased_quantity)
        * rng.uniform(0.70, 1.00)
    )

    refund_values.append(
        round(refund, 2)
    )

returns = pd.DataFrame({
    "order_id": returned_order_ids,

    "return_date": return_dates,

    "refund_value": refund_values,

    "return_reason": rng.choice(
        RETURN_REASONS,
        size=N_RETURNS,
        p=[0.25, 0.15, 0.25, 0.15, 0.20]
    ),

    "return_status": rng.choice(
        ["Approved", "Processing"],
        size=N_RETURNS,
        p=[0.85, 0.15]
    ),

    "returned_quantity": returned_quantities
})

# ============================================================
# 12. SYNCHRONIZE RETURNS WITH ORDER STATUS
# ============================================================

returned_mask = orders["order_id"].isin(
    returns["order_id"]
)

orders.loc[
    returned_mask,
    "status"
] = "Returned"

# ============================================================
# 13. DATE FORMATTING
# ============================================================

date_columns = {
    "customers": [
        "signup_date"
    ],
    "orders": [
        "order_date",
        "promised_delivery_date",
        "delivered_date",
        "canceled_at"
    ],
    "returns": [
        "return_date"
    ]
}

for table_name, columns in date_columns.items():

    dataframe = {
        "customers": customers,
        "orders": orders,
        "returns": returns
    }[table_name]

    for column in columns:

        dataframe[column] = pd.to_datetime(
            dataframe[column]
        ).dt.strftime("%Y-%m-%d")

# ============================================================
# 14. EXPORT CSV FILES
# ============================================================

tables = {
    "customers": customers,
    "products": products,
    "costs": costs,
    "orders": orders,
    "order_items": order_items,
    "payments": payments,
    "returns": returns,
    "geolocation": geolocation
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
# 15. SUMMARY
# ============================================================

total_generated = sum(
    len(df)
    for df in tables.values()
)

print("\n" + "=" * 55)
print("DATASET GENERATION COMPLETED")
print("=" * 55)

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
    f"{total_generated / TOTAL_ROWS:.2%}"
)

print(
    f"Output directory: "
    f"{OUTPUT_DIR}"
)

print("\nTable sizes:")

for name, df in tables.items():

    print(
        f"{name:<15}: {len(df):>10,}"
    )

# ============================================================
# 16. VALIDATIONS
# ============================================================

# Primary keys
assert orders["order_id"].is_unique
assert customers["customer_id"].is_unique
assert customers["customer_unique_id"].is_unique
assert products["product_id"].is_unique
assert costs["product_id"].is_unique
assert payments["order_id"].is_unique
assert returns["order_id"].is_unique
assert geolocation["state"].is_unique

# Foreign keys
assert orders["customer_id"].isin(
    customers["customer_id"]
).all()

assert order_items["order_id"].isin(
    orders["order_id"]
).all()

assert order_items["product_id"].isin(
    products["product_id"]
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

# Geographic relationships
assert customers["state"].isin(
    geolocation["state"]
).all()

# Date consistency
order_dates_check = pd.to_datetime(
    orders["order_date"]
)

signup_dates_check = pd.to_datetime(
    orders["customer_id"].map(
        customers.set_index("customer_id")["signup_date"]
    )
)

assert (
    order_dates_check >= signup_dates_check
).all()

assert (
    pd.to_datetime(
        orders["promised_delivery_date"]
    ) >= order_dates_check
).all()

assert (
    orders.loc[
        orders["status"] == "Canceled",
        "canceled_at"
    ].notna()
).all()

assert (
    orders.loc[
        orders["status"] == "Canceled",
        "delivered_date"
    ].isna()
).all()

assert (
    orders.loc[
        orders["status"].isin(["Delivered", "Returned"]),
        "delivered_date"
    ].notna()
).all()

assert (
    pd.to_datetime(
        returns["return_date"]
    ).to_numpy()
    >= pd.to_datetime(
        returns["order_id"].map(
            orders.set_index("order_id")["delivered_date"]
        )
    ).to_numpy()
).all()

# Return consistency
returned_orders = set(
    orders.loc[
        orders["status"] == "Returned",
        "order_id"
    ]
)

assert returned_orders == set(
    returns["order_id"]
)

# Returned quantity cannot exceed purchased quantity
assert (
    returns["returned_quantity"].to_numpy()
    <= returns["order_id"].map(
        order_quantities
    ).to_numpy()
).all()

# Financial consistency
assert (
    order_items["discount"] >= 0
).all()

assert (
    order_items["discount"]
    <= order_items["gross_merchandise_value"]
).all()

assert (
    order_items["line_revenue"] >= 0
).all()

assert (
    order_items["line_cost"] >= 0
).all()

assert (
    order_items["line_revenue"]
    == (
        order_items["gross_merchandise_value"]
        - order_items["discount"]
        + order_items["freight_value"]
    ).round(2)
).all()

payment_lookup = payments.set_index(
    "order_id"
)["payment_value"]

assert (
    returns["refund_value"].to_numpy()
    <= returns["order_id"].map(
        payment_lookup
    ).to_numpy()
).all()

# Promotion consistency
assert (
    order_items.loc[
        order_items["discount"] == 0,
        "promotion_id"
    ].isna()
).all()

assert (
    order_items.loc[
        order_items["discount"] > 0,
        "promotion_id"
    ].notna()
).all()

# Dataset budget
assert total_generated <= TOTAL_ROWS

# Required geographic coverage
assert set(geolocation["state"]) == set(STATES)

print("\nAll validation checks passed successfully!")
