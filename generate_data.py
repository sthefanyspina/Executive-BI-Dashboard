
import os
import numpy as np
import pandas as pd

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
COMPANY = "Vision Group"

OUTPUT_DIR = "vision_group_dataset"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")

os.makedirs(RAW_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# Business targets
N_CUSTOMERS = 82_400
N_ORDERS = 145_200
N_PRODUCTS = 500
N_SELLERS = 50
N_STORES = 20

TARGET_REVENUE = 8_200_000.00
TARGET_GROSS_PROFIT = 2_100_000.00

TARGET_COST = (
    TARGET_REVENUE - TARGET_GROSS_PROFIT
)

TARGET_MARGIN = (
    TARGET_GROSS_PROFIT / TARGET_REVENUE
)

# Historical period
START_DATE = pd.Timestamp("2021-01-01")
END_DATE = pd.Timestamp("2025-12-31")

# Business assumptions
RETURN_RATE = 0.10
CANCELLATION_RATE = 0.05

# ============================================================
# 2. BRAZILIAN GEOGRAPHICAL REFERENCE DATA
# ============================================================

CITY_STATE_DATA = [
    ("Sao Paulo", "SP", "Southeast"),
    ("Campinas", "SP", "Southeast"),
    ("Santos", "SP", "Southeast"),
    ("Rio de Janeiro", "RJ", "Southeast"),
    ("Niteroi", "RJ", "Southeast"),
    ("Belo Horizonte", "MG", "Southeast"),
    ("Uberlandia", "MG", "Southeast"),
    ("Vitoria", "ES", "Southeast"),
    ("Curitiba", "PR", "South"),
    ("Londrina", "PR", "South"),
    ("Porto Alegre", "RS", "South"),
    ("Caxias do Sul", "RS", "South"),
    ("Florianopolis", "SC", "South"),
    ("Joinville", "SC", "South"),
    ("Salvador", "BA", "Northeast"),
    ("Feira de Santana", "BA", "Northeast"),
    ("Recife", "PE", "Northeast"),
    ("Caruaru", "PE", "Northeast"),
    ("Fortaleza", "CE", "Northeast"),
    ("Juazeiro do Norte", "CE", "Northeast"),
    ("Natal", "RN", "Northeast"),
    ("Joao Pessoa", "PB", "Northeast"),
    ("Maceio", "AL", "Northeast"),
    ("Aracaju", "SE", "Northeast"),
    ("Teresina", "PI", "Northeast"),
    ("Sao Luis", "MA", "Northeast"),
    ("Belem", "PA", "North"),
    ("Manaus", "AM", "North"),
    ("Porto Velho", "RO", "North"),
    ("Rio Branco", "AC", "North"),
    ("Macapa", "AP", "North"),
    ("Boa Vista", "RR", "North"),
    ("Palmas", "TO", "North"),
    ("Brasilia", "DF", "Central-West"),
    ("Goiania", "GO", "Central-West"),
    ("Anapolis", "GO", "Central-West"),
    ("Campo Grande", "MS", "Central-West"),
    ("Cuiaba", "MT", "Central-West"),
    ("Dourados", "MS", "Central-West")
]

cities = pd.DataFrame(
    CITY_STATE_DATA,
    columns=["city", "state", "region"]
)

regions = (
    cities[["region"]]
    .drop_duplicates()
    .sort_values("region")
    .reset_index(drop=True)
)

regions["region_id"] = np.arange(
    1,
    len(regions) + 1
)

cities = cities.merge(
    regions,
    on="region",
    how="left"
)

# ============================================================
# 3. PRODUCT CATEGORIES
# ============================================================

CATEGORY_DATA = {
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

categories = pd.DataFrame(
    [
        {
            "category_id": category_id,
            "category_name": category,
            "subcategory": subcategory
        }
        for category_id, (category, subcategories)
        in enumerate(CATEGORY_DATA.items(), start=1)
        for subcategory in subcategories
    ]
)

category_dimension = (
    categories[
        ["category_id", "category_name"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

# ============================================================
# 4. CUSTOMER DIMENSION
# ============================================================

customer_ids = np.arange(
    1,
    N_CUSTOMERS + 1
)

customer_cities = cities.iloc[
    rng.integers(0, len(cities), N_CUSTOMERS)
].reset_index(drop=True)

customer_signup_dates = (
    pd.Timestamp("2019-01-01")
    + pd.to_timedelta(
        rng.integers(
            0,
            (END_DATE - pd.Timestamp("2019-01-01")).days + 1,
            N_CUSTOMERS
        ),
        unit="D"
    )
)

customers = pd.DataFrame({

    "customer_id": customer_ids,

    "customer_code": [
        f"CUST-{i:07d}"
        for i in customer_ids
    ],

    "city": customer_cities["city"],

    "state": customer_cities["state"],

    "region_id": customer_cities["region_id"],

    "signup_date": customer_signup_dates,

    "customer_segment": rng.choice(
        ["New", "Regular", "Loyal", "VIP"],
        N_CUSTOMERS,
        p=[0.30, 0.40, 0.22, 0.08]
    ),

    "acquisition_channel": rng.choice(
        [
            "Organic Search",
            "Paid Search",
            "Social Media",
            "Email Marketing",
            "Referral",
            "Direct"
        ],
        N_CUSTOMERS,
        p=[0.25, 0.20, 0.18, 0.12, 0.10, 0.15]
    )

})

# ============================================================
# 5. PRODUCT DIMENSION
# ============================================================

product_ids = np.arange(
    1,
    N_PRODUCTS + 1
)

selected_categories = rng.integers(
    0,
    len(category_dimension),
    N_PRODUCTS
)

product_category_ids = (
    category_dimension["category_id"]
    .to_numpy()[selected_categories]
)

subcategory_map = categories.groupby(
    "category_id"
)["subcategory"].apply(list).to_dict()

product_subcategories = [
    rng.choice(
        subcategory_map[category_id]
    )
    for category_id in product_category_ids
]

products = pd.DataFrame({

    "product_id": product_ids,

    "product_code": [
        f"PROD-{i:05d}"
        for i in product_ids
    ],

    "product_name": [
        f"Product {i:05d}"
        for i in product_ids
    ],

    "category_id": product_category_ids,

    "subcategory": product_subcategories,

    "unit_cost": np.round(
        rng.uniform(15, 250, N_PRODUCTS),
        2
    ),

    "list_price": np.round(
        rng.uniform(40, 500, N_PRODUCTS),
        2
    )

})

# ============================================================
# 6. SELLER DIMENSION
# ============================================================

seller_ids = np.arange(
    1,
    N_SELLERS + 1
)

sellers = pd.DataFrame({

    "seller_id": seller_ids,

    "seller_name": [
        f"Seller {i:03d}"
        for i in seller_ids
    ],

    "seller_type": rng.choice(
        ["Internal", "Marketplace"],
        N_SELLERS,
        p=[0.75, 0.25]
    ),

    "commission_rate": np.round(
        rng.uniform(0.02, 0.12, N_SELLERS),
        4
    )

})

# ============================================================
# 7. STORE DIMENSION
# ============================================================

store_ids = np.arange(
    1,
    N_STORES + 1
)

store_city_positions = rng.choice(
    len(cities),
    N_STORES,
    replace=False
)

store_cities = cities.iloc[
    store_city_positions
].reset_index(drop=True)

stores = pd.DataFrame({

    "store_id": store_ids,

    "store_name": [
        f"Vision Store {i:03d}"
        for i in store_ids
    ],

    "city": store_cities["city"],

    "state": store_cities["state"],

    "region_id": store_cities["region_id"],

    "store_type": rng.choice(
        ["Flagship", "Standard", "Outlet"],
        N_STORES,
        p=[0.15, 0.70, 0.15]
    )

})

# ============================================================
# 8. DATE DIMENSION
# ============================================================

calendar_dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

date_dimension = pd.DataFrame({

    "date": calendar_dates,

    "year": calendar_dates.year,

    "quarter": calendar_dates.quarter,

    "month": calendar_dates.month,

    "month_name": calendar_dates.strftime("%B"),

    "year_month": calendar_dates.strftime("%Y-%m"),

    "day": calendar_dates.day,

    "day_name": calendar_dates.strftime("%A"),

    "week": calendar_dates.isocalendar().week.astype(int),

    "is_weekend": calendar_dates.dayofweek >= 5

})

date_dimension["date_id"] = (
    date_dimension["date"].dt.strftime("%Y%m%d").astype(int)
)

# ============================================================
# 9. BUSINESS RULES DOCUMENTATION
# ============================================================

BUSINESS_RULES = {

    "Revenue":
        "Target revenue is calibrated using non-canceled sales items.",

    "Gross Profit":
        "Target profit represents gross profit before product returns.",

    "Canceled Orders":
        "Canceled orders remain in the dataset but their item revenue and cost are zero.",

    "Commercial KPIs":
        "Canceled orders must be excluded from commercial performance indicators.",

    "Returns":
        "Refunds are recorded at product-item level in fact_returns_items.",

    "Net Revenue":
        "Net revenue after returns equals commercial revenue minus approved refunds.",

    "Product Cost":
        "line_cost represents the realized commercial product cost.",

    "Customers":
        "Registered customers are not necessarily purchasing customers.",

    "Marketing":
        "Attributed marketing revenue is synthetic and is not verified attribution.",

    "Targets":
        "Monthly targets are distributed uniformly across the historical period.",

    "Inventory":
        "Inventory records are synthetic monthly snapshots, not actual stock movements.",

    "Geography":
        "Cities are associated with their correct Brazilian states and commercial regions.",

    "Financial":
        "Financial records are synthetic management estimates and are not audited accounting statements."

}

# ============================================================
# 10. DIMENSION TABLE COLLECTION
# ============================================================

dimensions = {

    "dim_customers": customers,

    "dim_products": products,

    "dim_categories": category_dimension,

    "dim_sellers": sellers,

    "dim_stores": stores,

    "dim_regions": regions,

    "dim_date": date_dimension

}

# ============================================================
# 11. MONETARY ALLOCATION FUNCTION
# ============================================================

def allocate_money(total_value, weights):

    """
    Distributes a monetary amount in cents.
    The sum of allocated values always equals
    the original total amount.
    """

    total_cents = int(
        round(total_value * 100)
    )

    weights = np.asarray(
        weights,
        dtype=float
    )

    weights = np.maximum(
        weights,
        0
    )

    if weights.sum() == 0:
        weights = np.ones(len(weights))

    exact_values = (
        weights / weights.sum()
    ) * total_cents

    allocated = np.floor(
        exact_values
    ).astype(np.int64)

    remaining = (
        total_cents - allocated.sum()
    )

    fractions = (
        exact_values - allocated
    )

    if remaining > 0:

        positions = np.argsort(
            -fractions
        )

        allocated[
            positions[:remaining]
        ] += 1

    return allocated / 100


# ============================================================
# 12. FACT_SALES - ORDER GENERATION
# ============================================================

order_ids = np.arange(
    1,
    N_ORDERS + 1
)

order_customer_ids = rng.integers(
    1,
    N_CUSTOMERS + 1,
    N_ORDERS
)

signup_map = customers.set_index(
    "customer_id"
)["signup_date"]

assigned_signup_dates = (
    pd.Series(order_customer_ids)
    .map(signup_map)
    .reset_index(drop=True)
)

random_order_dates = (
    START_DATE
    + pd.to_timedelta(
        rng.integers(
            0,
            (END_DATE - START_DATE).days + 1,
            N_ORDERS
        ),
        unit="D"
    )
)

order_dates = pd.concat(
    [
        pd.Series(random_order_dates),
        assigned_signup_dates
    ],
    axis=1
).max(axis=1)

order_dates = pd.to_datetime(
    order_dates
)

# Order statuses
order_statuses = rng.choice(
    [
        "Delivered",
        "Shipped",
        "Processing",
        "Canceled"
    ],
    N_ORDERS,
    p=[
        0.82,
        0.08,
        0.05,
        0.05
    ]
)

canceled_mask = (
    order_statuses == "Canceled"
)

# Delivery dates
delivery_days = rng.integers(
    2,
    15,
    N_ORDERS
)

actual_delivery_dates = (
    order_dates
    + pd.to_timedelta(
        delivery_days,
        unit="D"
    )
)

delivered_dates = pd.Series(
    actual_delivery_dates
)

delivered_mask = (
    order_statuses == "Delivered"
)

delivered_dates.loc[
    ~delivered_mask
] = pd.NaT

# Promised delivery dates
promised_delivery_dates = (
    order_dates
    + pd.to_timedelta(
        rng.integers(
            3,
            11,
            N_ORDERS
        ),
        unit="D"
    )
)

# Cancellation dates
canceled_dates = pd.Series(
    pd.NaT,
    index=np.arange(N_ORDERS),
    dtype="datetime64[ns]"
)

canceled_dates.loc[
    canceled_mask
] = (
    order_dates[canceled_mask]
    + pd.to_timedelta(
        rng.integers(
            0,
            4,
            canceled_mask.sum()
        ),
        unit="D"
    )
)

# Store and geographical assignment
order_store_ids = rng.integers(
    1,
    N_STORES + 1,
    N_ORDERS
)

store_geo = stores.set_index(
    "store_id"
)[["state", "region_id"]]

sales = pd.DataFrame({

    "order_id": order_ids,

    "customer_id": order_customer_ids,

    "seller_id": rng.integers(
        1,
        N_SELLERS + 1,
        N_ORDERS
    ),

    "store_id": order_store_ids,

    "order_date": order_dates,

    "promised_delivery_date": promised_delivery_dates,

    "delivered_date": delivered_dates,

    "canceled_at": canceled_dates,

    "status": order_statuses,

    "sales_channel": rng.choice(
        [
            "Website",
            "Mobile App",
            "Marketplace",
            "Physical Store"
        ],
        N_ORDERS,
        p=[
            0.35,
            0.25,
            0.20,
            0.20
        ]
    ),

    "shipping_cost": rng.uniform(
        8,
        45,
        N_ORDERS
    ).round(2),

    "is_canceled": canceled_mask,

    "is_commercial": ~canceled_mask

})

sales["state"] = (
    sales["store_id"]
    .map(store_geo["state"])
)

sales["region_id"] = (
    sales["store_id"]
    .map(store_geo["region_id"])
)

# ============================================================
# 13. FACT_SALES_ITEMS - ITEM GENERATION
# ============================================================

item_counts = rng.choice(
    [1, 2, 3, 4],
    N_ORDERS,
    p=[0.20, 0.45, 0.25, 0.10]
)

item_order_ids = np.repeat(
    order_ids,
    item_counts
)

N_ITEMS = len(item_order_ids)

item_ids = np.arange(
    1,
    N_ITEMS + 1
)

quantities = rng.integers(
    1,
    5,
    N_ITEMS
)

item_product_ids = rng.integers(
    1,
    N_PRODUCTS + 1,
    N_ITEMS
)

# Product reference costs
product_cost_map = products.set_index(
    "product_id"
)["unit_cost"]

reference_unit_cost = (
    pd.Series(item_product_ids)
    .map(product_cost_map)
    .to_numpy()
)

initial_item_cost = (
    quantities * reference_unit_cost
)

# Initial item prices
initial_unit_prices = rng.uniform(
    30,
    350,
    N_ITEMS
)

initial_gross_revenue = (
    quantities * initial_unit_prices
)

# Discount rates
discount_rates = rng.choice(
    [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25
    ],
    N_ITEMS,
    p=[
        0.45,
        0.18,
        0.15,
        0.12,
        0.07,
        0.03
    ]
)

initial_net_revenue = (
    initial_gross_revenue
    * (1 - discount_rates)
)

# Identify canceled items
order_status_map = sales.set_index(
    "order_id"
)["status"]

item_status = (
    pd.Series(item_order_ids)
    .map(order_status_map)
    .to_numpy()
)

item_canceled_mask = (
    item_status == "Canceled"
)

item_eligible_mask = (
    ~item_canceled_mask
)

eligible_positions = np.flatnonzero(
    item_eligible_mask
)

# ============================================================
# 14. CALIBRATE COMMERCIAL REVENUE
# ============================================================

net_revenue = np.zeros(
    N_ITEMS,
    dtype=float
)

net_revenue[
    eligible_positions
] = allocate_money(
    TARGET_REVENUE,
    initial_net_revenue[
        eligible_positions
    ]
)

# Reconstruct gross revenue and discounts
gross_revenue = np.zeros(
    N_ITEMS,
    dtype=float
)

discount_value = np.zeros(
    N_ITEMS,
    dtype=float
)

for position in eligible_positions:

    net_cents = int(
        round(
            net_revenue[position] * 100
        )
    )

    discount_rate = (
        discount_rates[position]
    )

    gross_cents = int(
        np.ceil(
            net_cents / (1 - discount_rate)
        )
    )

    gross_revenue[position] = (
        gross_cents / 100
    )

    discount_value[position] = (
        gross_cents - net_cents
    ) / 100

# ============================================================
# 15. CALIBRATE COMMERCIAL PRODUCT COST
# ============================================================

line_cost = np.zeros(
    N_ITEMS,
    dtype=float
)

line_cost[
    eligible_positions
] = allocate_money(
    TARGET_COST,
    initial_item_cost[
        eligible_positions
    ]
)

# Calculate realized unit cost
adjusted_unit_cost = np.divide(
    line_cost,
    quantities,
    out=np.zeros(N_ITEMS),
    where=quantities != 0
)

# Calculate effective unit price
unit_price = np.divide(
    gross_revenue,
    quantities,
    out=np.zeros(N_ITEMS),
    where=quantities != 0
)

# ============================================================
# 16. ZERO CANCELED ITEM VALUES
# ============================================================

gross_revenue[
    item_canceled_mask
] = 0.00

discount_value[
    item_canceled_mask
] = 0.00

net_revenue[
    item_canceled_mask
] = 0.00

line_cost[
    item_canceled_mask
] = 0.00

adjusted_unit_cost[
    item_canceled_mask
] = 0.00

unit_price[
    item_canceled_mask
] = 0.00

# Gross profit before returns
gross_profit = (
    net_revenue - line_cost
).round(2)

# Gross margin
gross_margin = np.divide(
    gross_profit,
    net_revenue,
    out=np.zeros(N_ITEMS),
    where=net_revenue != 0
)

# ============================================================
# 17. PROMOTION ASSIGNMENT
# ============================================================

promotion_ids = np.full(
    N_ITEMS,
    None,
    dtype=object
)

discounted_positions = np.flatnonzero(
    (discount_rates > 0)
    & item_eligible_mask
)

promotion_ids[
    discounted_positions
] = rng.choice(
    [
        "PROMO-001",
        "PROMO-002",
        "PROMO-003",
        "PROMO-004",
        "PROMO-005"
    ],
    len(discounted_positions)
)

# ============================================================
# 18. CREATE FACT_SALES_ITEMS
# ============================================================

items = pd.DataFrame({

    "item_id": item_ids,

    "order_id": item_order_ids,

    "product_id": item_product_ids,

    "quantity": quantities,

    "unit_price": np.round(
        unit_price,
        4
    ),

    "unit_cost": np.round(
        adjusted_unit_cost,
        4
    ),

    "discount_rate": discount_rates,

    "gross_revenue": np.round(
        gross_revenue,
        2
    ),

    "discount_value": np.round(
        discount_value,
        2
    ),

    "net_revenue": np.round(
        net_revenue,
        2
    ),

    "line_cost": np.round(
        line_cost,
        2
    ),

    "gross_profit": np.round(
        gross_profit,
        2
    ),

    "gross_margin": gross_margin,

    "promotion_id": promotion_ids,

    "is_canceled": item_canceled_mask,

    "is_commercial": item_eligible_mask,

    "status": item_status

})

# ============================================================
# 19. AGGREGATE ITEMS INTO FACT_SALES
# ============================================================

order_totals = (
    items.groupby("order_id")
    .agg(

        order_gross_revenue=(
            "gross_revenue",
            "sum"
        ),

        order_discount=(
            "discount_value",
            "sum"
        ),

        order_revenue=(
            "net_revenue",
            "sum"
        ),

        order_cost=(
            "line_cost",
            "sum"
        ),

        order_gross_profit=(
            "gross_profit",
            "sum"
        )

    )
    .reset_index()
)

sales = sales.merge(
    order_totals,
    on="order_id",
    how="left"
)

sales["order_margin"] = np.divide(
    sales["order_gross_profit"],
    sales["order_revenue"],
    out=np.zeros(N_ORDERS),
    where=sales["order_revenue"].to_numpy() != 0
)

# Initialize return-related fields
sales["is_returned"] = False

sales["total_refund"] = 0.00

sales["net_revenue_after_returns"] = (
    sales["order_revenue"]
)

sales["gross_profit_after_returns"] = (
    sales["order_gross_profit"]
)

# ============================================================
# 20. COMMERCIAL FACT TABLE COLLECTION
# ============================================================

facts_commercial = {

    "fact_sales": sales,

    "fact_sales_items": items

}


# ============================================================
# 21. FACT_RETURNS AND FACT_RETURNS_ITEMS
# ============================================================

delivered_orders = sales.loc[
    sales["status"] == "Delivered",
    "order_id"
].to_numpy()

N_RETURNS = int(
    round(N_ORDERS * RETURN_RATE)
)

N_RETURNS = min(
    N_RETURNS,
    len(delivered_orders)
)

selected_return_orders = rng.choice(
    delivered_orders,
    size=N_RETURNS,
    replace=False
)

RETURN_REASONS = [
    "Damaged Product",
    "Wrong Product",
    "Customer Dissatisfaction",
    "Late Delivery",
    "Product Defect"
]

return_ids = np.arange(
    1,
    N_RETURNS + 1
)

return_records = []
return_item_records = []

return_item_id = 1

items_by_order = {
    order_id: group
    for order_id, group in items.groupby("order_id")
}

delivery_map = sales.set_index(
    "order_id"
)["delivered_date"]

for return_id, order_id in zip(
    return_ids,
    selected_return_orders
):

    order_items = items_by_order[order_id]

    selected_count = int(
        rng.integers(
            1,
            len(order_items) + 1
        )
    )

    selected_items = order_items.sample(
        n=selected_count,
        random_state=int(
            rng.integers(0, 2**31 - 1)
        )
    )

    delivery_date = delivery_map.loc[order_id]

    return_date = (
        delivery_date
        + pd.Timedelta(
            days=int(rng.integers(1, 31))
        )
    )

    return_reason = rng.choice(
        RETURN_REASONS,
        p=[
            0.25,
            0.15,
            0.25,
            0.15,
            0.20
        ]
    )

    return_status = rng.choice(
        ["Approved", "Processing"],
        p=[0.85, 0.15]
    )

    total_refund = 0.00

    for _, item in selected_items.iterrows():

        purchased_quantity = int(
            item["quantity"]
        )

        returned_quantity = int(
            rng.integers(
                1,
                purchased_quantity + 1
            )
        )

        original_item_revenue = round(
            float(item["net_revenue"])
            * returned_quantity
            / purchased_quantity,
            2
        )

        refund_percentage = float(
            rng.uniform(0.70, 1.00)
        )

        if return_status == "Approved":

            refund_value = round(
                original_item_revenue
                * refund_percentage,
                2
            )

        else:

            refund_value = 0.00

        total_refund += refund_value

        return_item_records.append({

            "return_item_id": return_item_id,

            "return_id": return_id,

            "order_id": order_id,

            "product_id": int(
                item["product_id"]
            ),

            "original_item_id": int(
                item["item_id"]
            ),

            "returned_quantity": returned_quantity,

            "original_quantity": purchased_quantity,

            "original_item_revenue": original_item_revenue,

            "refund_percentage": round(
                refund_percentage,
                4
            ),

            "refund_value": refund_value,

            "return_status": return_status

        })

        return_item_id += 1

    return_records.append({

        "return_id": return_id,

        "order_id": order_id,

        "return_date": return_date,

        "return_reason": return_reason,

        "return_status": return_status,

        "refund_value": round(
            total_refund,
            2
        )

    })

returns = pd.DataFrame(return_records)

returns_items = pd.DataFrame(
    return_item_records
)

# ============================================================
# 22. UPDATE FACT_SALES_ITEMS
# ============================================================

refund_by_item = (
    returns_items.groupby(
        "original_item_id"
    )["refund_value"]
    .sum()
)

returned_quantity_by_item = (
    returns_items.groupby(
        "original_item_id"
    )["returned_quantity"]
    .sum()
)

items["refund_value"] = (
    items["item_id"]
    .map(refund_by_item)
    .fillna(0.00)
    .round(2)
)

items["returned_quantity"] = (
    items["item_id"]
    .map(returned_quantity_by_item)
    .fillna(0)
    .astype(int)
)

items["net_revenue_after_returns"] = (
    items["net_revenue"]
    - items["refund_value"]
).round(2)

items["gross_profit_after_returns"] = (
    items["net_revenue_after_returns"]
    - items["line_cost"]
).round(2)

items["gross_margin_after_returns"] = np.divide(
    items["gross_profit_after_returns"],
    items["net_revenue_after_returns"],
    out=np.zeros(len(items)),
    where=items["net_revenue_after_returns"].to_numpy() != 0
)

# ============================================================
# 23. UPDATE FACT_SALES
# ============================================================

refund_by_order = (
    returns.groupby("order_id")[
        "refund_value"
    ].sum()
)

sales["total_refund"] = (
    sales["order_id"]
    .map(refund_by_order)
    .fillna(0.00)
    .round(2)
)

returned_order_set = set(
    returns["order_id"]
)

sales["is_returned"] = (
    sales["order_id"].isin(
        returned_order_set
    )
)

sales["net_revenue_after_returns"] = (
    sales["order_revenue"]
    - sales["total_refund"]
).round(2)

sales["gross_profit_after_returns"] = (
    sales["net_revenue_after_returns"]
    - sales["order_cost"]
).round(2)

sales["net_margin_after_returns"] = np.divide(
    sales["gross_profit_after_returns"],
    sales["net_revenue_after_returns"],
    out=np.zeros(len(sales)),
    where=sales["net_revenue_after_returns"].to_numpy() != 0
)

# ============================================================
# 24. UPDATE CUSTOMER DIMENSION
# ============================================================

customer_purchases = (
    sales.loc[
        sales["is_commercial"]
    ]
    .groupby("customer_id")
    .agg(

        total_orders=(
            "order_id",
            "count"
        ),

        first_purchase_date=(
            "order_date",
            "min"
        ),

        last_purchase_date=(
            "order_date",
            "max"
        ),

        total_revenue=(
            "order_revenue",
            "sum"
        ),

        total_refunds=(
            "total_refund",
            "sum"
        )

    )
    .reset_index()
)

customers = customers.merge(
    customer_purchases,
    on="customer_id",
    how="left"
)

customers["total_orders"] = (
    customers["total_orders"]
    .fillna(0)
    .astype(int)
)

customers["total_revenue"] = (
    customers["total_revenue"]
    .fillna(0.00)
)

customers["total_refunds"] = (
    customers["total_refunds"]
    .fillna(0.00)
)

customers["is_buyer"] = (
    customers["total_orders"] > 0
)

customers["customer_type"] = np.where(
    customers["is_buyer"],
    "Buyer",
    "Registered Only"
)

customers["customer_net_revenue"] = (
    customers["total_revenue"]
    - customers["total_refunds"]
).round(2)

# ============================================================
# 25. FACT_INVENTORY - MONTHLY SNAPSHOTS
# ============================================================

inventory_months = pd.date_range(
    START_DATE,
    END_DATE,
    freq="MS"
)

inventory_index = pd.MultiIndex.from_product(
    [
        inventory_months,
        product_ids,
        store_ids
    ],
    names=[
        "inventory_date",
        "product_id",
        "store_id"
    ]
)

inventory = inventory_index.to_frame(
    index=False
)

inventory["stock_quantity"] = rng.integers(
    0,
    301,
    len(inventory)
)

inventory["reorder_point"] = rng.integers(
    10,
    50,
    len(inventory)
)

inventory["unit_cost"] = (
    inventory["product_id"]
    .map(
        products.set_index(
            "product_id"
        )["unit_cost"]
    )
)

inventory["stock_value"] = (
    inventory["stock_quantity"]
    * inventory["unit_cost"]
).round(2)

inventory["stock_status"] = np.select(
    [
        inventory["stock_quantity"] == 0,

        inventory["stock_quantity"]
        <= inventory["reorder_point"]
    ],
    [
        "Stockout",
        "Low Stock"
    ],
    default="Normal"
)

inventory["record_type"] = (
    "Synthetic Monthly Snapshot"
)

# ============================================================
# 26. FACT_MARKETING - SYNTHETIC CAMPAIGNS
# ============================================================

N_CAMPAIGNS = 300

marketing = pd.DataFrame({

    "campaign_id": np.arange(
        1,
        N_CAMPAIGNS + 1
    ),

    "campaign_name": [
        f"Vision Campaign {i:03d}"
        for i in range(1, N_CAMPAIGNS + 1)
    ],

    "campaign_date": rng.choice(
        inventory_months,
        N_CAMPAIGNS
    ),

    "channel": rng.choice(
        [
            "Google Ads",
            "Meta Ads",
            "Email Marketing",
            "Affiliates",
            "Influencers"
        ],
        N_CAMPAIGNS
    ),

    "campaign_cost": rng.uniform(
        500,
        15000,
        N_CAMPAIGNS
    ).round(2),

    "impressions": rng.integers(
        10000,
        500000,
        N_CAMPAIGNS
    ),

    "clicks": rng.integers(
        500,
        30000,
        N_CAMPAIGNS
    ),

    "conversions": rng.integers(
        10,
        1500,
        N_CAMPAIGNS
    )

})

marketing["attributed_revenue"] = (
    marketing["campaign_cost"]
    * rng.uniform(
        2,
        8,
        N_CAMPAIGNS
    )
).round(2)

marketing["roas"] = (
    marketing["attributed_revenue"]
    / marketing["campaign_cost"]
).round(2)

marketing["attribution_type"] = (
    "Synthetic Estimate"
)

marketing["is_verified_attribution"] = False


# ============================================================
# 27. FACT_FINANCIAL - FINANCIAL PERFORMANCE
# ============================================================

monthly_revenue = (
    sales.groupby(
        sales["order_date"].dt.to_period("M")
    )["order_revenue"]
    .sum()
)

monthly_cost = (
    sales.groupby(
        sales["order_date"].dt.to_period("M")
    )["order_cost"]
    .sum()
)

monthly_refunds = (
    returns.groupby(
        returns["return_date"].dt.to_period("M")
    )["refund_value"]
    .sum()
)

financial = pd.DataFrame({
    "month": inventory_months
})

financial["period"] = (
    financial["month"].dt.to_period("M")
)

financial["revenue"] = (
    financial["period"]
    .map(monthly_revenue)
    .fillna(0.00)
)

financial["product_cost"] = (
    financial["period"]
    .map(monthly_cost)
    .fillna(0.00)
)

financial["refund_expense"] = (
    financial["period"]
    .map(monthly_refunds)
    .fillna(0.00)
)

financial["net_revenue_after_returns"] = (
    financial["revenue"]
    - financial["refund_expense"]
).round(2)

financial["gross_profit"] = (
    financial["revenue"]
    - financial["product_cost"]
).round(2)

financial["gross_profit_after_returns"] = (
    financial["net_revenue_after_returns"]
    - financial["product_cost"]
).round(2)

financial["operating_expenses"] = (
    financial["revenue"]
    * rng.uniform(
        0.08,
        0.18,
        len(financial)
    )
).round(2)

financial["administrative_expenses"] = (
    financial["revenue"]
    * rng.uniform(
        0.02,
        0.06,
        len(financial)
    )
).round(2)

financial["commercial_expenses"] = (
    financial["revenue"]
    * rng.uniform(
        0.02,
        0.05,
        len(financial)
    )
).round(2)

financial["depreciation_amortization"] = (
    financial["revenue"] * 0.02
).round(2)

financial["ebitda"] = (
    financial["gross_profit_after_returns"]
    - financial["operating_expenses"]
    - financial["administrative_expenses"]
    - financial["commercial_expenses"]
).round(2)

financial["cash_inflow"] = (
    financial["revenue"]
    * rng.uniform(
        0.90,
        1.00,
        len(financial)
    )
).round(2)

financial["cash_outflow"] = (
    financial["product_cost"]
    + financial["refund_expense"]
    + financial["operating_expenses"]
    + financial["administrative_expenses"]
    + financial["commercial_expenses"]
).round(2)

financial["cash_flow"] = (
    financial["cash_inflow"]
    - financial["cash_outflow"]
).round(2)

financial["data_type"] = (
    "Synthetic Financial Simulation"
)

financial = financial.drop(
    columns="period"
)

# ============================================================
# 28. FACT_TARGETS - MONTHLY TARGETS
# ============================================================

targets = pd.DataFrame({
    "month": inventory_months
})

number_of_months = len(targets)

targets["revenue_target"] = (
    TARGET_REVENUE / number_of_months
)

targets["gross_profit_target"] = (
    TARGET_GROSS_PROFIT / number_of_months
)

targets["orders_target"] = (
    N_ORDERS / number_of_months
)

targets["margin_target"] = (
    TARGET_MARGIN
)

targets["target_type"] = (
    "Uniform Monthly Target"
)

targets["target_method"] = (
    "Linear Distribution"
)

# ============================================================
# 29. FACT TABLE COLLECTION
# ============================================================

facts = {

    "fact_sales": sales,

    "fact_sales_items": items,

    "fact_returns": returns,

    "fact_returns_items": returns_items,

    "fact_inventory": inventory,

    "fact_marketing": marketing,

    "fact_financial": financial,

    "fact_targets": targets

}

# ============================================================
# 30. BUSINESS RULES TABLE
# ============================================================

business_rules = pd.DataFrame({

    "rule": list(
        BUSINESS_RULES.keys()
    ),

    "description": list(
        BUSINESS_RULES.values()
    )

})

# ============================================================
# 31. DATASET METADATA
# ============================================================

dataset_metadata = pd.DataFrame({

    "parameter": [

        "Company",
        "Dataset Type",
        "Currency",
        "Country",
        "Start Date",
        "End Date",
        "Registered Customers",
        "Orders",
        "Products",
        "Sellers",
        "Stores",
        "Target Revenue",
        "Target Gross Profit",
        "Target Gross Margin",
        "Return Rate",
        "Cancellation Rate"

    ],

    "value": [

        COMPANY,
        "Synthetic",
        "BRL",
        "Brazil",
        str(START_DATE.date()),
        str(END_DATE.date()),
        N_CUSTOMERS,
        N_ORDERS,
        N_PRODUCTS,
        N_SELLERS,
        N_STORES,
        TARGET_REVENUE,
        TARGET_GROSS_PROFIT,
        TARGET_MARGIN,
        RETURN_RATE,
        CANCELLATION_RATE

    ]

})

# ============================================================
# 32. EXPORT ALL TABLES
# ============================================================

all_tables = {

    **dimensions,

    **facts,

    "business_rules": business_rules

}

for table_name, dataframe in all_tables.items():

    file_path = os.path.join(
        RAW_DIR,
        f"{table_name}.csv"
    )

    dataframe.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d"
    )

dataset_metadata.to_csv(
    os.path.join(
        RAW_DIR,
        "dataset_metadata.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 33. FINAL DATASET SUMMARY
# ============================================================

print("\n" + "=" * 65)

print("VISION GROUP - DATASET GENERATION COMPLETED")

print("=" * 65)

print(f"Company: {COMPANY}")

print(
    f"Registered customers: "
    f"{len(customers):,}"
)

print(
    f"Purchasing customers: "
    f"{customers['is_buyer'].sum():,}"
)

print(
    f"Orders: "
    f"{len(sales):,}"
)

print(
    f"Canceled orders: "
    f"{sales['is_canceled'].sum():,}"
)

print(
    f"Products: "
    f"{len(products):,}"
)

print(
    f"Returns: "
    f"{len(returns):,}"
)

print(
    f"Returned items: "
    f"{len(returns_items):,}"
)

print(
    f"Commercial revenue: "
    f"R$ {sales['order_revenue'].sum():,.2f}"
)

print(
    f"Commercial product cost: "
    f"R$ {sales['order_cost'].sum():,.2f}"
)

print(
    f"Gross profit: "
    f"R$ {sales['order_gross_profit'].sum():,.2f}"
)

print(
    f"Gross margin: "
    f"{sales['order_gross_profit'].sum() / sales['order_revenue'].sum():.2%}"
)

print(
    f"Total refunds: "
    f"R$ {returns['refund_value'].sum():,.2f}"
)

print(
    f"Revenue after returns: "
    f"R$ {sales['net_revenue_after_returns'].sum():,.2f}"
)

print("\nGENERATED TABLES")

for table_name, dataframe in all_tables.items():

    print(
        f"{table_name:<25} "
        f"{len(dataframe):>12,} rows"
    )

print("\nOUTPUT DIRECTORY:")

print(
    os.path.abspath(RAW_DIR)
)

print("\nDATASET GENERATION FINISHED!")
