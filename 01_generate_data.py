import os
import numpy as np
import pandas as pd

# ============================================================
# 1. GENERAL CONFIGURATION
# ============================================================

COMPANY = "Vision Group"

SEED = 42

rng = np.random.default_rng(SEED)

OUTPUT_DIR = "vision_group_dataset"
RAW_DIR = os.path.join(
    OUTPUT_DIR,
    "raw"
)

os.makedirs(
    RAW_DIR,
    exist_ok=True
)

# ============================================================
# 2. BUSINESS PARAMETERS
# ============================================================

START_DATE = pd.Timestamp("2021-01-01")
END_DATE = pd.Timestamp("2025-12-31")

N_CUSTOMERS = 82_400
N_PRODUCTS = 5_000
N_ORDERS = 145_200

N_SELLERS = 350
N_STORES = 30

AVG_ITEMS_PER_ORDER = 2.25

RETURN_RATE = 0.10
CANCELLATION_RATE = 0.05

TARGET_REVENUE = 8_200_000.00
TARGET_COST = 6_100_000.00
TARGET_GROSS_PROFIT = 2_100_000.00

TARGET_MARGIN = (
    TARGET_GROSS_PROFIT / TARGET_REVENUE
)

N_CAMPAIGNS = 300

# ============================================================
# 3. MONEY ALLOCATION FUNCTION
# ============================================================

def allocate_money(total_value, weights):

    weights = np.asarray(
        weights,
        dtype=float
    )

    if len(weights) == 0:
        return np.array([], dtype=np.int64)

    weights = np.maximum(
        weights,
        0
    )

    if weights.sum() == 0:

        weights = np.ones(
            len(weights),
            dtype=float
        )

    total_cents = int(
        round(total_value * 100)
    )

    exact_values = (
        weights / weights.sum()
    ) * total_cents

    allocated = np.floor(
        exact_values
    ).astype(np.int64)

    remaining = int(
        total_cents - allocated.sum()
    )

    fractions = (
        exact_values - allocated
    )

    if remaining > 0:

        largest_remainders = np.argsort(
            -fractions
        )

        allocated[
            largest_remainders[:remaining]
        ] += 1

    elif remaining < 0:

        smallest_remainders = np.argsort(
            fractions
        )

        allocated[
            smallest_remainders[:abs(remaining)]
        ] -= 1

    return allocated


def cents_to_currency(values):

    return (
        np.asarray(
            values,
            dtype=np.int64
        ) / 100
    ).round(2)


# ============================================================
# 4. REFERENCE DATA - BRAZILIAN STATES AND CITIES
# ============================================================

BRAZIL_CITIES = {

    "AC": {
        "region": "North",
        "cities": ["Rio Branco", "Cruzeiro do Sul"]
    },

    "AL": {
        "region": "Northeast",
        "cities": ["Maceio", "Arapiraca"]
    },

    "AP": {
        "region": "North",
        "cities": ["Macapa", "Santana"]
    },

    "AM": {
        "region": "North",
        "cities": ["Manaus", "Parintins"]
    },

    "BA": {
        "region": "Northeast",
        "cities": ["Salvador", "Feira de Santana", "Vitoria da Conquista"]
    },

    "CE": {
        "region": "Northeast",
        "cities": ["Fortaleza", "Juazeiro do Norte"]
    },

    "DF": {
        "region": "Central-West",
        "cities": ["Brasilia"]
    },

    "ES": {
        "region": "Southeast",
        "cities": ["Vitoria", "Vila Velha"]
    },

    "GO": {
        "region": "Central-West",
        "cities": ["Goiania", "Anapolis"]
    },

    "MA": {
        "region": "Northeast",
        "cities": ["Sao Luis", "Imperatriz"]
    },

    "MT": {
        "region": "Central-West",
        "cities": ["Cuiaba", "Rondonopolis"]
    },

    "MS": {
        "region": "Central-West",
        "cities": ["Campo Grande", "Dourados"]
    },

    "MG": {
        "region": "Southeast",
        "cities": ["Belo Horizonte", "Uberlandia", "Juiz de Fora"]
    },

    "PA": {
        "region": "North",
        "cities": ["Belem", "Santarem"]
    },

    "PB": {
        "region": "Northeast",
        "cities": ["Joao Pessoa", "Campina Grande"]
    },

    "PR": {
        "region": "South",
        "cities": ["Curitiba", "Londrina", "Maringa"]
    },

    "PE": {
        "region": "Northeast",
        "cities": ["Recife", "Caruaru"]
    },

    "PI": {
        "region": "Northeast",
        "cities": ["Teresina", "Parnaiba"]
    },

    "RJ": {
        "region": "Southeast",
        "cities": ["Rio de Janeiro", "Niteroi", "Petropolis"]
    },

    "RN": {
        "region": "Northeast",
        "cities": ["Natal", "Mossoro"]
    },

    "RS": {
        "region": "South",
        "cities": ["Porto Alegre", "Caxias do Sul", "Pelotas"]
    },

    "RO": {
        "region": "North",
        "cities": ["Porto Velho", "Ji-Parana"]
    },

    "RR": {
        "region": "North",
        "cities": ["Boa Vista"]
    },

    "SC": {
        "region": "South",
        "cities": ["Florianopolis", "Joinville", "Blumenau"]
    },

    "SP": {
        "region": "Southeast",
        "cities": ["Sao Paulo", "Campinas", "Santos", "Ribeirao Preto"]
    },

    "SE": {
        "region": "Northeast",
        "cities": ["Aracaju", "Itabaiana"]
    },

    "TO": {
        "region": "North",
        "cities": ["Palmas", "Araguaina"]
    }

}

# ============================================================
# 5. GEOGRAPHIC REFERENCE TABLE
# ============================================================

geographic_records = []

for state, information in BRAZIL_CITIES.items():

    for city in information["cities"]:

        geographic_records.append({

            "city": city,

            "state": state,

            "region": information["region"],

            "country": "Brazil"

        })

dim_geography = pd.DataFrame(
    geographic_records
)

# ============================================================
# 6. DIM_REGIONS
# ============================================================

dim_regions = (
    dim_geography[
        ["region"]
    ]
    .drop_duplicates()
    .sort_values("region")
    .reset_index(drop=True)
)

dim_regions.insert(
    0,
    "region_id",
    np.arange(
        1,
        len(dim_regions) + 1
    )
)

# ============================================================
# 7. DIM_CATEGORIES
# ============================================================

CATEGORY_STRUCTURE = {

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

category_records = []

category_id = 1

subcategory_id = 1

for category, subcategories in CATEGORY_STRUCTURE.items():

    for subcategory in subcategories:

        category_records.append({

            "category_id": category_id,

            "category_name": category,

            "subcategory_id": subcategory_id,

            "subcategory_name": subcategory

        })

        subcategory_id += 1

    category_id += 1

dim_categories = pd.DataFrame(
    category_records
)

# ============================================================
# 8. DIM_CUSTOMERS
# ============================================================

customer_ids = np.arange(
    1,
    N_CUSTOMERS + 1
)

customer_geo_indices = rng.integers(
    0,
    len(dim_geography),
    size=N_CUSTOMERS
)

customer_geography = (
    dim_geography.iloc[
        customer_geo_indices
    ]
    .reset_index(drop=True)
)

customer_signup_dates = (
    pd.Timestamp("2019-01-01")
    + pd.to_timedelta(
        rng.integers(
            0,
            (END_DATE - pd.Timestamp("2019-01-01")).days + 1,
            size=N_CUSTOMERS
        ),
        unit="D"
    )
)

dim_customers = pd.DataFrame({

    "customer_id": customer_ids,

    "customer_unique_id": [
        f"CUST-{i:07d}"
        for i in customer_ids
    ],

    "city": customer_geography["city"].to_numpy(),

    "state": customer_geography["state"].to_numpy(),

    "region": customer_geography["region"].to_numpy(),

    "country": "Brazil",

    "signup_date": customer_signup_dates,

    "customer_segment": rng.choice(
        [
            "New",
            "Regular",
            "Loyal",
            "VIP"
        ],
        size=N_CUSTOMERS,
        p=[
            0.30,
            0.40,
            0.22,
            0.08
        ]
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
        size=N_CUSTOMERS,
        p=[
            0.25,
            0.20,
            0.18,
            0.12,
            0.10,
            0.15
        ]
    )

})

# ============================================================
# 9. DIM_PRODUCTS
# ============================================================

product_ids = np.arange(
    1,
    N_PRODUCTS + 1
)

category_names = list(
    CATEGORY_STRUCTURE.keys()
)

product_categories = rng.choice(
    category_names,
    size=N_PRODUCTS
)

product_records = []

for product_id, category in zip(
    product_ids,
    product_categories
):

    subcategory = rng.choice(
        CATEGORY_STRUCTURE[category]
    )

    reference_cost = float(
        np.round(
            rng.lognormal(
                mean=3.0,
                sigma=0.8
            ),
            2
        )
    )

    product_records.append({

        "product_id": product_id,

        "product_name": f"Vision Product {product_id:05d}",

        "category_name": category,

        "subcategory_name": subcategory,

        "unit_cost": reference_cost,

        "brand": rng.choice([
            "Vision",
            "Premium",
            "Essentials",
            "Select",
            "Urban"
        ]),

        "product_status": "Active"

    })

dim_products = pd.DataFrame(
    product_records
)

# Add category identifiers
dim_products = dim_products.merge(
    dim_categories[
        [
            "category_id",
            "category_name",
            "subcategory_id",
            "subcategory_name"
        ]
    ],
    on=[
        "category_name",
        "subcategory_name"
    ],
    how="left",
    validate="many_to_one"
)

# ============================================================
# 10. DIM_STORES
# ============================================================

store_geo_indices = rng.choice(
    len(dim_geography),
    size=N_STORES,
    replace=True
)

store_geography = (
    dim_geography.iloc[
        store_geo_indices
    ]
    .reset_index(drop=True)
)

dim_stores = pd.DataFrame({

    "store_id": np.arange(
        1,
        N_STORES + 1
    ),

    "store_name": [
        f"Vision Store {i:03d}"
        for i in range(1, N_STORES + 1)
    ],

    "city": store_geography["city"].to_numpy(),

    "state": store_geography["state"].to_numpy(),

    "region": store_geography["region"].to_numpy(),

    "country": "Brazil",

    "store_type": rng.choice(
        [
            "Flagship",
            "Standard",
            "Express"
        ],
        size=N_STORES,
        p=[
            0.20,
            0.60,
            0.20
        ]
    ),

    "store_status": "Active"

})

# ============================================================
# 11. DIM_SELLERS
# ============================================================

seller_ids = np.arange(
    1,
    N_SELLERS + 1
)

seller_store_ids = rng.integers(
    1,
    N_STORES + 1,
    size=N_SELLERS
)

dim_sellers = pd.DataFrame({

    "seller_id": seller_ids,

    "seller_name": [
        f"Vision Seller {i:04d}"
        for i in seller_ids
    ],

    "store_id": seller_store_ids,

    "hire_date": (
        pd.Timestamp("2018-01-01")
        + pd.to_timedelta(
            rng.integers(
                0,
                (END_DATE - pd.Timestamp("2018-01-01")).days + 1,
                size=N_SELLERS
            ),
            unit="D"
        )
    ),

    "seller_status": "Active"

})

# ============================================================
# 12. DIM_DATE
# ============================================================

calendar_dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

dim_date = pd.DataFrame({

    "date": calendar_dates,

    "year": calendar_dates.year,

    "quarter": calendar_dates.quarter,

    "month": calendar_dates.month,

    "month_name": calendar_dates.strftime("%B"),

    "year_month": calendar_dates.strftime("%Y-%m"),

    "day": calendar_dates.day,

    "day_name": calendar_dates.strftime("%A"),

    "week_of_year": calendar_dates.isocalendar().week.to_numpy(),

    "is_weekend": calendar_dates.dayofweek >= 5

})

dim_date.insert(
    0,
    "date_id",
    np.arange(
        1,
        len(dim_date) + 1
    )
)

# ============================================================
# 13. DIMENSION COLLECTION
# ============================================================

dimensions = {

    "dim_customers": dim_customers,

    "dim_products": dim_products,

    "dim_categories": dim_categories,

    "dim_sellers": dim_sellers,

    "dim_stores": dim_stores,

    "dim_regions": dim_regions,

    "dim_date": dim_date,

    "dim_geography": dim_geography

}


# ============================================================
# 14. ORDER GENERATION
# ============================================================

order_ids = np.arange(
    1,
    N_ORDERS + 1
)

order_customer_ids = rng.integers(
    1,
    N_CUSTOMERS + 1,
    size=N_ORDERS
)

customer_signup_map = (
    dim_customers.set_index(
        "customer_id"
    )["signup_date"]
)

assigned_signup_dates = (
    customer_signup_map
    .reindex(order_customer_ids)
    .reset_index(drop=True)
)

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

order_dates = pd.DatetimeIndex(
    pd.concat(
        [
            pd.Series(random_order_dates),
            assigned_signup_dates
        ],
        axis=1
    ).max(axis=1)
)

# Keep all orders inside the historical period
order_dates = pd.DatetimeIndex(
    np.minimum(
        order_dates.to_numpy(),
        END_DATE.to_datetime64()
    )
)

# ============================================================
# 15. ORDER STATUS
# ============================================================

order_statuses = rng.choice(
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

promised_delivery_dates = (
    order_dates
    + pd.to_timedelta(
        rng.integers(
            3,
            11,
            size=N_ORDERS
        ),
        unit="D"
    )
)

delivery_days = rng.integers(
    2,
    15,
    size=N_ORDERS
)

actual_delivery_dates = (
    order_dates
    + pd.to_timedelta(
        delivery_days,
        unit="D"
    )
)

delivered_dates = pd.Series(
    pd.NaT,
    index=np.arange(N_ORDERS),
    dtype="datetime64[ns]"
)

delivered_mask = (
    order_statuses == "Delivered"
)

delivered_dates.loc[
    delivered_mask
] = actual_delivery_dates[
    delivered_mask
]

canceled_dates = pd.Series(
    pd.NaT,
    index=np.arange(N_ORDERS),
    dtype="datetime64[ns]"
)

canceled_mask = (
    order_statuses == "Canceled"
)

canceled_dates.loc[
    canceled_mask
] = (
    order_dates[canceled_mask]
    + pd.to_timedelta(
        rng.integers(
            0,
            4,
            size=canceled_mask.sum()
        ),
        unit="D"
    )
)

# ============================================================
# 16. SALES GEOGRAPHY AND ASSIGNMENTS
# ============================================================

customer_state_map = (
    dim_customers.set_index(
        "customer_id"
    )["state"]
)

customer_city_map = (
    dim_customers.set_index(
        "customer_id"
    )["city"]
)

customer_region_map = (
    dim_customers.set_index(
        "customer_id"
    )["region"]
)

order_states = (
    pd.Series(order_customer_ids)
    .map(customer_state_map)
)

order_cities = (
    pd.Series(order_customer_ids)
    .map(customer_city_map)
)

order_regions = (
    pd.Series(order_customer_ids)
    .map(customer_region_map)
)

order_store_ids = rng.integers(
    1,
    N_STORES + 1,
    size=N_ORDERS
)

order_seller_ids = rng.integers(
    1,
    N_SELLERS + 1,
    size=N_ORDERS
)

# ============================================================
# 17. FACT_SALES
# ============================================================

sales = pd.DataFrame({

    "order_id": order_ids,

    "customer_id": order_customer_ids,

    "seller_id": order_seller_ids,

    "store_id": order_store_ids,

    "order_date": order_dates,

    "status": order_statuses,

    "promised_delivery_date": promised_delivery_dates,

    "delivered_date": delivered_dates.to_numpy(),

    "canceled_at": canceled_dates.to_numpy(),

    "city": order_cities.to_numpy(),

    "state": order_states.to_numpy(),

    "region": order_regions.to_numpy(),

    "country": "Brazil",

    "sales_channel": rng.choice(
        [
            "Website",
            "Mobile App",
            "Marketplace",
            "Physical Store"
        ],
        size=N_ORDERS,
        p=[
            0.40,
            0.30,
            0.20,
            0.10
        ]
    ),

    "shipping_cost": np.round(
        rng.uniform(
            5,
            50,
            size=N_ORDERS
        ),
        2
    )

})

sales["is_canceled"] = (
    sales["status"] == "Canceled"
)

sales["is_commercial"] = (
    ~sales["is_canceled"]
)

# ============================================================
# 18. FACT_SALES_ITEMS - ITEM GENERATION
# ============================================================

item_counts = rng.choice(
    [1, 2, 3, 4],
    size=N_ORDERS,
    p=[
        0.20,
        0.45,
        0.25,
        0.10
    ]
)

item_order_ids = np.repeat(
    order_ids,
    item_counts
)

N_ITEMS = len(
    item_order_ids
)

item_ids = np.arange(
    1,
    N_ITEMS + 1
)

item_product_ids = rng.integers(
    1,
    N_PRODUCTS + 1,
    size=N_ITEMS
)

quantities = rng.integers(
    1,
    6,
    size=N_ITEMS
)

prices = np.round(
    rng.lognormal(
        mean=3.3,
        sigma=0.7,
        size=N_ITEMS
    ),
    2
)

discount_rates = rng.choice(
    [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30
    ],
    size=N_ITEMS,
    p=[
        0.45,
        0.15,
        0.15,
        0.12,
        0.08,
        0.05
    ]
)

gross_merchandise_value = (
    quantities * prices
).round(2)

discount_values = (
    gross_merchandise_value
    * discount_rates
).round(2)

freight_values = np.round(
    rng.uniform(
        2,
        35,
        size=N_ITEMS
    ),
    2
)

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
        size=N_ITEMS
    ),
    None
)

# ============================================================
# 19. ADD PRODUCT REFERENCE COST
# ============================================================

product_reference_cost_map = (
    dim_products.set_index(
        "product_id"
    )["unit_cost"]
)

unit_cost_reference = (
    pd.Series(item_product_ids)
    .map(product_reference_cost_map)
    .to_numpy()
)

# ============================================================
# 20. INITIAL FACT_SALES_ITEMS
# ============================================================

items = pd.DataFrame({

    "item_id": item_ids,

    "order_id": item_order_ids,

    "product_id": item_product_ids,

    "quantity": quantities,

    "price": prices,

    "gross_merchandise_value": gross_merchandise_value,

    "discount_rate": discount_rates,

    "discount": discount_values,

    "promotion_id": promotion_ids,

    "freight_value": freight_values,

    "unit_cost_reference": unit_cost_reference

})

# ============================================================
# 21. COMMERCIAL CALIBRATION
# ============================================================

canceled_order_ids = set(
    sales.loc[
        sales["is_canceled"],
        "order_id"
    ]
)

eligible_mask = (
    ~items["order_id"].isin(
        canceled_order_ids
    )
)

items["net_revenue"] = 0.00
items["line_cost"] = 0.00

# ------------------------------------------------------------
# REVENUE CALIBRATION
# ------------------------------------------------------------

initial_item_revenue = (

    items["gross_merchandise_value"].to_numpy()

    - items["discount"].to_numpy()

    + items["freight_value"].to_numpy()

)

eligible_positions = np.flatnonzero(
    eligible_mask.to_numpy()
)

allocated_revenue = allocate_money(
    TARGET_REVENUE,
    initial_item_revenue[
        eligible_positions
    ]
)

items.loc[
    eligible_mask,
    "net_revenue"
] = cents_to_currency(
    allocated_revenue
)

# ------------------------------------------------------------
# COST CALIBRATION
# ------------------------------------------------------------

initial_item_cost = (
    items["quantity"].to_numpy()
    * items["unit_cost_reference"].to_numpy()
)

allocated_cost = allocate_money(
    TARGET_COST,
    initial_item_cost[
        eligible_positions
    ]
)

items.loc[
    eligible_mask,
    "line_cost"
] = cents_to_currency(
    allocated_cost
)

# ------------------------------------------------------------
# CANCELED ITEMS
# ------------------------------------------------------------

items.loc[
    ~eligible_mask,
    [
        "net_revenue",
        "line_cost"
    ]
] = 0.00

items["is_canceled"] = (
    ~eligible_mask
)

items["is_commercial"] = (
    eligible_mask
)

# ============================================================
# 22. REALIZED UNIT COST AND GROSS PROFIT
# ============================================================

items["unit_cost"] = np.divide(
    items["line_cost"].to_numpy(),
    items["quantity"].to_numpy(),
    out=np.zeros(N_ITEMS),
    where=items["quantity"].to_numpy() != 0
).round(2)

items["gross_profit"] = (
    items["net_revenue"]
    - items["line_cost"]
).round(2)

items["gross_margin"] = np.divide(
    items["gross_profit"].to_numpy(),
    items["net_revenue"].to_numpy(),
    out=np.zeros(N_ITEMS),
    where=items["net_revenue"].to_numpy() != 0
)

# ============================================================
# 23. UPDATE FACT_SALES FROM FACT_SALES_ITEMS
# ============================================================

sales_totals = (
    items.groupby("order_id")
    .agg(

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
    sales_totals,
    on="order_id",
    how="left",
    validate="one_to_one"
)

sales[
    [
        "order_revenue",
        "order_cost",
        "order_gross_profit"
    ]
] = sales[
    [
        "order_revenue",
        "order_cost",
        "order_gross_profit"
    ]
].fillna(0.00)

sales["total_refund"] = 0.00

sales["is_returned"] = False

sales["net_revenue_after_returns"] = (
    sales["order_revenue"]
)

sales["gross_profit_after_returns"] = (
    sales["order_gross_profit"]
)

sales["net_margin_after_returns"] = np.divide(
    sales["gross_profit_after_returns"].to_numpy(),
    sales["net_revenue_after_returns"].to_numpy(),
    out=np.zeros(N_ORDERS),
    where=sales["net_revenue_after_returns"].to_numpy() != 0
)

# ============================================================
# 24. COMMERCIAL TARGETS AND RETURN CONFIGURATION
# ============================================================

# Return rate is calculated against delivered orders.

delivered_orders = sales.loc[
    sales["status"] == "Delivered",
    "order_id"
].to_numpy()

N_DELIVERED_ORDERS = len(
    delivered_orders
)

N_RETURNS = min(
    int(
        round(
            N_DELIVERED_ORDERS
            * RETURN_RATE
        )
    ),
    N_DELIVERED_ORDERS
)

# ============================================================
# 25. RETURN CONFIGURATION
# ============================================================

RETURN_REASONS = [
    "Damaged Product",
    "Wrong Product",
    "Customer Dissatisfaction",
    "Late Delivery",
    "Product Defect"
]

RETURN_STATUSES = [
    "Approved",
    "Processing"
]

# ============================================================
# 26. FACT_RETURNS AND FACT_RETURNS_ITEMS
# ============================================================

return_ids = np.arange(
    1,
    N_RETURNS + 1
)

return_records = []

return_item_records = []

return_item_id = 1

# Group original sales items by order
items_by_order = {
    order_id: group
    for order_id, group in items.groupby(
        "order_id",
        sort=False
    )
}

delivery_map = (
    sales.set_index(
        "order_id"
    )["delivered_date"]
)

# Track the quantity already returned
returned_quantity_tracker = {}

for return_id, order_id in zip(
    return_ids,
    delivered_orders[
        rng.permutation(len(delivered_orders))
    ][:N_RETURNS]
):

    order_items = items_by_order[
        order_id
    ]

    available_items = []

    # --------------------------------------------------------
    # IDENTIFY ITEMS AVAILABLE FOR RETURN
    # --------------------------------------------------------

    for _, item in order_items.iterrows():

        item_id = int(
            item["item_id"]
        )

        purchased_quantity = int(
            item["quantity"]
        )

        already_returned = (
            returned_quantity_tracker.get(
                item_id,
                0
            )
        )

        remaining_quantity = (
            purchased_quantity
            - already_returned
        )

        if remaining_quantity > 0:

            available_items.append({

                "item_id": item_id,

                "product_id": int(
                    item["product_id"]
                ),

                "purchased_quantity": purchased_quantity,

                "remaining_quantity": remaining_quantity,

                "net_revenue": float(
                    item["net_revenue"]
                )

            })

    if not available_items:
        continue

    # --------------------------------------------------------
    # SELECT ITEMS FOR RETURN
    # --------------------------------------------------------

    selected_count = int(
        rng.integers(
            1,
            len(available_items) + 1
        )
    )

    selected_indices = rng.choice(
        len(available_items),
        size=selected_count,
        replace=False
    )

    # --------------------------------------------------------
    # RETURN DATE
    # --------------------------------------------------------

    delivery_date = pd.Timestamp(
        delivery_map.loc[order_id]
    )

    return_date = (
        delivery_date
        + pd.Timedelta(
            days=int(
                rng.integers(1, 31)
            )
        )
    )

    # --------------------------------------------------------
    # RETURN CLASSIFICATION
    # --------------------------------------------------------

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
        RETURN_STATUSES,
        p=[
            0.85,
            0.15
        ]
    )

    total_refund = 0.00

    # --------------------------------------------------------
    # GENERATE RETURNED ITEMS
    # --------------------------------------------------------

    for selected_index in selected_indices:

        item = available_items[
            selected_index
        ]

        returned_quantity = int(
            rng.integers(
                1,
                item["remaining_quantity"] + 1
            )
        )

        original_item_revenue = round(
            item["net_revenue"]
            * returned_quantity
            / item["purchased_quantity"],
            2
        )

        refund_percentage = float(
            rng.uniform(
                0.70,
                1.00
            )
        )

        # Only approved returns generate refunds
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

            "product_id": item["product_id"],

            "original_item_id": item["item_id"],

            "returned_quantity": returned_quantity,

            "original_quantity": item["purchased_quantity"],

            "original_item_revenue": original_item_revenue,

            "refund_percentage": round(
                refund_percentage,
                4
            ),

            "refund_value": refund_value,

            "return_status": return_status

        })

        # Update returned quantity tracker
        returned_quantity_tracker[
            item["item_id"]
        ] = (
            returned_quantity_tracker.get(
                item["item_id"],
                0
            )
            + returned_quantity
        )

        return_item_id += 1

    # --------------------------------------------------------
    # GENERATE RETURN HEADER
    # --------------------------------------------------------

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

# Create return tables
returns = pd.DataFrame(
    return_records
)

returns_items = pd.DataFrame(
    return_item_records
)

# ============================================================
# 27. UPDATE FACT_SALES_ITEMS AFTER RETURNS
# ============================================================

if not returns_items.empty:

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

else:

    refund_by_item = pd.Series(
        dtype=float
    )

    returned_quantity_by_item = pd.Series(
        dtype=int
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

# Commercial revenue after refunds
items["net_revenue_after_returns"] = (
    items["net_revenue"]
    - items["refund_value"]
).round(2)

# Original commercial cost is retained.
# No automatic cost reversal is assumed.

items["gross_profit_after_returns"] = (
    items["net_revenue_after_returns"]
    - items["line_cost"]
).round(2)

items["gross_margin_after_returns"] = np.divide(
    items["gross_profit_after_returns"].to_numpy(),
    items["net_revenue_after_returns"].to_numpy(),
    out=np.zeros(len(items)),
    where=items["net_revenue_after_returns"].to_numpy() != 0
)

# ============================================================
# 28. UPDATE FACT_SALES AFTER RETURNS
# ============================================================

if not returns.empty:

    refund_by_order = (
        returns.groupby(
            "order_id"
        )["refund_value"]
        .sum()
    )

else:

    refund_by_order = pd.Series(
        dtype=float
    )

sales["total_refund"] = (
    sales["order_id"]
    .map(refund_by_order)
    .fillna(0.00)
    .round(2)
)

sales["is_returned"] = (
    sales["order_id"].isin(
        returns["order_id"]
        if not returns.empty
        else []
    )
)

sales["net_revenue_after_returns"] = (
    sales["order_revenue"]
    - sales["total_refund"]
).round(2)

# Original cost is not automatically reversed
sales["gross_profit_after_returns"] = (
    sales["net_revenue_after_returns"]
    - sales["order_cost"]
).round(2)

sales["net_margin_after_returns"] = np.divide(
    sales["gross_profit_after_returns"].to_numpy(),
    sales["net_revenue_after_returns"].to_numpy(),
    out=np.zeros(len(sales)),
    where=sales["net_revenue_after_returns"].to_numpy() != 0
)

# ============================================================
# 29. UPDATE CUSTOMER DIMENSION
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

dim_customers = dim_customers.merge(
    customer_purchases,
    on="customer_id",
    how="left",
    validate="one_to_one"
)

dim_customers["total_orders"] = (
    dim_customers["total_orders"]
    .fillna(0)
    .astype(int)
)

dim_customers["total_revenue"] = (
    dim_customers["total_revenue"]
    .fillna(0.00)
)

dim_customers["total_refunds"] = (
    dim_customers["total_refunds"]
    .fillna(0.00)
)

dim_customers["is_buyer"] = (
    dim_customers["total_orders"] > 0
)

dim_customers["customer_type"] = np.where(
    dim_customers["is_buyer"],
    "Buyer",
    "Registered Only"
)

dim_customers["customer_net_revenue"] = (
    dim_customers["total_revenue"]
    - dim_customers["total_refunds"]
).round(2)

# ============================================================
# 30. FACT_INVENTORY - MONTHLY STOCK SNAPSHOTS
# ============================================================

inventory_months = pd.date_range(
    START_DATE,
    END_DATE,
    freq="MS"
)

# Full combination size
total_inventory_combinations = (
    len(inventory_months)
    * N_PRODUCTS
    * N_STORES
)

# Sample 10% of the full product/store/month combinations
# This keeps the dataset manageable.

INVENTORY_SAMPLE_RATE = 0.10

N_INVENTORY_ROWS = int(
    total_inventory_combinations
    * INVENTORY_SAMPLE_RATE
)

inventory_flat_indices = rng.choice(
    total_inventory_combinations,
    size=N_INVENTORY_ROWS,
    replace=False
)

# Convert flattened indexes into dimensions
inventory_store_indices = (
    inventory_flat_indices % N_STORES
)

inventory_product_indices = (
    inventory_flat_indices // N_STORES
) % N_PRODUCTS

inventory_month_indices = (
    inventory_flat_indices
    // (N_STORES * N_PRODUCTS)
)

inventory = pd.DataFrame({

    "inventory_date": inventory_months[
        inventory_month_indices
    ],

    "product_id": (
        inventory_product_indices + 1
    ),

    "store_id": (
        inventory_store_indices + 1
    ),

    "stock_quantity": rng.integers(
        0,
        301,
        size=N_INVENTORY_ROWS
    ),

    "reorder_point": rng.integers(
        10,
        50,
        size=N_INVENTORY_ROWS
    )

})

# Add reference cost from product dimension
inventory["unit_cost_reference"] = (
    inventory["product_id"]
    .map(
        dim_products.set_index(
            "product_id"
        )["unit_cost"]
    )
)

inventory["stock_value"] = (
    inventory["stock_quantity"]
    * inventory["unit_cost_reference"]
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
# 31. FACT_MARKETING - SYNTHETIC CAMPAIGNS
# ============================================================

marketing_channels = [
    "Google Ads",
    "Meta Ads",
    "Email Marketing",
    "Affiliates",
    "Influencers"
]

marketing = pd.DataFrame({

    "campaign_id": np.arange(
        1,
        N_CAMPAIGNS + 1
    ),

    "campaign_name": [
        f"Vision Campaign {i:03d}"
        for i in range(
            1,
            N_CAMPAIGNS + 1
        )
    ],

    "campaign_date": rng.choice(
        pd.date_range(
            START_DATE,
            END_DATE,
            freq="D"
        ),
        size=N_CAMPAIGNS
    ),

    "channel": rng.choice(
        marketing_channels,
        size=N_CAMPAIGNS
    ),

    "campaign_cost": rng.uniform(
        500,
        15000,
        size=N_CAMPAIGNS
    ).round(2),

    "impressions": rng.integers(
        10000,
        500000,
        size=N_CAMPAIGNS
    ),

    "clicks": rng.integers(
        500,
        30000,
        size=N_CAMPAIGNS
    ),

    "conversions": rng.integers(
        10,
        1500,
        size=N_CAMPAIGNS
    )

})

marketing["attributed_revenue"] = (
    marketing["campaign_cost"]
    * rng.uniform(
        2,
        8,
        size=N_CAMPAIGNS
    )
).round(2)

marketing["roas"] = (
    marketing["attributed_revenue"]
    / marketing["campaign_cost"]
).round(2)

# Marketing attribution is synthetic
marketing["attribution_type"] = (
    "Synthetic Estimate"
)

marketing["is_verified_attribution"] = False

marketing["data_quality"] = (
    "Simulated"
)

marketing["roas_methodology"] = (
    "Synthetic Revenue Attribution"
)

# ============================================================
# 32. FINANCIAL CALENDAR
# ============================================================

if not returns.empty:

    last_return_date = (
        returns["return_date"].max()
    )

else:

    last_return_date = END_DATE

FINANCIAL_END_DATE = max(
    END_DATE,
    pd.Timestamp(last_return_date)
)

financial_months = pd.date_range(
    START_DATE,
    FINANCIAL_END_DATE,
    freq="MS"
)

financial = pd.DataFrame({

    "month": financial_months

})

financial["period"] = (
    financial["month"].dt.to_period("M")
)

# ============================================================
# 33. MONTHLY COMMERCIAL REVENUE
# ============================================================

commercial_sales = sales.loc[
    sales["is_commercial"]
].copy()

monthly_revenue = (
    commercial_sales.groupby(
        commercial_sales["order_date"].dt.to_period("M")
    )["order_revenue"]
    .sum()
)

# ============================================================
# 34. MONTHLY PRODUCT COST
# ============================================================

monthly_cost = (
    commercial_sales.groupby(
        commercial_sales["order_date"].dt.to_period("M")
    )["order_cost"]
    .sum()
)

# ============================================================
# 35. MONTHLY APPROVED REFUNDS
# ============================================================

if not returns.empty:

    approved_returns = returns.loc[
        returns["return_status"] == "Approved"
    ].copy()

    monthly_refunds = (
        approved_returns.groupby(
            approved_returns["return_date"].dt.to_period("M")
        )["refund_value"]
        .sum()
    )

else:

    monthly_refunds = pd.Series(
        dtype=float
    )

# ============================================================
# 36. FACT_FINANCIAL - MONTHLY FINANCIAL DATA
# ============================================================

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

# Commercial revenue after approved refunds
financial["net_revenue_after_returns"] = (
    financial["revenue"]
    - financial["refund_expense"]
).round(2)

# ============================================================
# 37. GROSS PROFIT
# ============================================================

financial["gross_profit"] = (
    financial["revenue"]
    - financial["product_cost"]
).round(2)

financial["gross_profit_after_returns"] = (
    financial["net_revenue_after_returns"]
    - financial["product_cost"]
).round(2)

# ============================================================
# 38. SIMULATED OPERATING EXPENSES
# ============================================================

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
    financial["revenue"]
    * 0.02
).round(2)

# ============================================================
# 39. EBITDA
# ============================================================

financial["ebitda"] = (

    financial["gross_profit_after_returns"]

    - financial["operating_expenses"]

    - financial["administrative_expenses"]

    - financial["commercial_expenses"]

).round(2)

financial["ebitda_margin"] = np.divide(

    financial["ebitda"].to_numpy(),

    financial["net_revenue_after_returns"].to_numpy(),

    out=np.zeros(len(financial)),

    where=financial[
        "net_revenue_after_returns"
    ].to_numpy() != 0

)

# ============================================================
# 40. SIMULATED CASH FLOW
# ============================================================

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

# ============================================================
# 41. FINANCIAL DATA CLASSIFICATION
# ============================================================

financial["data_type"] = (
    "Synthetic Financial Simulation"
)

financial["is_audited"] = False

financial["ebitda_methodology"] = (
    "Simulated Gross Profit Less Operating Expenses"
)

financial["cash_flow_methodology"] = (
    "Simulated Cash Inflow Less Cash Outflow"
)

financial["accounting_reconciliation"] = False

financial["revenue_recognition_method"] = (
    "Order Date"
)

financial["refund_recognition_method"] = (
    "Return Date"
)

financial = financial.drop(
    columns=["period"]
)

# ============================================================
# 42. FACT_TARGETS - MONTHLY TARGETS
# ============================================================

targets = pd.DataFrame({

    "month": pd.date_range(
        START_DATE,
        END_DATE,
        freq="MS"
    )

})

number_of_months = len(
    targets
)

# Uniform distribution of annual-period targets
targets["revenue_target"] = (
    TARGET_REVENUE
    / number_of_months
)

targets["gross_profit_target"] = (
    TARGET_GROSS_PROFIT
    / number_of_months
)

targets["orders_target"] = (
    N_ORDERS
    / number_of_months
)

targets["margin_target"] = (
    TARGET_MARGIN
)

targets["revenue_target_type"] = (
    "Commercial Revenue Before Returns"
)

targets["profit_target_type"] = (
    "Gross Profit Before Returns"
)

targets["target_method"] = (
    "Uniform Monthly Distribution"
)

targets["seasonality_adjusted"] = False

# ============================================================
# 43. BUSINESS RULES DOCUMENTATION
# ============================================================

BUSINESS_RULES = {

    "dataset_nature":
        "Synthetic dataset designed for Executive Business Intelligence demonstrations. It does not represent actual company transactions.",

    "commercial_revenue":
        "Commercial revenue is synthetically calibrated to R$ 8.2 million using proportional monetary allocation in cents.",

    "commercial_cost":
        "Commercial product cost is synthetically calibrated to R$ 6.1 million using proportional monetary allocation in cents.",

    "gross_profit":
        "Gross profit is defined as commercial revenue minus realized commercial product cost, before returns.",

    "gross_margin":
        "Gross margin is calculated as gross profit divided by commercial revenue.",

    "money_allocation":
        "The allocate_money function distributes monetary amounts proportionally and preserves the exact total in cents.",

    "reference_product_cost":
        "dim_products.unit_cost represents the original product reference cost.",

    "realized_product_cost":
        "fact_sales_items.line_cost represents the realized commercial cost recorded for each sales item.",

    "realized_unit_cost":
        "fact_sales_items.unit_cost represents realized unit cost calculated from line_cost divided by quantity.",

    "cost_calibration":
        "Realized product costs are synthetically calibrated and may differ from reference costs in dim_products.",

    "canceled_orders":
        "Canceled orders are excluded from commercial KPIs. Their sales item revenue and commercial costs are zero.",

    "commercial_orders":
        "Commercial KPIs include non-canceled orders.",

    "registered_customers":
        "All customers are included in dim_customers, including customers who have never purchased.",

    "purchasing_customers":
        "A purchasing customer is a registered customer with at least one non-canceled order.",

    "returns_structure":
        "Returns are recorded at order level in fact_returns and at product level in fact_returns_items.",

    "return_rate":
        "Return rate is calculated as returned orders divided by delivered orders.",

    "return_selection":
        "The RETURN_RATE parameter represents the target percentage of delivered orders selected for synthetic returns.",

    "refund_recognition":
        "Approved refunds are recognized using the actual return date, not the original order date.",

    "revenue_recognition":
        "Commercial revenue is recognized using the original order date.",

    "return_cost_treatment":
        "Original commercial product costs are not automatically reversed when a product is returned.",

    "inventory_recovery":
        "The dataset does not simulate the physical recovery, resale or disposal of returned products.",

    "return_dates":
        "Return dates may occur after the end of the original sales period, including January 2026.",

    "monthly_targets":
        "Revenue and gross profit targets are distributed uniformly across all historical months.",

    "target_seasonality":
        "Monthly targets are not adjusted for seasonality or different commercial periods.",

    "target_comparison":
        "Commercial revenue targets must be compared against commercial revenue before refunds.",

    "refund_reporting":
        "Approved refunds must be reported separately from commercial revenue targets.",

    "marketing_attribution":
        "Marketing attributed revenue is synthetically estimated and does not represent verified attribution.",

    "marketing_roas":
        "Marketing ROAS is a synthetic estimate and must not be interpreted as proven campaign performance.",

    "financial_data":
        "Financial expenses, EBITDA and cash flow are synthetic simulations.",

    "financial_audit":
        "Financial records are not audited accounting transactions and are not reconciled with an actual general ledger.",

    "ebitda":
        "EBITDA is simulated using gross profit after refunds less operating, administrative and commercial expenses.",

    "cash_flow":
        "Cash flow is simulated using estimated cash inflows and cash outflows.",

    "inventory_methodology":
        "Inventory records represent synthetic monthly stock snapshots and not actual inventory movements.",

    "inventory_sampling":
        "Approximately 10 percent of all possible product, store and month combinations are sampled.",

    "geographic_relationship":
        "Cities and states are generated from a predefined Brazilian geographic reference table.",

    "currency":
        "All monetary amounts are expressed in Brazilian reais (BRL).",

    "monthly_revenue":
        "Monthly commercial revenue is grouped by order date.",

    "monthly_refunds":
        "Monthly approved refunds are grouped by return date.",

    "historical_period":
        "Commercial history covers January 2021 through December 2025.",

    "financial_period":
        "The financial calendar may extend into January 2026 to include refunds associated with December 2025 orders."

}

business_rules = pd.DataFrame({

    "rule": list(
        BUSINESS_RULES.keys()
    ),

    "description": list(
        BUSINESS_RULES.values()
    )

})

# ============================================================
# 44. DATASET METADATA
# ============================================================

dataset_metadata = pd.DataFrame({

    "parameter": [

        "Company",

        "Dataset Type",

        "Currency",

        "Country",

        "Start Date",

        "End Date",

        "Financial End Date",

        "Registered Customers",

        "Orders",

        "Products",

        "Sellers",

        "Stores",

        "Campaigns",

        "Target Revenue",

        "Target Product Cost",

        "Target Gross Profit",

        "Target Gross Margin",

        "Return Rate Parameter",

        "Cancellation Rate Parameter",

        "Inventory Sample Rate",

        "Historical Period",

        "Commercial Revenue Calibration",

        "Financial Data Classification",

        "Marketing Attribution Classification"

    ],

    "value": [

        COMPANY,

        "Synthetic",

        "BRL",

        "Brazil",

        str(START_DATE.date()),

        str(END_DATE.date()),

        str(FINANCIAL_END_DATE.date()),

        N_CUSTOMERS,

        N_ORDERS,

        N_PRODUCTS,

        N_SELLERS,

        N_STORES,

        N_CAMPAIGNS,

        TARGET_REVENUE,

        TARGET_COST,

        TARGET_GROSS_PROFIT,

        TARGET_MARGIN,

        RETURN_RATE,

        CANCELLATION_RATE,

        INVENTORY_SAMPLE_RATE,

        "5 Years",

        "Artificially Calibrated",

        "Synthetic Financial Simulation",

        "Unverified Synthetic Attribution"

    ]

})

# ============================================================
# 45. COLLECT ALL FACT TABLES
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
# 46. COLLECT ALL DATASET TABLES
# ============================================================

all_tables = {

    **dimensions,

    **facts,

    "business_rules": business_rules

}

# ============================================================
# 47. EXPORT DATASET TO CSV
# ============================================================

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

# ============================================================
# 48. EXPORT DATASET METADATA
# ============================================================

metadata_path = os.path.join(
    RAW_DIR,
    "dataset_metadata.csv"
)

dataset_metadata.to_csv(

    metadata_path,

    index=False,

    encoding="utf-8-sig"

)

# ============================================================
# 49. EXPORT GENERATOR CONFIGURATION
# ============================================================

configuration = pd.DataFrame({

    "parameter": [

        "SEED",

        "N_CUSTOMERS",

        "N_PRODUCTS",

        "N_ORDERS",

        "N_SELLERS",

        "N_STORES",

        "TARGET_REVENUE",

        "TARGET_COST",

        "TARGET_GROSS_PROFIT",

        "TARGET_MARGIN",

        "RETURN_RATE",

        "CANCELLATION_RATE",

        "INVENTORY_SAMPLE_RATE"

    ],

    "value": [

        SEED,

        N_CUSTOMERS,

        N_PRODUCTS,

        N_ORDERS,

        N_SELLERS,

        N_STORES,

        TARGET_REVENUE,

        TARGET_COST,

        TARGET_GROSS_PROFIT,

        TARGET_MARGIN,

        RETURN_RATE,

        CANCELLATION_RATE,

        INVENTORY_SAMPLE_RATE

    ]

})

configuration.to_csv(

    os.path.join(
        RAW_DIR,
        "generator_configuration.csv"
    ),

    index=False,

    encoding="utf-8-sig"

)

# ============================================================
# 50. DATASET GENERATION COMPLETE
# ============================================================

print(
    "VISION GROUP DATASET GENERATION COMPLETED"
)

print(
    f"Output directory: {os.path.abspath(RAW_DIR)}"
)
