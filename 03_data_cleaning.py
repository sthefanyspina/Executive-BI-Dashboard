def standardize_text(df):

    df = df.copy()

    for column in df.select_dtypes(
        include="object"
    ).columns:

        df[column] = (
            df[column]
            .str.strip()
        )

    return df

for table_name in tables:

    tables[table_name] = standardize_text(
        tables[table_name]
    )

date_columns = {

    "dim_customers": [
        "signup_date",
        "first_purchase_date",
        "last_purchase_date"
    ],

    "fact_sales": [
        "order_date",
        "promised_delivery_date",
        "delivered_date",
        "canceled_at"
    ],

    "fact_returns": [
        "return_date"
    ],

    "fact_inventory": [
        "inventory_date"
    ],

    "fact_marketing": [
        "campaign_date"
    ],

    "fact_financial": [
        "month"
    ],

    "fact_targets": [
        "month"
    ]

}

for table_name, columns in date_columns.items():

    df = tables[table_name]

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    tables[table_name] = df



numeric_columns = {

    "fact_sales": [
        "order_revenue",
        "order_cost",
        "order_gross_profit",
        "total_refund",
        "shipping_cost"
    ],

    "fact_sales_items": [
        "quantity",
        "price",
        "discount",
        "net_revenue",
        "line_cost",
        "gross_profit"
    ],

    "fact_financial": [
        "revenue",
        "product_cost",
        "operating_expenses",
        "administrative_expenses",
        "commercial_expenses",
        "ebitda",
        "cash_flow"
    ]

}

for table_name, columns in numeric_columns.items():

    df = tables[table_name]

    for column in columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    tables[table_name] = df

PROCESSED_DIR = "../data/processed"

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)

for table_name, df in tables.items():

    df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            f"{table_name}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )
