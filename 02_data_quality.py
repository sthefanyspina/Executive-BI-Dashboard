import os
import pandas as pd
import numpy as np

RAW_DIR = "../data/raw"

# =====================================================
# 1. LOAD DATA
# =====================================================

tables = {}

table_names = [
    "dim_customers",
    "dim_products",
    "dim_categories",
    "dim_sellers",
    "dim_stores",
    "dim_regions",
    "dim_date",
    "dim_geography",
    "fact_sales",
    "fact_sales_items",
    "fact_returns",
    "fact_returns_items",
    "fact_inventory",
    "fact_marketing",
    "fact_financial",
    "fact_targets",
    "business_rules"
]

for table_name in table_names:

    file_path = os.path.join(
        RAW_DIR,
        f"{table_name}.csv"
    )

    tables[table_name] = pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )

    print(
        f"{table_name}: "
        f"{len(tables[table_name]):,} registros"
    )

def data_quality_report(df, table_name):

    report = pd.DataFrame({

        "column": df.columns,

        "data_type": [
            str(df[col].dtype)
            for col in df.columns
        ],

        "null_count": [
            df[col].isna().sum()
            for col in df.columns
        ],

        "null_percentage": [
            df[col].isna().mean() * 100
            for col in df.columns
        ],

        "unique_values": [
            df[col].nunique()
            for col in df.columns
        ]

    })

    report["table"] = table_name

    return report

quality_reports = []

for table_name, df in tables.items():

    report = data_quality_report(
        df,
        table_name
    )

    quality_reports.append(report)

quality_report = pd.concat(
    quality_reports,
    ignore_index=True
)

quality_report.to_csv(
    "../data/analysis/data_quality_report.csv",
    index=False
)

print(quality_report)

primary_keys = {

    "dim_customers": "customer_id",

    "dim_products": "product_id",

    "dim_sellers": "seller_id",

    "dim_stores": "store_id",

    "fact_sales": "order_id",

    "fact_sales_items": "item_id",

    "fact_returns": "return_id",

    "fact_returns_items": "return_item_id"

}

for table_name, key in primary_keys.items():

    df = tables[table_name]

    duplicates = df[key].duplicated().sum()

    print(
        f"{table_name}: "
        f"{duplicates} duplicidades"
    )

def validate_foreign_key(
    fact_df,
    fact_column,
    dimension_df,
    dimension_column
):

    invalid = ~fact_df[fact_column].isin(
        dimension_df[dimension_column]
    )

    return invalid.sum()

sales = tables["fact_sales"]

customers = tables["dim_customers"]

sellers = tables["dim_sellers"]

stores = tables["dim_stores"]

print(
    "Clientes inválidos:",
    validate_foreign_key(
        sales,
        "customer_id",
        customers,
        "customer_id"
    )
)

print(
    "Vendedores inválidos:",
    validate_foreign_key(
        sales,
        "seller_id",
        sellers,
        "seller_id"
    )
)

print(
    "Lojas inválidas:",
    validate_foreign_key(
        sales,
        "store_id",
        stores,
        "store_id"
    )
)

items = tables["fact_sales_items"]

items["calculated_profit"] = (
    items["net_revenue"]
    - items["line_cost"]
).round(2)

items["profit_difference"] = (
    items["gross_profit"]
    - items["calculated_profit"]
).round(2)

invalid_profit = items.loc[
    items["profit_difference"].abs() > 0.01
]

print(
    "Itens com lucro inconsistente:",
    len(invalid_profit)
)

commercial_items = items.loc[
    items["is_commercial"]
]

total_revenue = commercial_items[
    "net_revenue"
].sum()

total_cost = commercial_items[
    "line_cost"
].sum()

total_profit = commercial_items[
    "gross_profit"
].sum()

margin = (
    total_profit / total_revenue
)

print("Receita:", total_revenue)

print("Custo:", total_cost)

print("Lucro:", total_profit)

print("Margem:", margin * 100)
