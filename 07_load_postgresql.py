import os
import pandas as pd

from sqlalchemy import create_engine

# =====================================================
# 1. CONNECTION
# =====================================================

DB_USER = "postgres"

DB_PASSWORD = os.environ["VISION_DB_PASSWORD"]

DB_HOST = "localhost"

DB_PORT = "5432"

DB_NAME = "vision_group_dw"

DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}@"
    f"{DB_HOST}:{DB_PORT}/"
    f"{DB_NAME}"
)

engine = create_engine(
    DATABASE_URL
)

# =====================================================
# 2. TABLE CONFIGURATION
# =====================================================

PROCESSED_DIR = "../data/processed"

table_names = [
    "dim_customers",
    "dim_products",
    "dim_stores",
    "dim_sellers",
    "dim_date",
    "fact_sales",
    "fact_sales_items",
    "fact_returns",
    "fact_returns_items",
    "fact_inventory",
    "fact_marketing",
    "fact_financial",
    "fact_targets"
]

# =====================================================
# 3. LOAD DATA
# =====================================================

for table_name in table_names:

    file_path = os.path.join(
        PROCESSED_DIR,
        f"{table_name}.csv"
    )

    df = pd.read_csv(
        file_path
    )

    df.to_sql(
        name=table_name,
        con=engine,
        schema="staging",
        if_exists="replace",
        index=False,
        chunksize=10000,
        method="multi"
    )

    print(
        f"{table_name}: "
        f"{len(df):,} records loaded"
    )
