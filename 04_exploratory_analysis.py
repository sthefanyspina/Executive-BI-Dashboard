#Receita e lucro geral
import pandas as pd

sales = tables["fact_sales"]

commercial_sales = sales.loc[
    sales["is_commercial"]
].copy()

summary = {

    "total_orders": len(sales),

    "commercial_orders": len(
        commercial_sales
    ),

    "canceled_orders": (
        sales["status"] == "Canceled"
    ).sum(),

    "revenue": commercial_sales[
        "order_revenue"
    ].sum(),

    "cost": commercial_sales[
        "order_cost"
    ].sum(),

    "profit": commercial_sales[
        "order_gross_profit"
    ].sum()

}

summary["margin"] = (
    summary["profit"]
    / summary["revenue"]
)

summary["average_ticket"] = (
    summary["revenue"]
    / summary["commercial_orders"]
)

for key, value in summary.items():

    print(key, value)

#Receita mensal
sales["order_date"] = pd.to_datetime(
    sales["order_date"]
)

commercial_sales = sales.loc[
    sales["is_commercial"]
].copy()

commercial_sales["year_month"] = (
    commercial_sales["order_date"]
    .dt.to_period("M")
)

monthly_sales = (
    commercial_sales
    .groupby("year_month")
    .agg(

        revenue=(
            "order_revenue",
            "sum"
        ),

        profit=(
            "order_gross_profit",
            "sum"
        ),

        orders=(
            "order_id",
            "nunique"
        )

    )
    .reset_index()
)

monthly_sales["margin"] = (
    monthly_sales["profit"]
    / monthly_sales["revenue"]
)

print(monthly_sales.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_sales["year_month"].astype(str),
    monthly_sales["revenue"]
)

plt.title(
    "Vision Group - Monthly Revenue"
)

plt.xlabel("Month")

plt.ylabel("Revenue (BRL)")

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
#Desempenho por região
regional_sales = (
    commercial_sales
    .groupby("region")
    .agg(

        revenue=(
            "order_revenue",
            "sum"
        ),

        profit=(
            "order_gross_profit",
            "sum"
        ),

        orders=(
            "order_id",
            "nunique"
        )

    )
    .reset_index()
)

regional_sales["margin"] = (
    regional_sales["profit"]
    / regional_sales["revenue"]
)

regional_sales["revenue_share"] = (
    regional_sales["revenue"]
    / regional_sales["revenue"].sum()
)

regional_sales = regional_sales.sort_values(
    "revenue",
    ascending=False
)

print(regional_sales)



#Desempenho por categoria
items = tables["fact_sales_items"]

products = tables["dim_products"]

category_analysis = items.merge(
    products[
        [
            "product_id",
            "category_name"
        ]
    ],
    on="product_id",
    how="left"
)

category_analysis = category_analysis.loc[
    category_analysis["is_commercial"]
]

category_result = (
    category_analysis
    .groupby("category_name")
    .agg(

        revenue=(
            "net_revenue",
            "sum"
        ),

        cost=(
            "line_cost",
            "sum"
        ),

        profit=(
            "gross_profit",
            "sum"
        )

    )
    .reset_index()
)

category_result["margin"] = (
    category_result["profit"]
    / category_result["revenue"]
)

category_result = category_result.sort_values(
    "revenue",
    ascending=False
)

print(category_result)


#Análise de clientes
customers = tables["dim_customers"]

customer_analysis = customers.copy()

customer_analysis["revenue_per_order"] = np.where(
    customer_analysis["total_orders"] > 0,
    customer_analysis["customer_net_revenue"]
    / customer_analysis["total_orders"],
    0
)

customer_summary = {

    "registered_customers": len(
        customer_analysis
    ),

    "buyers": (
        customer_analysis["is_buyer"]
    ).sum(),

    "registered_only": (
        customer_analysis["is_buyer"] == False
    ).sum(),

    "average_customer_revenue": (
        customer_analysis.loc[
            customer_analysis["is_buyer"],
            "customer_net_revenue"
        ].mean()
    )

}

print(customer_summary)

#Análise de devoluções
returns = tables["fact_returns"]

delivered = sales.loc[
    sales["status"] == "Delivered"
]

returned_orders = returns[
    "order_id"
].nunique()

return_rate = (
    returned_orders
    / delivered["order_id"].nunique()
)

print(
    "Return Rate:",
    round(return_rate * 100, 2),
    "%"
)

return_reasons = (
    returns
    .groupby("return_reason")
    .agg(

        returns=(
            "return_id",
            "count"
        ),

        refund=(
            "refund_value",
            "sum"
        )

    )
    .reset_index()
)

return_reasons = return_reasons.sort_values(
    "returns",
    ascending=False
)

print(return_reasons)

#Análise de estoque
inventory = tables["fact_inventory"]

inventory_summary = (
    inventory
    .groupby("stock_status")
    .agg(

        records=(
            "product_id",
            "count"
        ),

        total_units=(
            "stock_quantity",
            "sum"
        ),

        stock_value=(
            "stock_value",
            "sum"
        )

    )
    .reset_index()
)

inventory_summary["record_percentage"] = (
    inventory_summary["records"]
    / inventory_summary["records"].sum()
)

print(inventory_summary)
