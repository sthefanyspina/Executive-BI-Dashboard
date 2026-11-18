def calculate_executive_kpis(
    sales,
    targets
):

    commercial = sales.loc[
        sales["is_commercial"]
    ].copy()

    revenue = commercial[
        "order_revenue"
    ].sum()

    cost = commercial[
        "order_cost"
    ].sum()

    profit = revenue - cost

    orders = commercial[
        "order_id"
    ].nunique()

    customers = commercial[
        "customer_id"
    ].nunique()

    ticket = revenue / orders if orders else 0

    margin = revenue and profit / revenue

    target_revenue = targets[
        "revenue_target"
    ].sum()

    achievement = (
        revenue / target_revenue
        if target_revenue
        else 0
    )

    return {

        "revenue": revenue,

        "cost": cost,

        "profit": profit,

        "margin": margin,

        "orders": orders,

        "customers": customers,

        "average_ticket": ticket,

        "revenue_target": target_revenue,

        "target_achievement": achievement

    }

kpis = calculate_executive_kpis(
    sales,
    tables["fact_targets"]
)

for indicator, value in kpis.items():

    print(
        indicator,
        value
    )


monthly_sales = monthly_sales.sort_values(
    "year_month"
)

monthly_sales["revenue_previous_month"] = (
    monthly_sales["revenue"].shift(1)
)

monthly_sales["revenue_mom"] = (
    monthly_sales["revenue"]
    / monthly_sales["revenue_previous_month"]
    - 1
)

monthly_sales["revenue_previous_year"] = (
    monthly_sales["revenue"].shift(12)
)

monthly_sales["revenue_yoy"] = (
    monthly_sales["revenue"]
    / monthly_sales["revenue_previous_year"]
    - 1
)

#Construção do RFM
import pandas as pd

sales = tables["fact_sales"]

commercial = sales.loc[
    sales["is_commercial"]
].copy()

reference_date = (
    pd.Timestamp("2025-12-31")
)

rfm = (
    commercial
    .groupby("customer_id")
    .agg(

        last_purchase=(
            "order_date",
            "max"
        ),

        frequency=(
            "order_id",
            "nunique"
        ),

        monetary=(
            "net_revenue_after_returns",
            "sum"
        )

    )
    .reset_index()
)

rfm["recency"] = (
    reference_date
    - pd.to_datetime(rfm["last_purchase"])
).dt.days

#Criar pontuação RFM
rfm["R_score"] = pd.qcut(
    rfm["recency"].rank(
        method="first",
        ascending=False
    ),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_score"] = pd.qcut(
    rfm["frequency"].rank(
        method="first"
    ),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_score"] = pd.qcut(
    rfm["monetary"].rank(
        method="first"
    ),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["RFM_score"] = (
    rfm["R_score"].astype(str)
    + rfm["F_score"].astype(str)
    + rfm["M_score"].astype(str)
)

#Criar segmentos
def classify_customer(row):

    if (
        row["R_score"] >= 4
        and row["F_score"] >= 4
        and row["M_score"] >= 4
    ):
        return "Champions"

    elif (
        row["R_score"] >= 3
        and row["F_score"] >= 3
    ):
        return "Loyal Customers"

    elif row["R_score"] >= 4:

        return "Recent Customers"

    elif row["R_score"] <= 2:

        return "At Risk"

    else:

        return "Needs Attention"


rfm["customer_segment"] = rfm.apply(
    classify_customer,
    axis=1
)

#
cohort_data = commercial.copy()

cohort_data["order_date"] = pd.to_datetime(
    cohort_data["order_date"]
)

cohort_data["order_month"] = (
    cohort_data["order_date"]
    .dt.to_period("M")
)

cohort_data["cohort_month"] = (
    cohort_data
    .groupby("customer_id")["order_date"]
    .transform("min")
    .dt.to_period("M")
)

cohort_data["cohort_index"] = (

    (
        cohort_data["order_month"].dt.year
        - cohort_data["cohort_month"].dt.year
    ) * 12

    +

    (
        cohort_data["order_month"].dt.month
        - cohort_data["cohort_month"].dt.month
    )

)

cohort_table = (
    cohort_data
    .groupby(
        [
            "cohort_month",
            "cohort_index"
        ]
    )["customer_id"]
    .nunique()
    .unstack()
)

cohort_size = cohort_table.iloc[:, 0]

cohort_retention = (
    cohort_table
    .div(cohort_size, axis=0)
)

print(cohort_retention)

#Lifetime Value — LTV
ltv = (
    commercial
    .groupby("customer_id")
    .agg(

        total_orders=(
            "order_id",
            "nunique"
        ),

        total_revenue=(
            "net_revenue_after_returns",
            "sum"
        )

    )
    .reset_index()
)

ltv["average_order_value"] = (
    ltv["total_revenue"]
    / ltv["total_orders"]
)



#Etapa 6: identificar comportamentos fora do padrão
#Desvio padrão
monthly_sales["revenue_mean"] = (
    monthly_sales["revenue"].mean()
)

monthly_sales["revenue_std"] = (
    monthly_sales["revenue"].std()
)

monthly_sales["z_score"] = (

    monthly_sales["revenue"]
    - monthly_sales["revenue_mean"]

) / monthly_sales["revenue_std"]

monthly_sales["anomaly"] = (
    monthly_sales["z_score"].abs() > 2
)

print(
    monthly_sales.loc[
        monthly_sales["anomaly"]
    ]
)

#Isolation Forest
from sklearn.ensemble import IsolationForest

features = monthly_sales[
    [
        "revenue",
        "profit",
        "orders"
    ]
].copy()

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

monthly_sales["anomaly_model"] = (
    model.fit_predict(features)
)

monthly_sales["anomaly_model"] = (
    monthly_sales["anomaly_model"] == -1
)


