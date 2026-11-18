CREATE OR REPLACE VIEW analytics.vw_executive_kpis AS

WITH monthly_sales AS (

    SELECT

        DATE_TRUNC(
            'month',
            order_date
        )::date AS month,

        COUNT(DISTINCT order_id)
            AS commercial_orders,

        COUNT(DISTINCT customer_id)
            AS purchasing_customers,

        SUM(order_revenue)
            AS revenue,

        SUM(order_cost)
            AS product_cost,

        SUM(order_gross_profit)
            AS gross_profit,

        SUM(total_refund)
            AS refunds,

        SUM(net_revenue_after_returns)
            AS net_revenue

    FROM dw.fact_sales

    WHERE is_commercial = TRUE

    GROUP BY 1

),

monthly_targets AS (

    SELECT

        DATE_TRUNC(
            'month',
            month
        )::date AS month,

        SUM(revenue_target)
            AS revenue_target,

        SUM(gross_profit_target)
            AS gross_profit_target,

        SUM(orders_target)
            AS orders_target,

        AVG(margin_target)
            AS margin_target

    FROM dw.fact_targets

    GROUP BY 1

),

combined AS (

    SELECT

        s.*,

        t.revenue_target,

        t.gross_profit_target,

        t.orders_target,

        t.margin_target

    FROM monthly_sales s

    LEFT JOIN monthly_targets t

        ON s.month = t.month

)

SELECT

    month,

    commercial_orders,

    purchasing_customers,

    revenue,

    product_cost,

    gross_profit,

    refunds,

    net_revenue,

    gross_profit / NULLIF(revenue, 0)
        AS gross_margin,

    revenue / NULLIF(commercial_orders, 0)
        AS average_ticket,

    revenue_target,

    revenue - revenue_target
        AS revenue_variance,

    revenue / NULLIF(revenue_target, 0)
        AS revenue_target_achievement,

    gross_profit_target,

    gross_profit / NULLIF(gross_profit_target, 0)
        AS profit_target_achievement,

    orders_target,

    commercial_orders - orders_target
        AS orders_variance

FROM combined;



#View 2 — vw_sales_performance
CREATE OR REPLACE VIEW analytics.vw_sales_performance AS

WITH monthly_region_sales AS (

    SELECT

        DATE_TRUNC(
            'month',
            s.order_date
        )::date AS month,

        s.region,

        s.sales_channel,

        COUNT(DISTINCT s.order_id)
            AS orders,

        COUNT(DISTINCT s.customer_id)
            AS customers,

        SUM(s.order_revenue)
            AS revenue,

        SUM(s.order_gross_profit)
            AS gross_profit

    FROM dw.fact_sales s

    WHERE s.is_commercial = TRUE

    GROUP BY

        1, 2, 3

),

sales_with_growth AS (

    SELECT

        *,

        LAG(revenue, 1) OVER (

            PARTITION BY region, sales_channel

            ORDER BY month

        ) AS previous_month_revenue,

        LAG(revenue, 12) OVER (

            PARTITION BY region, sales_channel

            ORDER BY month

        ) AS previous_year_revenue,

        SUM(revenue) OVER (

            PARTITION BY region, sales_channel

            ORDER BY month

            ROWS BETWEEN UNBOUNDED PRECEDING
            AND CURRENT ROW

        ) AS cumulative_revenue

    FROM monthly_region_sales

)

SELECT

    *,

    (revenue / NULLIF(previous_month_revenue, 0)) - 1
        AS mom_growth,

    (revenue / NULLIF(previous_year_revenue, 0)) - 1
        AS yoy_growth,

    gross_profit / NULLIF(revenue, 0)
        AS gross_margin,

    revenue / NULLIF(orders, 0)
        AS average_ticket

FROM sales_with_growth;




CREATE OR REPLACE VIEW analytics.vw_sales_by_product AS

SELECT

    p.category_name,

    p.subcategory_name,

    p.product_id,

    p.product_name,

    SUM(i.quantity) AS units_sold,

    SUM(i.net_revenue) AS revenue,

    SUM(i.gross_profit) AS gross_profit,

    SUM(i.net_revenue_after_returns)
        AS net_revenue_after_returns,

    COUNT(DISTINCT i.order_id)
        AS orders

FROM dw.fact_sales_items i

JOIN dw.fact_sales s

    ON i.order_id = s.order_id

JOIN dw.dim_products p

    ON i.product_id = p.product_id

WHERE i.is_commercial = TRUE

GROUP BY

    p.category_name,

    p.subcategory_name,

    p.product_id,

    p.product_name;

#View 3 — vw_customer_performance
CREATE OR REPLACE VIEW analytics.vw_customer_performance AS

WITH customer_orders AS (

    SELECT

        customer_id,

        COUNT(DISTINCT order_id)
            AS frequency,

        MIN(order_date)
            AS first_purchase,

        MAX(order_date)
            AS last_purchase,

        SUM(order_revenue)
            AS total_revenue,

        SUM(total_refund)
            AS total_refunds,

        SUM(net_revenue_after_returns)
            AS net_revenue

    FROM dw.fact_sales

    WHERE is_commercial = TRUE

    GROUP BY customer_id

),

customer_base AS (

    SELECT

        c.customer_id,

        c.customer_segment,

        c.acquisition_channel,

        c.region,

        COALESCE(o.frequency, 0)
            AS frequency,

        o.first_purchase,

        o.last_purchase,

        COALESCE(o.total_revenue, 0)
            AS total_revenue,

        COALESCE(o.total_refunds, 0)
            AS total_refunds,

        COALESCE(o.net_revenue, 0)
            AS net_revenue,

        COALESCE(

            CURRENT_DATE - o.last_purchase,

            99999

        ) AS recency_days

    FROM dw.dim_customers c

    LEFT JOIN customer_orders o

        ON c.customer_id = o.customer_id

)

SELECT

    *,

    CASE

        WHEN frequency = 0
            THEN 'Registered Only'

        WHEN recency_days <= 90
            THEN 'Active'

        WHEN recency_days <= 180
            THEN 'At Risk'

        ELSE 'Inactive'

    END AS behavioral_status

FROM customer_base;

#cohort
WITH first_purchase AS (

    SELECT

        customer_id,

        DATE_TRUNC(
            'month',
            MIN(order_date)
        )::date AS cohort_month

    FROM dw.fact_sales

    WHERE is_commercial = TRUE

    GROUP BY customer_id

),

customer_activity AS (

    SELECT DISTINCT

        s.customer_id,

        f.cohort_month,

        DATE_TRUNC(
            'month',
            s.order_date
        )::date AS activity_month,

        (

            EXTRACT(
                YEAR FROM AGE(
                    DATE_TRUNC('month', s.order_date),
                    f.cohort_month
                )
            ) * 12

            +

            EXTRACT(
                MONTH FROM AGE(
                    DATE_TRUNC('month', s.order_date),
                    f.cohort_month
                )
            )

        )::integer AS cohort_index

    FROM dw.fact_sales s

    JOIN first_purchase f

        ON s.customer_id = f.customer_id

    WHERE s.is_commercial = TRUE

)

SELECT

    cohort_month,

    cohort_index,

    COUNT(DISTINCT customer_id)
        AS retained_customers

FROM customer_activity

GROUP BY

    cohort_month,

    cohort_index

ORDER BY

    cohort_month,

    cohort_index;

#View 4 — vw_product_profitability
CREATE OR REPLACE VIEW analytics.vw_product_profitability AS

WITH product_metrics AS (

    SELECT

        p.product_id,

        p.product_name,

        p.category_name,

        p.subcategory_name,

        COUNT(DISTINCT i.order_id)
            AS total_orders,

        SUM(i.quantity)
            AS units_sold,

        SUM(i.net_revenue)
            AS gross_commercial_revenue,

        SUM(i.line_cost)
            AS product_cost,

        SUM(i.discount)
            AS discounts,

        SUM(i.refund_value)
            AS refunds,

        SUM(i.net_revenue_after_returns)
            AS net_revenue,

        SUM(i.gross_profit_after_returns)
            AS gross_profit_after_returns

    FROM dw.fact_sales_items i

    JOIN dw.dim_products p

        ON i.product_id = p.product_id

    WHERE i.is_commercial = TRUE

    GROUP BY

        p.product_id,

        p.product_name,

        p.category_name,

        p.subcategory_name

)

SELECT

    *,

    gross_commercial_revenue
        - product_cost AS gross_profit,

    product_cost / NULLIF(
        gross_commercial_revenue, 0
    ) AS cost_ratio,

    discounts / NULLIF(
        gross_commercial_revenue, 0
    ) AS discount_ratio,

    gross_profit_after_returns
        / NULLIF(net_revenue, 0)
        AS margin_after_returns,

    gross_profit_after_returns
        / NULLIF(net_revenue, 0)
        AS profitability_ratio

FROM product_metrics;



SELECT

    category_name,

    SUM(gross_commercial_revenue)
        AS revenue,

    SUM(product_cost)
        AS cost,

    SUM(gross_profit)
        AS gross_profit,

    SUM(gross_profit)
        / NULLIF(
            SUM(gross_commercial_revenue),
            0
        ) AS margin

FROM analytics.vw_product_profitability

GROUP BY category_name

ORDER BY revenue DESC;

#View 5 — vw_operational_performance
CREATE OR REPLACE VIEW analytics.vw_operational_performance AS

WITH order_metrics AS (

    SELECT

        DATE_TRUNC(
            'month',
            order_date
        )::date AS month,

        COUNT(*) AS total_orders,

        COUNT(*) FILTER (

            WHERE status = 'Canceled'

        ) AS canceled_orders,

        COUNT(*) FILTER (

            WHERE status = 'Delivered'

        ) AS delivered_orders,

        COUNT(*) FILTER (

            WHERE status IN (
                'Processing',
                'Shipped'
            )

        ) AS open_orders,

        COUNT(*) FILTER (

            WHERE status = 'Delivered'

            AND delivered_date <= promised_delivery_date

        ) AS on_time_deliveries,

        AVG(

            delivered_date - order_date

        ) FILTER (

            WHERE status = 'Delivered'

        ) AS average_delivery_days

    FROM dw.fact_sales

    GROUP BY 1

),

return_metrics AS (

    SELECT

        DATE_TRUNC(
            'month',
            return_date
        )::date AS month,

        COUNT(DISTINCT order_id)
            AS returned_orders,

        SUM(refund_value)
            AS refund_value

    FROM dw.fact_returns

    GROUP BY 1

),

inventory_metrics AS (

    SELECT

        DATE_TRUNC(
            'month',
            inventory_date
        )::date AS month,

        COUNT(*) AS inventory_records,

        COUNT(*) FILTER (

            WHERE stock_status = 'Stockout'

        ) AS stockout_records,

        COUNT(*) FILTER (

            WHERE stock_status = 'Low Stock'

        ) AS low_stock_records,

        SUM(stock_value)
            AS stock_value

    FROM dw.fact_inventory

    GROUP BY 1

)

SELECT

    o.month,

    o.total_orders,

    o.canceled_orders,

    o.delivered_orders,

    o.open_orders,

    o.on_time_deliveries,

    o.average_delivery_days,

    o.canceled_orders::numeric
        / NULLIF(o.total_orders, 0)
        AS cancellation_rate,

    o.on_time_deliveries::numeric
        / NULLIF(o.delivered_orders, 0)
        AS on_time_sla,

    COALESCE(r.returned_orders, 0)
        AS returned_orders,

    COALESCE(r.refund_value, 0)
        AS refund_value,

    COALESCE(i.inventory_records, 0)
        AS inventory_records,

    COALESCE(i.stockout_records, 0)
        AS stockout_records,

    COALESCE(i.low_stock_records, 0)
        AS low_stock_records,

    COALESCE(i.stock_value, 0)
        AS stock_value,

    i.stockout_records::numeric
        / NULLIF(i.inventory_records, 0)
        AS stockout_record_rate

FROM order_metrics o

LEFT JOIN return_metrics r

    ON o.month = r.month

LEFT JOIN inventory_metrics i

    ON o.month = i.month;


#View 6 — vw_financial_performance
CREATE OR REPLACE VIEW analytics.vw_financial_performance AS

WITH financial_base AS (

    SELECT

        DATE_TRUNC(
            'month',
            month
        )::date AS month,

        revenue,

        refund_expense,

        net_revenue_after_returns,

        product_cost,

        gross_profit_after_returns,

        operating_expenses,

        administrative_expenses,

        commercial_expenses,

        depreciation_amortization,

        ebitda,

        cash_inflow,

        cash_outflow,

        cash_flow

    FROM dw.fact_financial

),

financial_comparison AS (

    SELECT

        *,

        LAG(net_revenue_after_returns)
            OVER (ORDER BY month)
            AS previous_month_revenue,

        LAG(ebitda)
            OVER (ORDER BY month)
            AS previous_month_ebitda,

        SUM(cash_flow)
            OVER (
                ORDER BY month
                ROWS BETWEEN UNBOUNDED PRECEDING
                AND CURRENT ROW
            ) AS cumulative_cash_flow

    FROM financial_base

)

SELECT

    *,

    net_revenue_after_returns
        - product_cost
        AS gross_profit_recalculated,

    ebitda / NULLIF(
        net_revenue_after_returns, 0
    ) AS ebitda_margin,

    cash_flow / NULLIF(
        cash_inflow, 0
    ) AS cash_flow_ratio,

    net_revenue_after_returns
        - previous_month_revenue
        AS revenue_variance_month,

    ebitda
        - previous_month_ebitda
        AS ebitda_variance_month

FROM financial_comparison;







#CTE — Top 10 produtos por receita
WITH product_revenue AS (

    SELECT

        product_id,

        product_name,

        category_name,

        SUM(gross_commercial_revenue)
            AS revenue

    FROM analytics.vw_product_profitability

    GROUP BY

        product_id,

        product_name,

        category_name

),

ranked_products AS (

    SELECT

        *,

        DENSE_RANK() OVER (

            ORDER BY revenue DESC

        ) AS revenue_rank

    FROM product_revenue

)

SELECT *

FROM ranked_products

WHERE revenue_rank <= 10

ORDER BY revenue_rank;


#Window Function — Ranking regional
SELECT

    region,

    SUM(revenue) AS revenue,

    RANK() OVER (

        ORDER BY SUM(revenue) DESC

    ) AS regional_rank,

    SUM(revenue) * 100.0 /

        SUM(SUM(revenue)) OVER ()

        AS revenue_percentage

FROM analytics.vw_sales_performance

GROUP BY region

ORDER BY regional_rank;


#Subquery — Produtos abaixo da margem média
SELECT *

FROM analytics.vw_product_profitability

WHERE margin_after_returns < (

    SELECT

        SUM(gross_profit_after_returns)
        / NULLIF(SUM(net_revenue), 0)

    FROM analytics.vw_product_profitability

)

ORDER BY margin_after_returns;


#Comparação entre receita e meta
SELECT

    month,

    revenue,

    revenue_target,

    revenue - revenue_target
        AS variance,

    revenue / NULLIF(revenue_target, 0)
        AS achievement,

    CASE

        WHEN revenue >= revenue_target
            THEN 'Target Achieved'

        ELSE 'Below Target'

    END AS performance_status

FROM analytics.vw_executive_kpis

ORDER BY month;

#Análise de evolução do EBITDA
SELECT

    month,

    ebitda,

    ebitda_margin,

    previous_month_ebitda,

    ebitda_variance_month,

    cumulative_cash_flow,

    SUM(ebitda) OVER (

        ORDER BY month

    ) AS cumulative_ebitda

FROM analytics.vw_financial_performance

ORDER BY month;


#Identificar clientes de maior valor
SELECT

    customer_id,

    frequency,

    total_revenue,

    total_refunds,

    net_revenue,

    behavioral_status

FROM analytics.vw_customer_performance

WHERE frequency >= 3

ORDER BY net_revenue DESC

LIMIT 100;

