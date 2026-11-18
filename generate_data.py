import os
import numpy as np
import pandas as pd

# ============================================================
# 1. CONFIGURACOES GERAIS
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

COMPANY = "Vision Group"

OUTPUT_DIR = "vision_group_dataset"

RAW_DIR = os.path.join(
    OUTPUT_DIR,
    "raw"
)

os.makedirs(RAW_DIR, exist_ok=True)

# Quantidades da empresa

N_CUSTOMERS = 82_400
N_ORDERS = 145_200
N_PRODUCTS = 500
N_SELLERS = 120
N_STORES = 35

# Metas financeiras

TARGET_REVENUE = 8_200_000.00
TARGET_PROFIT = 2_100_000.00

TARGET_COST = (
    TARGET_REVENUE - TARGET_PROFIT
)

# Periodo historico

START_DATE = pd.Timestamp("2021-01-01")
END_DATE = pd.Timestamp("2025-12-31")

# ============================================================
# 2. REGIOES BRASILEIRAS
# ============================================================

STATE_REGION = {

    "AC": "Norte",
    "AP": "Norte",
    "AM": "Norte",
    "PA": "Norte",
    "RO": "Norte",
    "RR": "Norte",
    "TO": "Norte",

    "AL": "Nordeste",
    "BA": "Nordeste",
    "CE": "Nordeste",
    "MA": "Nordeste",
    "PB": "Nordeste",
    "PE": "Nordeste",
    "PI": "Nordeste",
    "RN": "Nordeste",
    "SE": "Nordeste",

    "DF": "Centro-Oeste",
    "GO": "Centro-Oeste",
    "MT": "Centro-Oeste",
    "MS": "Centro-Oeste",

    "ES": "Sudeste",
    "MG": "Sudeste",
    "RJ": "Sudeste",
    "SP": "Sudeste",

    "PR": "Sul",
    "RS": "Sul",
    "SC": "Sul"
}

STATES = list(STATE_REGION.keys())

# ============================================================
# 3. DIM_REGIONS
# ============================================================

regions = pd.DataFrame({

    "region_id": range(1, 6),

    "region_name": [
        "Norte",
        "Nordeste",
        "Centro-Oeste",
        "Sudeste",
        "Sul"
    ]

})

# ============================================================
# 4. DIM_CUSTOMERS
# ============================================================

customer_ids = np.arange(
    1,
    N_CUSTOMERS + 1
)

# Cadastro anterior ao inicio do historico
signup_dates = (
    START_DATE
    - pd.to_timedelta(
        rng.integers(
            0,
            365 * 4,
            N_CUSTOMERS
        ),
        unit="D"
    )
)

customers = pd.DataFrame({

    "customer_id": customer_ids,

    "customer_code": [
        f"VG-CUST-{i:07d}"
        for i in customer_ids
    ],

    "customer_name": [
        f"Cliente Vision {i:07d}"
        for i in customer_ids
    ],

    "state": rng.choice(
        STATES,
        N_CUSTOMERS
    ),

    "city": rng.choice(
        [
            "Sao Paulo",
            "Rio de Janeiro",
            "Belo Horizonte",
            "Curitiba",
            "Salvador",
            "Recife",
            "Fortaleza",
            "Brasilia",
            "Goiania",
            "Porto Alegre",
            "Manaus",
            "Belem",
            "Florianopolis",
            "Natal",
            "Campinas"
        ],
        N_CUSTOMERS
    ),

    "signup_date": signup_dates,

    "customer_segment": rng.choice(
        [
            "Novo",
            "Regular",
            "Fiel",
            "VIP"
        ],
        N_CUSTOMERS,
        p=[0.30, 0.40, 0.22, 0.08]
    ),

    "acquisition_channel": rng.choice(
        [
            "Organico",
            "Midia Paga",
            "Redes Sociais",
            "Email Marketing",
            "Indicacao",
            "Direto"
        ],
        N_CUSTOMERS,
        p=[0.25, 0.20, 0.18, 0.12, 0.10, 0.15]
    )

})

customers["region"] = (
    customers["state"].map(STATE_REGION)
)

customers["region_id"] = (
    customers["region"].map({
        "Norte": 1,
        "Nordeste": 2,
        "Centro-Oeste": 3,
        "Sudeste": 4,
        "Sul": 5
    })
)

# ============================================================
# 5. DIM_CATEGORIES
# ============================================================

CATEGORIES = {

    "Eletronicos": [
        "Smartphones",
        "Notebooks",
        "Fones",
        "Acessorios",
        "Cameras"
    ],

    "Casa": [
        "Moveis",
        "Cozinha",
        "Decoracao",
        "Cama e Banho",
        "Organizacao"
    ],

    "Moda": [
        "Roupas",
        "Calcados",
        "Bolsas",
        "Relogios",
        "Acessorios"
    ],

    "Esportes": [
        "Fitness",
        "Camping",
        "Ciclismo",
        "Esportes",
        "Outdoor"
    ],

    "Beleza": [
        "Skincare",
        "Maquiagem",
        "Cabelos",
        "Perfumes",
        "Cuidados Pessoais"
    ],

    "Livros": [
        "Literatura",
        "Negocios",
        "Tecnologia",
        "Educacao",
        "Infantil"
    ]
}

categories = pd.DataFrame({

    "category_id": range(
        1,
        len(CATEGORIES) + 1
    ),

    "category_name": list(
        CATEGORIES.keys()
    )

})

# ============================================================
# 6. DIM_PRODUCTS
# ============================================================

product_ids = np.arange(
    1,
    N_PRODUCTS + 1
)

product_category_names = rng.choice(
    list(CATEGORIES.keys()),
    N_PRODUCTS
)

products = pd.DataFrame({

    "product_id": product_ids,

    "product_code": [
        f"VG-PROD-{i:05d}"
        for i in product_ids
    ],

    "product_name": [
        f"Produto Vision {i:04d}"
        for i in product_ids
    ],

    "category_name": product_category_names,

    "subcategory": [
        rng.choice(CATEGORIES[category])
        for category in product_category_names
    ],

    "brand": rng.choice(
        [
            "Vision",
            "Premium",
            "TechPro",
            "HomeStyle",
            "SportMax",
            "Basic"
        ],
        N_PRODUCTS
    ),

    "unit_cost": rng.uniform(
        15,
        300,
        N_PRODUCTS
    ).round(2)

})

products = products.merge(
    categories,
    on="category_name",
    how="left"
)

# ============================================================
# 7. DIM_SELLERS
# ============================================================

seller_ids = np.arange(
    1,
    N_SELLERS + 1
)

sellers = pd.DataFrame({

    "seller_id": seller_ids,

    "seller_name": [
        f"Vendedor {i:03d}"
        for i in seller_ids
    ],

    "commission_rate": rng.uniform(
        0.01,
        0.06,
        N_SELLERS
    ).round(4),

    "seller_status": rng.choice(
        ["Ativo", "Inativo"],
        N_SELLERS,
        p=[0.90, 0.10]
    )

})

# ============================================================
# 8. DIM_STORES
# ============================================================

store_ids = np.arange(
    1,
    N_STORES + 1
)

stores = pd.DataFrame({

    "store_id": store_ids,

    "store_name": [
        f"Loja Vision {i:02d}"
        for i in store_ids
    ],

    "state": rng.choice(
        STATES,
        N_STORES
    ),

    "store_type": rng.choice(
        [
            "Loja Fisica",
            "Centro de Distribuicao"
        ],
        N_STORES,
        p=[0.85, 0.15]
    )

})

stores["region"] = (
    stores["state"].map(STATE_REGION)
)

stores["region_id"] = (
    stores["region"].map({
        "Norte": 1,
        "Nordeste": 2,
        "Centro-Oeste": 3,
        "Sudeste": 4,
        "Sul": 5
    })
)

# ============================================================
# 9. DIM_DATE
# ============================================================

date_range = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

dim_date = pd.DataFrame({

    "date": date_range

})

dim_date["date_id"] = (
    dim_date["date"]
    .dt.strftime("%Y%m%d")
    .astype(int)
)

dim_date["year"] = (
    dim_date["date"].dt.year
)

dim_date["month"] = (
    dim_date["date"].dt.month
)

dim_date["day"] = (
    dim_date["date"].dt.day
)

dim_date["quarter"] = (
    dim_date["date"].dt.quarter
)

dim_date["month_name"] = (
    dim_date["date"].dt.month_name()
)

dim_date["year_month"] = (
    dim_date["date"].dt.strftime("%Y-%m")
)

dim_date["day_of_week"] = (
    dim_date["date"].dt.day_name()
)

dim_date["is_weekend"] = (
    dim_date["date"].dt.dayofweek >= 5
)

# ============================================================
# 10. DICIONARIO DAS DIMENSOES
# ============================================================

dimensions = {

    "dim_customers": customers,

    "dim_products": products,

    "dim_categories": categories,

    "dim_sellers": sellers,

    "dim_stores": stores,

    "dim_regions": regions,

    "dim_date": dim_date

}


# ============================================================
# 11. FACT_SALES - CRIACAO DOS PEDIDOS
# ============================================================

order_ids = np.arange(
    1,
    N_ORDERS + 1
)

# Gerar datas dentro do historico
random_days = rng.integers(
    0,
    (END_DATE - START_DATE).days + 1,
    N_ORDERS
)

random_order_dates = (
    START_DATE
    + pd.to_timedelta(random_days, unit="D")
)

# Distribuir clientes
order_customer_ids = rng.integers(
    1,
    N_CUSTOMERS + 1,
    N_ORDERS
)

# Garantir que a compra aconteca depois do cadastro
signup_map = customers.set_index(
    "customer_id"
)["signup_date"]

assigned_signup_dates = (
    pd.Series(order_customer_ids)
    .map(signup_map)
    .reset_index(drop=True)
)

order_dates = pd.concat(
    [
        pd.Series(random_order_dates),
        assigned_signup_dates
    ],
    axis=1
).max(axis=1)

order_dates = pd.to_datetime(order_dates)

# Definir status dos pedidos
order_statuses = rng.choice(
    [
        "Entregue",
        "Enviado",
        "Processando",
        "Cancelado"
    ],
    N_ORDERS,
    p=[0.82, 0.08, 0.06, 0.04]
)

# Criar datas de entrega
delivery_days = rng.integers(
    2,
    15,
    N_ORDERS
)

delivered_dates = (
    order_dates
    + pd.to_timedelta(delivery_days, unit="D")
)

delivered_dates = pd.Series(
    delivered_dates
)

delivered_dates.loc[
    order_statuses != "Entregue"
] = pd.NaT

# Datas prometidas
promised_delivery_dates = (
    order_dates
    + pd.to_timedelta(
        rng.integers(3, 11, N_ORDERS),
        unit="D"
    )
)

# Datas de cancelamento
canceled_dates = pd.Series(
    pd.NaT,
    index=np.arange(N_ORDERS),
    dtype="datetime64[ns]"
)

canceled_mask = (
    order_statuses == "Cancelado"
)

canceled_dates.loc[canceled_mask] = (
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

# Criar tabela de pedidos
sales = pd.DataFrame({

    "order_id": order_ids,

    "customer_id": order_customer_ids,

    "seller_id": rng.integers(
        1,
        N_SELLERS + 1,
        N_ORDERS
    ),

    "store_id": rng.integers(
        1,
        N_STORES + 1,
        N_ORDERS
    ),

    "order_date": order_dates,

    "promised_delivery_date": promised_delivery_dates,

    "delivered_date": delivered_dates,

    "canceled_at": canceled_dates,

    "status": order_statuses,

    "sales_channel": rng.choice(
        [
            "E-commerce",
            "Aplicativo",
            "Marketplace",
            "Loja Fisica"
        ],
        N_ORDERS,
        p=[0.35, 0.25, 0.20, 0.20]
    ),

    "shipping_cost": rng.uniform(
        8,
        45,
        N_ORDERS
    ).round(2)

})

# ============================================================
# 12. FACT_SALES_ITEMS - CRIACAO DOS ITENS
# ============================================================

# Quantidade de itens por pedido
item_counts = rng.choice(
    [1, 2, 3, 4],
    N_ORDERS,
    p=[0.20, 0.45, 0.25, 0.10]
)

# Repetir ID do pedido conforme quantidade de itens
item_order_ids = np.repeat(
    order_ids,
    item_counts
)

N_ITEMS = len(item_order_ids)

# Criar identificadores dos itens
item_ids = np.arange(
    1,
    N_ITEMS + 1
)

# Quantidades
quantities = rng.integers(
    1,
    5,
    N_ITEMS
)

# Produtos vendidos
item_product_ids = rng.integers(
    1,
    N_PRODUCTS + 1,
    N_ITEMS
)

# Precos originais
unit_prices = rng.uniform(
    30,
    350,
    N_ITEMS
)

# Descontos
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

# Valor bruto
gross_revenue = (
    quantities * unit_prices
)

# Valor do desconto
discount_value = (
    gross_revenue * discount_rates
)

# ============================================================
# 13. IDENTIFICAR PEDIDOS CANCELADOS
# ============================================================

item_status_map = sales.set_index(
    "order_id"
)["status"]

item_status = (
    pd.Series(item_order_ids)
    .map(item_status_map)
    .to_numpy()
)

item_canceled_mask = (
    item_status == "Cancelado"
)

item_eligible_mask = (
    ~item_canceled_mask
)

# ============================================================
# 14. CALIBRACAO DA RECEITA
# ============================================================

# Receita antes da calibracao
initial_net_revenue = (
    gross_revenue - discount_value
)

# Considerar somente itens de pedidos nao cancelados
eligible_revenue = initial_net_revenue[
    item_eligible_mask
].sum()

# Fator de ajuste para atingir R$ 8,2 milhoes
revenue_factor = (
    TARGET_REVENUE / eligible_revenue
)

# Ajustar preco unitario
unit_prices = (
    unit_prices * revenue_factor
)

# Recalcular valores comerciais
gross_revenue = (
    quantities * unit_prices
).round(2)

discount_value = (
    gross_revenue * discount_rates
).round(2)

net_revenue = (
    gross_revenue - discount_value
).round(2)

# Corrigir diferenca de arredondamento
revenue_difference = round(
    TARGET_REVENUE
    - net_revenue[item_eligible_mask].sum(),
    2
)

# Ajustar o ultimo item elegivel
last_eligible_index = np.flatnonzero(
    item_eligible_mask
)[-1]

gross_revenue[last_eligible_index] += (
    revenue_difference
)

net_revenue[last_eligible_index] += (
    revenue_difference
)

# ============================================================
# 15. CALIBRACAO DOS CUSTOS
# ============================================================

product_cost_map = products.set_index(
    "product_id"
)["unit_cost"]

original_unit_costs = (
    pd.Series(item_product_ids)
    .map(product_cost_map)
    .to_numpy()
)

initial_line_cost = (
    quantities * original_unit_costs
)

# Considerar somente pedidos nao cancelados
eligible_cost = initial_line_cost[
    item_eligible_mask
].sum()

# Fator para atingir o custo desejado
cost_factor = (
    TARGET_COST / eligible_cost
)

# Ajustar custo unitario
adjusted_unit_costs = (
    original_unit_costs * cost_factor
)

# Recalcular custo por item
line_cost = (
    quantities * adjusted_unit_costs
).round(2)

# Corrigir diferenca de arredondamento
cost_difference = round(
    TARGET_COST
    - line_cost[item_eligible_mask].sum(),
    2
)

line_cost[last_eligible_index] += (
    cost_difference
)

# ============================================================
# 16. ZERAR VALORES DOS PEDIDOS CANCELADOS
# ============================================================

gross_revenue[item_canceled_mask] = 0.00

discount_value[item_canceled_mask] = 0.00

net_revenue[item_canceled_mask] = 0.00

line_cost[item_canceled_mask] = 0.00

# Lucro bruto por item
gross_profit = (
    net_revenue - line_cost
).round(2)

# Margem por item
gross_margin = np.divide(
    gross_profit,
    net_revenue,
    out=np.zeros(N_ITEMS),
    where=net_revenue != 0
)

# ============================================================
# 17. CRIAR TABELA FACT_SALES_ITEMS
# ============================================================

items = pd.DataFrame({

    "item_id": item_ids,

    "order_id": item_order_ids,

    "product_id": item_product_ids,

    "quantity": quantities,

    "unit_price": np.round(
        unit_prices,
        2
    ),

    "unit_cost": np.round(
        adjusted_unit_costs,
        2
    ),

    "discount_rate": discount_rates,

    "gross_revenue": gross_revenue,

    "discount_value": discount_value,

    "net_revenue": net_revenue,

    "line_cost": line_cost,

    "gross_profit": gross_profit,

    "gross_margin": gross_margin,

    "promotion_id": np.where(
        discount_rates > 0,
        rng.choice(
            [
                "PROMO-001",
                "PROMO-002",
                "PROMO-003",
                "PROMO-004",
                "PROMO-005"
            ],
            N_ITEMS
        ),
        None
    ),

    "status": item_status

})

# ============================================================
# 18. CONSOLIDAR VALORES POR PEDIDO
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

        order_profit=(
            "gross_profit",
            "sum"
        )

    )
    .reset_index()
)

# Associar os valores a fact_sales
sales = sales.merge(
    order_totals,
    on="order_id",
    how="left"
)

# ============================================================
# 19. CAMPOS FINANCEIROS DOS PEDIDOS
# ============================================================

sales["order_margin"] = np.divide(
    sales["order_profit"],
    sales["order_revenue"],
    out=np.zeros(len(sales)),
    where=sales["order_revenue"].to_numpy() != 0
)

# ============================================================
# 20. RESULTADO DA PARTE 2
# ============================================================

# As tabelas comerciais estao prontas.
# A receita foi distribuida somente entre pedidos
# nao cancelados.

facts_commercial = {

    "fact_sales": sales,

    "fact_sales_items": items

}


# ============================================================
# 21. FACT_RETURNS E FACT_RETURNS_ITEMS
# ============================================================

# Considerar somente pedidos entregues
delivered_orders = sales.loc[
    sales["status"] == "Entregue",
    "order_id"
].to_numpy()

# Percentual de pedidos devolvidos
RETURN_RATE = 0.08

N_RETURNS = int(
    len(delivered_orders) * RETURN_RATE
)

# Selecionar pedidos que terao devolucao
returned_order_ids = rng.choice(
    delivered_orders,
    size=N_RETURNS,
    replace=False
)

# Motivos de devolucao
RETURN_REASONS = [
    "Produto danificado",
    "Produto incorreto",
    "Insatisfacao do cliente",
    "Atraso na entrega",
    "Defeito do produto"
]

# Criar identificadores
return_ids = np.arange(
    1,
    N_RETURNS + 1
)

return_records = []

return_item_records = []

return_item_id = 1

# Criar devolucoes por pedido e produto
for return_id, order_id in zip(
    return_ids,
    returned_order_ids
):

    # Itens pertencentes ao pedido
    order_items = items.loc[
        items["order_id"] == order_id
    ].copy()

    # Quantidade de produtos que serao devolvidos
    number_of_returned_products = rng.integers(
        1,
        len(order_items) + 1
    )

    selected_items = order_items.sample(
        n=number_of_returned_products,
        random_state=int(
            rng.integers(0, 2**31 - 1)
        )
    )

    # Data da entrega
    delivery_date = sales.loc[
        sales["order_id"] == order_id,
        "delivered_date"
    ].iloc[0]

    # Data da devolucao
    return_date = (
        delivery_date
        + pd.Timedelta(
            days=int(rng.integers(1, 31))
        )
    )

    # Motivo
    reason = rng.choice(
        RETURN_REASONS,
        p=[0.25, 0.15, 0.25, 0.15, 0.20]
    )

    # Status
    return_status = rng.choice(
        ["Aprovada", "Em processamento"],
        p=[0.85, 0.15]
    )

    total_refund = 0.00

    # Criar os itens da devolucao
    for _, item in selected_items.iterrows():

        purchased_quantity = int(
            item["quantity"]
        )

        # Quantidade devolvida
        returned_quantity = int(
            rng.integers(
                1,
                purchased_quantity + 1
            )
        )

        # Valor proporcional do produto
        original_item_value = (
            float(item["net_revenue"])
            * returned_quantity
            / purchased_quantity
        )

        # Reembolso integral ou parcial
        refund_percentage = float(
            rng.uniform(0.70, 1.00)
        )

        if return_status == "Aprovada":

            refund_value = round(
                original_item_value
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

            "original_item_revenue": round(
                original_item_value,
                2
            ),

            "refund_percentage": round(
                refund_percentage,
                4
            ),

            "refund_value": refund_value

        })

        return_item_id += 1

    # Criar registro principal da devolucao
    return_records.append({

        "return_id": return_id,

        "order_id": order_id,

        "return_date": return_date,

        "return_reason": reason,

        "return_status": return_status,

        "refund_value": round(
            total_refund,
            2
        )

    })

# Tabela principal de devolucoes
returns = pd.DataFrame(
    return_records
)

# Tabela de itens devolvidos
returns_items = pd.DataFrame(
    return_item_records
)

# ============================================================
# 22. FACT_INVENTORY - ESTOQUE
# ============================================================

# Periodicidade mensal
inventory_months = pd.date_range(
    START_DATE,
    END_DATE,
    freq="MS"
)

# Criar combinacoes de mes, produto e loja
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

# Quantidade em estoque
inventory["stock_quantity"] = rng.integers(
    0,
    301,
    len(inventory)
)

# Ponto de reposicao
inventory["reorder_point"] = rng.integers(
    10,
    50,
    len(inventory)
)

# Custo unitario dos produtos
inventory["unit_cost"] = (
    inventory["product_id"]
    .map(
        products.set_index(
            "product_id"
        )["unit_cost"]
    )
)

# Valor total do estoque
inventory["stock_value"] = (
    inventory["stock_quantity"]
    * inventory["unit_cost"]
).round(2)

# Status do estoque
inventory["stock_status"] = np.select(
    [
        inventory["stock_quantity"] == 0,

        inventory["stock_quantity"]
        <= inventory["reorder_point"]
    ],
    [
        "Ruptura",
        "Estoque Baixo"
    ],
    default="Normal"
)

# ============================================================
# 23. FACT_MARKETING - CAMPANHAS
# ============================================================

N_CAMPAIGNS = 300

marketing = pd.DataFrame({

    "campaign_id": np.arange(
        1,
        N_CAMPAIGNS + 1
    ),

    "campaign_name": [
        f"Campanha Vision {i:03d}"
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
            "Afiliados",
            "Influenciadores"
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

# Receita atribuida as campanhas
marketing["attributed_revenue"] = (
    marketing["campaign_cost"]
    * rng.uniform(
        2,
        8,
        N_CAMPAIGNS
    )
).round(2)

# Retorno sobre investimento em anuncios
marketing["roas"] = (
    marketing["attributed_revenue"]
    / marketing["campaign_cost"]
).round(2)

# ============================================================
# 24. FACT_FINANCIAL - FINANCEIRO
# ============================================================

# Receita mensal a partir dos pedidos
monthly_revenue = (
    sales.groupby(
        sales["order_date"].dt.to_period("M")
    )["order_revenue"]
    .sum()
)

financial = monthly_revenue.rename(
    "revenue"
).reset_index()

financial["month"] = (
    financial["order_date"]
    .dt.to_timestamp()
)

financial = financial.drop(
    columns="order_date"
)

# Despesas operacionais
financial["operating_expenses"] = (
    financial["revenue"]
    * rng.uniform(
        0.08,
        0.18,
        len(financial)
    )
).round(2)

# Despesas administrativas
financial["administrative_expenses"] = (
    financial["revenue"]
    * rng.uniform(
        0.02,
        0.06,
        len(financial)
    )
).round(2)

# Despesas comerciais
financial["commercial_expenses"] = (
    financial["revenue"]
    * rng.uniform(
        0.02,
        0.05,
        len(financial)
    )
).round(2)

# Depreciacao e amortizacao
financial["depreciation_amortization"] = (
    financial["revenue"] * 0.02
).round(2)

# EBITDA
financial["ebitda"] = (
    financial["revenue"]
    - financial["operating_expenses"]
    - financial["administrative_expenses"]
    - financial["commercial_expenses"]
)

# Entradas financeiras
financial["cash_inflow"] = (
    financial["revenue"]
    * rng.uniform(
        0.90,
        1.00,
        len(financial)
    )
).round(2)

# Saidas financeiras
financial["cash_outflow"] = (
    financial["operating_expenses"]
    + financial["administrative_expenses"]
    + financial["commercial_expenses"]
    + financial["depreciation_amortization"]
).round(2)

# Fluxo de caixa
financial["cash_flow"] = (
    financial["cash_inflow"]
    - financial["cash_outflow"]
).round(2)

# ============================================================
# 25. FACT_TARGETS - METAS
# ============================================================

targets = financial[["month"]].copy()

# Metas mensais iniciais
targets["revenue_target"] = (
    TARGET_REVENUE / len(targets)
)

targets["profit_target"] = (
    TARGET_PROFIT / len(targets)
)

targets["orders_target"] = (
    N_ORDERS / len(targets)
)

targets["margin_target"] = (
    TARGET_PROFIT / TARGET_REVENUE
)

# ============================================================
# 26. ORGANIZAR TODAS AS TABELAS
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
# 27. EXPORTAR DIMENSOES E FATOS PARA CSV
# ============================================================

all_tables = {

    **dimensions,

    **facts

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

# ============================================================
# 28. FINALIZACAO
# ============================================================

print("\n" + "=" * 60)

print("VISION GROUP - DATASET GERADO")

print("=" * 60)

print(f"Empresa: {COMPANY}")

print(f"Clientes: {N_CUSTOMERS:,}")

print(f"Pedidos: {N_ORDERS:,}")

print(f"Produtos: {N_PRODUCTS:,}")

print(f"Vendedores: {N_SELLERS:,}")

print(f"Lojas: {N_STORES:,}")

print(f"Receita comercial definida: R$ {TARGET_REVENUE:,.2f}")

print(f"Lucro bruto definido: R$ {TARGET_PROFIT:,.2f}")

print(f"Margem bruta definida: {TARGET_PROFIT / TARGET_REVENUE:.2%}")

print(f"Pedidos com devolucao: {N_RETURNS:,}")

print(f"Itens devolvidos: {len(returns_items):,}")

print("\nTABELAS GERADAS")

for name, dataframe in all_tables.items():

    print(
        f"{name:<25} {len(dataframe):>12,} registros"
    )

print("\nArquivos salvos em:")

print(os.path.abspath(RAW_DIR))

print("\nGERACAO CONCLUIDA!")
