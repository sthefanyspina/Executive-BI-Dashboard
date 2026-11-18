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

Entrar
Cadastre-se grátis
arrume o script python abaixo para que atenda aos pedidos para fazer o projeto abaixo. Caso o codigo fique muito grande, divida o em 3 partes: me explique detalhadamente e com os codigos como fazer o projeto: 10. 🏢 Executive BI Dashboard
Cenário empresarial
Crie a empresa Vision Group, uma rede varejista fictícia.
A empresa possui:
 
•	82.400 clientes.
•	145.200 pedidos.
•	R$ 8,2 milhões em receita.
•	R$ 2,1 milhões em lucro.
•	25,6% de margem.
•	5 anos de histórico.
•	Vendas em todo o Brasil.
 
A diretoria deseja responder:
Como está o negócio e quais áreas precisam de atenção?

Base de dados
Você pode criar um banco de dados consolidado com:
 
dim_customers	Clientes
dim_products	Produtos
dim_categories	Categorias
dim_sellers	Vendedores
dim_stores	Lojas
dim_regions	Regiões
dim_date	Calendário
fact_sales	Vendas
fact_sales_items	Itens vendidos
fact_returns	Devoluções
fact_inventory	Estoque
fact_marketing	Campanhas
fact_financial	Receitas e despesas
fact_targets	Metas
 
Esse projeto pode reaproveitar conceitos e dados dos projetos anteriores, mas deve possuir uma base consolidada e coerente, com regras de negócio documentadas.

Python
Faça:
 
•	Integração das bases.
•	Limpeza dos dados.
•	Validação de consistência.
•	Análise exploratória.
•	Criação de indicadores.
•	Comparação de resultados com metas.
•	Identificação de anomalias.
•	Análise de tendências.
 
SQL
Crie uma camada analítica com views específicas:
 
•	vw_executive_kpis
•	vw_sales_performance
•	vw_customer_performance
•	vw_product_profitability
•	vw_operational_performance
•	vw_financial_performance
 

Inclua consultas com:
 
•	CTEs.
•	Window Functions.
•	Subqueries.
•	Views.
•	Agregações.
•	Comparações temporais.
•	Indicadores acumulados.
 

Dashboard executivo
Página 1 — Executive Overview
Indicadores:
 
•	Receita.
•	Lucro.
•	Margem.
•	Clientes.
•	Pedidos.
•	Ticket médio.
•	Crescimento.
•	Meta versus realizado.
 
Pergunta respondida: como está o negócio?

Página 2 - Sales Performance
Indicadores:
•	Receita por mês.
•	Receita por região.
•	Vendas por categoria.
•	Vendas por produto.
•	Ticket médio.
•	Crescimento MoM e YoY.
Pergunta respondida: como estão as vendas?

Página 3 — Customer Analytics
Indicadores:
 
•	Clientes novos.
•	Clientes recorrentes.
•	Retenção.
•	Churn.
•	Cohort.
•	RFM.
•	LTV.
 
Pergunta respondida: como está nossa base de clientes?

Página 4 — Profitability
Indicadores:
 
•	Receita.
•	Custos.
•	Lucro.
•	Margem.
•	Margem por produto.
•	Margem por categoria.
 
Pergunta respondida: estamos vendendo com rentabilidade?

Página 5 — Operations
Indicadores:
 
•	Pedidos.
•	Cancelamentos.
•	Devoluções.
•	SLA.
•	Estoque.
•	Rupturas.
•	Performance operacional.
 
Pergunta respondida: nossa operação está funcionando adequadamente?

Página 6 — Financial Performance
Indicadores:
 
•	Receita.
•	Despesas.
•	EBITDA.
•	Cash Flow.
•	Budget vs Actual.
•	Desvios orçamentários.
 
Pergunta respondida: como está a saúde financeira?

Análise executiva
Além do dashboard, escreva um relatório com cinco partes:
 
1.	Situação geral da empresa.
2.	Principais resultados positivos.
3.	Principais problemas identificados.
4.	Oportunidades de melhoria.
5.	Recomendações para a diretoria.
 

Exemplo de conclusão:
"A empresa apresentou crescimento de receita no período analisado, mas houve redução da margem de determinadas categorias. A análise indica que descontos e alterações no mix de produtos estão associados à redução da rentabilidade. Recomenda-se investigar as políticas comerciais e acompanhar a margem por categoria nas próximas avaliações."

Entrega final
Esse projeto deve parecer um trabalho desenvolvido por um analista dentro de uma empresa, com documentação técnica, dashboards e apresentação executiva.


Usando o script: 
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

Projeto 10 — Executive BI Dashboard | Vision Group
Seu script atual é uma boa base para gerar dados fictícios de um varejo, mas precisa de algumas mudanças importantes para atender ao projeto de Business Intelligence Executivo.

Minha recomendação é transformar o código em um pipeline completo de dados, com três responsabilidades:

Gerar uma base fictícia coerente com a realidade da Vision Group.

Integrar, limpar, validar e analisar os dados usando Python.

Preparar as tabelas para análise em SQL e construção do dashboard executivo no Power BI.

Vou organizar a solução em três partes, com códigos que podem ser utilizados em sequência.

1. O que precisa ser corrigido no script original
Existem algumas diferenças entre os dados que o seu script gera e os requisitos do projeto.

Requisito

Script original

Como ficará

Clientes

100.000

82.400

Pedidos

Calculado automaticamente

145.200

Receita

Aleatória

R$ 8.200.000

Lucro

Não controlado

R$ 2.100.000

Margem

Não controlada

Aproximadamente 25,6%

Histórico

2010–2025

2021–2025

Geografia

Estados americanos

27 UFs brasileiras

Produtos

5.000

500

Vendedores

Não possui

Incluídos

Lojas

Não possui

Incluídas

Categorias

Misturadas na dimensão produto

Dimensão própria

Calendário

Não possui

Incluído

Estoque

Não possui

Incluído

Marketing

Não possui

Incluído

Financeiro

Pagamentos simplificados

Incluído

Metas

Não possui

Incluídas

Também precisamos corrigir alguns problemas de modelagem:

Pedidos devolvidos: o código altera o status de Delivered para Returned, perdendo a informação de que o pedido foi originalmente entregue. Vamos preservar o status operacional e registrar a devolução separadamente.

Receita: os valores são aleatórios e não garantem os R$ 8,2 milhões desejados.

Lucro: o custo dos produtos não está calibrado para produzir os R$ 2,1 milhões.

Geografia: o projeto exige cobertura nacional brasileira.

Granularidade: precisamos separar pedidos, itens, devoluções e movimentações financeiras.

Indicadores: precisamos criar dados que permitam calcular retenção, churn, RFM, LTV, EBITDA, fluxo de caixa e desempenho operacional.

2. Arquitetura que vamos construir
A Vision Group terá uma arquitetura dimensional, com dimensões e tabelas fato.

Arquitetura de dados — Vision Group
FONTES DE DADOS

Clientes • Produtos • Pedidos • Lojas • Vendedores • Marketing • Financeiro

PYTHON — ETL E QUALIDADE

Geração • Integração • Limpeza • Validação • Análise exploratória • KPIs

BANCO DE DADOS SQL

Modelo dimensional • Tabelas fato • Dimensões • Views analíticas

POWER BI — EXECUTIVE DASHBOARD

6 páginas • Indicadores • Tendências • Metas • Alertas

RELATÓRIO EXECUTIVO

Diagnóstico • Oportunidades • Riscos • Recomendações à diretoria

Modelo de dados
Teremos as seguintes tabelas:

Dimensões

Tabelas fato

dim_customers

fact_sales

dim_products

fact_sales_items

dim_categories

fact_returns

dim_sellers

fact_inventory

dim_stores

fact_marketing

dim_regions

fact_financial

dim_date

fact_targets

Decisão importante: os R$ 8,2 milhões serão tratados como receita líquida comercial antes das devoluções, e os R$ 2,1 milhões como lucro bruto dos produtos antes das despesas operacionais. Isso produz uma margem bruta de aproximadamente 25,6%.

Para não confundir os indicadores:

Receita bruta: vendas antes dos descontos.

Receita líquida comercial: vendas após descontos, antes das devoluções.

Receita líquida ajustada: receita comercial menos devoluções.

Lucro bruto: receita líquida comercial menos custo dos produtos.

EBITDA: lucro operacional antes de juros, impostos, depreciação e amortização.

Essas definições serão documentadas para evitar divergências entre Python, SQL e Power BI.

PARTE 1/3 — Configuração e criação das dimensões
Nesta primeira parte vamos criar a estrutura da empresa e suas dimensões.

1.1. Preparação do ambiente
Instale as bibliotecas:


bash
pip install pandas numpy sqlalchemy matplotlib seaborn
Crie a seguinte estrutura de pastas:


text
VISION_GROUP/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── analytics/
│
├── reports/
│
├── sql/
│
├── src/
│   ├── 01_generate_data.py
│   ├── 02_data_quality.py
│   └── 03_analysis.py
│
└── main.py
Para facilitar a execução, inicialmente vamos colocar as três partes em um único arquivo chamado main.py. Depois você poderá separar os módulos.

1.2. Código — configuração e dimensões
Crie o arquivo main.py e insira:


python

# ============================================================
# VISION GROUP - EXECUTIVE BI DASHBOARD
# PARTE 1/3 - CONFIGURACAO E DIMENSOES
# ============================================================

import os
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

# ------------------------------------------------------------
# 1. CONFIGURACOES
# ------------------------------------------------------------

SEED = 42
rng = np.random.default_rng(SEED)

OUTPUT_DIR = "data"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed")
ANALYTICS_DIR = os.path.join(OUTPUT_DIR, "analytics")

for folder in [RAW_DIR, PROCESSED_DIR, ANALYTICS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Parametros empresariais

COMPANY = "Vision Group"

N_CUSTOMERS = 82_400
N_ORDERS = 145_200
N_PRODUCTS = 500
N_SELLERS = 120
N_STORES = 35

TARGET_REVENUE = 8_200_000.00
TARGET_PROFIT = 2_100_000.00

START_DATE = pd.Timestamp("2021-01-01")
END_DATE = pd.Timestamp("2025-12-31")

# ------------------------------------------------------------
# 2. GEOGRAFIA BRASILEIRA
# ------------------------------------------------------------

STATE_REGION = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte",
    "PA": "Norte", "RO": "Norte", "RR": "Norte",
    "TO": "Norte",

    "AL": "Nordeste", "BA": "Nordeste",
    "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste",
    "PI": "Nordeste", "RN": "Nordeste",
    "SE": "Nordeste",

    "DF": "Centro-Oeste", "GO": "Centro-Oeste",
    "MT": "Centro-Oeste", "MS": "Centro-Oeste",

    "ES": "Sudeste", "MG": "Sudeste",
    "RJ": "Sudeste", "SP": "Sudeste",

    "PR": "Sul", "RS": "Sul", "SC": "Sul"
}

STATES = list(STATE_REGION.keys())

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

regions.to_csv(
    f"{RAW_DIR}/dim_regions.csv",
    index=False
)

# ------------------------------------------------------------
# 3. DIM_CUSTOMERS
# ------------------------------------------------------------

customer_ids = np.arange(1, N_CUSTOMERS + 1)

customer_signup_dates = (
    START_DATE
    - pd.to_timedelta(
        rng.integers(0, 365 * 4, N_CUSTOMERS),
        unit="D"
    )
)

customers = pd.DataFrame({
    "customer_id": customer_ids,

    "customer_code": [
        f"VG-CUST-{i:07d}"
        for i in customer_ids
    ],

    "state": rng.choice(STATES, N_CUSTOMERS),

    "city": rng.choice([
        "Sao Paulo", "Rio de Janeiro",
        "Belo Horizonte", "Curitiba",
        "Salvador", "Recife", "Fortaleza",
        "Brasilia", "Goiania", "Porto Alegre",
        "Manaus", "Belem", "Florianopolis",
        "Natal", "Campinas"
    ], N_CUSTOMERS),

    "signup_date": customer_signup_dates,

    "customer_segment": rng.choice(
        ["Novo", "Regular", "Fiel", "VIP"],
        N_CUSTOMERS,
        p=[0.30, 0.40, 0.22, 0.08]
    ),

    "acquisition_channel": rng.choice(
        [
            "Organico",
            "Midia Paga",
            "Redes Sociais",
            "Email",
            "Indicacao",
            "Direto"
        ],
        N_CUSTOMERS
    )
})

customers["region"] = customers["state"].map(
    STATE_REGION
)

# ------------------------------------------------------------
# 4. DIM_CATEGORIES
# ------------------------------------------------------------

categories_dict = {
    "Eletronicos": [
        "Smartphones", "Notebooks",
        "Fones", "Acessorios"
    ],

    "Casa": [
        "Moveis", "Cozinha",
        "Decoracao", "Cama e Banho"
    ],

    "Moda": [
        "Roupas", "Calcados",
        "Bolsas", "Relogios"
    ],

    "Esportes": [
        "Fitness", "Camping",
        "Ciclismo", "Esportes"
    ],

    "Beleza": [
        "Skincare", "Maquiagem",
        "Cabelos", "Perfumes"
    ],

    "Livros": [
        "Literatura", "Negocios",
        "Tecnologia", "Educacao"
    ]
}

categories = pd.DataFrame({
    "category_id": range(1, len(categories_dict) + 1),
    "category_name": list(categories_dict.keys())
})

# ------------------------------------------------------------
# 5. DIM_PRODUCTS
# ------------------------------------------------------------

product_ids = np.arange(1, N_PRODUCTS + 1)

category_names = list(categories_dict.keys())

product_categories = rng.choice(
    category_names,
    N_PRODUCTS
)

products = pd.DataFrame({
    "product_id": product_ids,

    "product_name": [
        f"Produto Vision {i:04d}"
        for i in product_ids
    ],

    "category_name": product_categories,

    "subcategory": [
        rng.choice(categories_dict[c])
        for c in product_categories
    ],

    "brand": rng.choice(
        ["Vision", "Premium", "TechPro",
         "HomeStyle", "SportMax", "Basic"],
        N_PRODUCTS
    ),

    "unit_cost": rng.uniform(
        15, 300, N_PRODUCTS
    ).round(2)
})

products = products.merge(
    categories,
    on="category_name",
    how="left"
)

# ------------------------------------------------------------
# 6. DIM_SELLERS
# ------------------------------------------------------------

sellers = pd.DataFrame({
    "seller_id": np.arange(1, N_SELLERS + 1),

    "seller_name": [
        f"Vendedor {i:03d}"
        for i in range(1, N_SELLERS + 1)
    ],

    "commission_rate": rng.uniform(
        0.01, 0.06, N_SELLERS
    ).round(4),

    "seller_status": "Ativo"
})

# ------------------------------------------------------------
# 7. DIM_STORES
# ------------------------------------------------------------

store_ids = np.arange(1, N_STORES + 1)

stores = pd.DataFrame({
    "store_id": store_ids,

    "store_name": [
        f"Loja Vision {i:02d}"
        for i in store_ids
    ],

    "state": rng.choice(STATES, N_STORES),

    "store_type": rng.choice(
        ["Fisica", "Centro de Distribuicao"],
        N_STORES,
        p=[0.85, 0.15]
    )
})

stores["region"] = stores["state"].map(
    STATE_REGION
)

# ------------------------------------------------------------
# 8. DIM_DATE
# ------------------------------------------------------------

date_range = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

dim_date = pd.DataFrame({
    "date": date_range
})

dim_date["date_id"] = (
    dim_date["date"].dt.strftime("%Y%m%d").astype(int)
)

dim_date["year"] = dim_date["date"].dt.year
dim_date["month"] = dim_date["date"].dt.month
dim_date["day"] = dim_date["date"].dt.day
dim_date["quarter"] = dim_date["date"].dt.quarter

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

# ------------------------------------------------------------
# 9. VERIFICACAO DAS DIMENSOES
# ------------------------------------------------------------

dimensions = {
    "dim_customers": customers,
    "dim_products": products,
    "dim_categories": categories,
    "dim_sellers": sellers,
    "dim_stores": stores,
    "dim_regions": regions,
    "dim_date": dim_date
}

print("\nDIMENSOES CRIADAS")

for name, df in dimensions.items():
    print(f"{name}: {len(df):,} registros")
O que construímos?
Agora temos as sete dimensões do modelo.

Alguns detalhes importantes:

dim_customers: permite analisar comportamento, segmentação e aquisição.

dim_products: permite analisar rentabilidade por produto.

dim_categories: permite comparar categorias.

dim_sellers: permite acompanhar vendedores.

dim_stores: permite analisar lojas físicas e centros de distribuição.

dim_regions: permite analisar o desempenho das cinco regiões brasileiras.

dim_date: será utilizada para análises temporais, MoM, YoY e acumulados.

Observação: o código utiliza nomes fictícios de cidades e vendedores. O objetivo é produzir uma base sintética para análise, não representar cadastros reais.

PARTE 2/3 — Criação das tabelas fato
Agora vamos criar as tabelas que representam as operações da Vision Group.

Teremos sete tabelas:

fact_sales

fact_sales_items

fact_returns

fact_inventory

fact_marketing

fact_financial

fact_targets

2.1. Vendas e itens vendidos
A lógica será:

Criar exatamente 145.200 pedidos.

Distribuir os pedidos ao longo de cinco anos.

Associar cada pedido a um cliente, vendedor e loja.

Criar entre um e quatro itens por pedido.

Ajustar os valores comerciais para alcançar R$ 8,2 milhões de receita.

Ajustar os custos para alcançar R$ 2,1 milhões de lucro bruto.

Adicione o código abaixo ao final da primeira parte:


python

# ============================================================
# VISION GROUP
# PARTE 2/3 - TABELAS FATO
# ============================================================

# ------------------------------------------------------------
# 10. FACT_SALES - PEDIDOS
# ------------------------------------------------------------

order_ids = np.arange(1, N_ORDERS + 1)

order_dates = pd.to_datetime(
    rng.integers(
        START_DATE.value // 10**9,
        END_DATE.value // 10**9,
        N_ORDERS
    ),
    unit="s"
).normalize()

order_dates = pd.Series(order_dates)

# Selecionar clientes
order_customers = rng.integers(
    1, N_CUSTOMERS + 1, N_ORDERS
)

# Garantir que pedidos ocorram depois do cadastro
signup_map = customers.set_index(
    "customer_id"
)["signup_date"]

signup_dates = order_customers.copy()

signup_dates = pd.Series(
    order_customers
).map(signup_map).reset_index(drop=True)

order_dates = pd.concat(
    [order_dates.reset_index(drop=True),
     signup_dates],
    axis=1
).max(axis=1)

# Garantir que todas as datas estejam dentro do histórico
order_dates = order_dates.clip(
    lower=START_DATE,
    upper=END_DATE
)

# Associar loja, vendedor e canal
sales = pd.DataFrame({
    "order_id": order_ids,

    "customer_id": order_customers,

    "seller_id": rng.integers(
        1, N_SELLERS + 1, N_ORDERS
    ),

    "store_id": rng.integers(
        1, N_STORES + 1, N_ORDERS
    ),

    "order_date": order_dates,

    "sales_channel": rng.choice(
        ["E-commerce", "Aplicativo",
         "Marketplace", "Loja Física"],
        N_ORDERS,
        p=[0.35, 0.25, 0.20, 0.20]
    ),

    "status": rng.choice(
        ["Entregue", "Enviado",
         "Processando", "Cancelado"],
        N_ORDERS,
        p=[0.82, 0.08, 0.06, 0.04]
    )
})

sales["promised_delivery_date"] = (
    sales["order_date"]
    + pd.to_timedelta(
        rng.integers(3, 11, N_ORDERS),
        unit="D"
    )
)

delivery_days = rng.integers(
    2, 15, N_ORDERS
)

sales["delivered_date"] = (
    sales["order_date"]
    + pd.to_timedelta(delivery_days, unit="D")
)

sales.loc[
    sales["status"] != "Entregue",
    "delivered_date"
] = pd.NaT

sales["canceled_at"] = pd.NaT

sales.loc[
    sales["status"] == "Cancelado",
    "canceled_at"
] = (
    sales.loc[
        sales["status"] == "Cancelado",
        "order_date"
    ]
    + pd.to_timedelta(1, unit="D")
)

sales["shipping_cost"] = rng.uniform(
    8, 45, N_ORDERS
).round(2)

# ------------------------------------------------------------
# 11. FACT_SALES_ITEMS
# ------------------------------------------------------------

item_counts = rng.choice(
    [1, 2, 3, 4],
    N_ORDERS,
    p=[0.20, 0.45, 0.25, 0.10]
)

item_order_ids = np.repeat(
    order_ids,
    item_counts
)

n_items = len(item_order_ids)

items = pd.DataFrame({
    "order_id": item_order_ids,

    "product_id": rng.integers(
        1, N_PRODUCTS + 1, n_items
    ),

    "quantity": rng.integers(
        1, 4, n_items
    ),

    "unit_price": rng.uniform(
        30, 250, n_items
    ).round(2),

    "discount_rate": rng.choice(
        [0, 0.05, 0.10, 0.15, 0.20],
        n_items,
        p=[0.45, 0.20, 0.15, 0.12, 0.08]
    )
})

items["gross_revenue"] = (
    items["quantity"] * items["unit_price"]
).round(2)

items["discount_value"] = (
    items["gross_revenue"]
    * items["discount_rate"]
).round(2)

items["net_revenue"] = (
    items["gross_revenue"]
    - items["discount_value"]
).round(2)

# Ajustar receita para o valor corporativo desejado
revenue_factor = (
    TARGET_REVENUE / items["net_revenue"].sum()
)

items["unit_price"] = (
    items["unit_price"] * revenue_factor
).round(4)

items["gross_revenue"] = (
    items["quantity"] * items["unit_price"]
).round(2)

items["discount_value"] = (
    items["gross_revenue"]
    * items["discount_rate"]
).round(2)

items["net_revenue"] = (
    items["gross_revenue"]
    - items["discount_value"]
).round(2)

# Corrigir pequena diferenca de arredondamento
difference = round(
    TARGET_REVENUE - items["net_revenue"].sum(),
    2
)

items.loc[items.index[-1], "net_revenue"] += difference

# ------------------------------------------------------------
# 12. CUSTOS E LUCRO
# ------------------------------------------------------------

items = items.merge(
    products[["product_id", "unit_cost"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

# Calcular custos proporcionais para atingir o lucro alvo
target_cost = TARGET_REVENUE - TARGET_PROFIT

raw_cost = (
    items["quantity"] * items["unit_cost"]
)

cost_factor = target_cost / raw_cost.sum()

items["line_cost"] = (
    raw_cost * cost_factor
).round(2)

cost_difference = round(
    target_cost - items["line_cost"].sum(),
    2
)

items.loc[
    items.index[-1], "line_cost"
] += cost_difference

# Lucro por item
items["gross_profit"] = (
    items["net_revenue"] - items["line_cost"]
).round(2)

items["gross_margin"] = (
    items["gross_profit"]
    / items["net_revenue"].replace(0, np.nan)
)

items["gross_margin"] = (
    items["gross_margin"].fillna(0)
)

# ------------------------------------------------------------
# 13. ASSOCIAR ITENS AOS PEDIDOS
# ------------------------------------------------------------

items["item_id"] = np.arange(
    1, len(items) + 1
)

# ------------------------------------------------------------
# 14. ATUALIZAR VALORES DOS PEDIDOS
# ------------------------------------------------------------

order_totals = (
    items.groupby("order_id")
    .agg(
        order_revenue=("net_revenue", "sum"),
        order_cost=("line_cost", "sum"),
        order_profit=("gross_profit", "sum")
    )
    .reset_index()
)

sales = sales.merge(
    order_totals,
    on="order_id",
    how="left",
    validate="one_to_one"
)

# Cancelamentos não geram receita realizada
canceled_mask = sales["status"] == "Cancelado"

sales.loc[
    canceled_mask,
    ["order_revenue", "order_cost", "order_profit"]
] = 0

items = items.merge(
    sales[["order_id", "status"]],
    on="order_id",
    how="left",
    validate="many_to_one"
)

# ------------------------------------------------------------
# 15. FACT_RETURNS
# ------------------------------------------------------------

delivered_orders = sales.loc[
    sales["status"] == "Entregue",
    "order_id"
].to_numpy()

n_returns = int(len(delivered_orders) * 0.08)

returned_orders = rng.choice(
    delivered_orders,
    n_returns,
    replace=False
)

returns = pd.DataFrame({
    "return_id": np.arange(1, n_returns + 1),

    "order_id": returned_orders,

    "return_date": pd.NaT,

    "return_reason": rng.choice(
        [
            "Produto danificado",
            "Produto incorreto",
            "Insatisfacao",
            "Atraso",
            "Defeito"
        ],
        n_returns
    ),

    "return_status": rng.choice(
        ["Aprovada", "Em processamento"],
        n_returns,
        p=[0.85, 0.15]
    )
})

delivery_map = sales.set_index(
    "order_id"
)["delivered_date"]

returns["return_date"] = (
    returns["order_id"]
    .map(delivery_map)
    + pd.to_timedelta(
        rng.integers(1, 31, n_returns),
        unit="D"
    )
)

order_revenue_map = sales.set_index(
    "order_id"
)["order_revenue"]

returns["refund_value"] = (
    returns["order_id"]
    .map(order_revenue_map)
    * rng.uniform(0.20, 0.80, n_returns)
).round(2)

# ------------------------------------------------------------
# 16. FACT_INVENTORY
# ------------------------------------------------------------

months = pd.date_range(
    START_DATE,
    END_DATE,
    freq="MS"
)

inventory = pd.MultiIndex.from_product(
    [months, product_ids, store_ids],
    names=["inventory_date", "product_id", "store_id"]
).to_frame(index=False)

inventory["stock_quantity"] = rng.integers(
    0, 300, len(inventory)
)

inventory["stock_value"] = (
    inventory["stock_quantity"]
    * inventory["product_id"].map(
        products.set_index("product_id")["unit_cost"]
    )
).round(2)

inventory["reorder_point"] = rng.integers(
    10, 40, len(inventory)
)

inventory["stock_status"] = np.where(
    inventory["stock_quantity"] == 0,
    "Ruptura",
    np.where(
        inventory["stock_quantity"]
        <= inventory["reorder_point"],
        "Estoque Baixo",
        "Normal"
    )
)

# ------------------------------------------------------------
# 17. FACT_MARKETING
# ------------------------------------------------------------

marketing = pd.DataFrame({
    "campaign_id": np.arange(1, 301),

    "campaign_date": rng.choice(
        months,
        300
    ),

    "channel": rng.choice(
        ["Google Ads", "Meta Ads",
         "Email", "Afiliados"],
        300
    ),

    "campaign_cost": rng.uniform(
        500, 15000, 300
    ).round(2),

    "impressions": rng.integers(
        10000, 500000, 300
    ),

    "clicks": rng.integers(
        500, 30000, 300
    ),

    "conversions": rng.integers(
        10, 1500, 300
    )
})

marketing["roas_revenue"] = (
    marketing["campaign_cost"]
    * rng.uniform(2, 8, 300)
).round(2)

marketing["roas"] = (
    marketing["roas_revenue"]
    / marketing["campaign_cost"]
)

# ------------------------------------------------------------
# 18. FACT_FINANCIAL
# ------------------------------------------------------------

monthly_sales = (
    sales.groupby(
        sales["order_date"].dt.to_period("M")
    )["order_revenue"]
    .sum()
)

financial = monthly_sales.rename(
    "revenue"
).reset_index()

financial["month"] = (
    financial["order_date"].dt.to_timestamp()
)

financial = financial.drop(
    columns="order_date"
)

financial["operating_expenses"] = (
    financial["revenue"]
    * rng.uniform(0.08, 0.18, len(financial))
).round(2)

financial["depreciation_amortization"] = (
    financial["revenue"] * 0.02
).round(2)

financial["ebitda"] = (
    financial["revenue"]
    - financial["operating_expenses"]
)

financial["cash_inflow"] = (
    financial["revenue"]
    * rng.uniform(0.90, 1.00, len(financial))
).round(2)

financial["cash_outflow"] = (
    financial["operating_expenses"]
    + financial["depreciation_amortization"]
).round(2)

financial["cash_flow"] = (
    financial["cash_inflow"]
    - financial["cash_outflow"]
).round(2)

# ------------------------------------------------------------
# 19. FACT_TARGETS
# ------------------------------------------------------------

targets = financial[["month", "revenue"]].copy()

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

# ------------------------------------------------------------
# 20. RESUMO DOS FATOS
# ------------------------------------------------------------

facts = {
    "fact_sales": sales,
    "fact_sales_items": items,
    "fact_returns": returns,
    "fact_inventory": inventory,
    "fact_marketing": marketing,
    "fact_financial": financial,
    "fact_targets": targets
}

print("\nTABELAS FATO CRIADAS")

for name, df in facts.items():
    print(f"{name}: {len(df):,} registros")
2.2. Entendendo os indicadores gerados
É importante compreender como as tabelas serão utilizadas.

Tabela

Finalidade

fact_sales

Quantidade de pedidos, receita e SLA

fact_sales_items

Produtos, custos, descontos e lucro

fact_returns

Devoluções e reembolsos

fact_inventory

Estoque e rupturas

fact_marketing

Investimentos e retorno das campanhas

fact_financial

EBITDA, despesas e fluxo de caixa

fact_targets

Comparação entre metas e resultados

Uma observação sobre o lucro: a margem de 25,6% será garantida na base de itens. O EBITDA será calculado separadamente, considerando despesas operacionais. Portanto, não devemos apresentar os R$ 2,1 milhões como EBITDA.

Também é importante observar que os dados são sintéticos. As metas financeiras mensais serão inicialmente distribuídas de maneira uniforme, servindo como ponto de partida para o projeto. Posteriormente, podemos criar metas sazonais mais realistas.

PARTE 3/3 — Qualidade, validação, integração e exportação
Nesta última parte vamos transformar os dados gerados em uma base consolidada.

O processo terá cinco etapas:

Validar chaves primárias.

Validar relacionamentos entre tabelas.

Verificar inconsistências financeiras.

Criar indicadores executivos.

Exportar os dados para CSV e SQLite.

3.1. Código de validação
Adicione ao final do arquivo:


python

# ============================================================
# VISION GROUP
# PARTE 3/3 - QUALIDADE E EXPORTACAO
# ============================================================

# ------------------------------------------------------------
# 21. VALIDACAO DAS DIMENSOES
# ------------------------------------------------------------

assert customers["customer_id"].is_unique
assert products["product_id"].is_unique
assert categories["category_id"].is_unique
assert sellers["seller_id"].is_unique
assert stores["store_id"].is_unique
assert regions["region_id"].is_unique
assert dim_date["date_id"].is_unique

# ------------------------------------------------------------
# 22. VALIDACAO DAS TABELAS FATO
# ------------------------------------------------------------

assert sales["order_id"].is_unique
assert items["item_id"].is_unique
assert returns["return_id"].is_unique

# ------------------------------------------------------------
# 23. VALIDACAO DE CHAVES ESTRANGEIRAS
# ------------------------------------------------------------

assert sales["customer_id"].isin(
    customers["customer_id"]
).all()

assert sales["seller_id"].isin(
    sellers["seller_id"]
).all()

assert sales["store_id"].isin(
    stores["store_id"]
).all()

assert items["order_id"].isin(
    sales["order_id"]
).all()

assert items["product_id"].isin(
    products["product_id"]
).all()

assert returns["order_id"].isin(
    sales["order_id"]
).all()

# ------------------------------------------------------------
# 24. VALIDACAO FINANCEIRA
# ------------------------------------------------------------

assert (items["quantity"] > 0).all()

assert (items["unit_price"] > 0).all()

assert (items["discount_value"] >= 0).all()

assert (
    items["discount_value"]
    <= items["gross_revenue"]
).all()

assert (
    items["line_cost"] >= 0
).all()

assert np.isfinite(
    items["gross_margin"]
).all()

# Verificar receita e lucro por item
assert np.allclose(
    items["gross_profit"],
    items["net_revenue"] - items["line_cost"],
    atol=0.01
)

# ------------------------------------------------------------
# 25. VALIDACAO DE DEVOLUCOES
# ------------------------------------------------------------

assert (
    returns["return_date"]
    >= returns["order_id"].map(
        sales.set_index("order_id")["delivered_date"]
    )
).all()

assert (
    returns["refund_value"] >= 0
).all()

# ------------------------------------------------------------
# 26. VALIDACAO DE ESTOQUE
# ------------------------------------------------------------

assert (
    inventory["stock_quantity"] >= 0
).all()

assert (
    inventory["stock_value"] >= 0
).all()

# ------------------------------------------------------------
# 27. VALIDACAO DE MARKETING
# ------------------------------------------------------------

assert (
    marketing["clicks"]
    <= marketing["impressions"]
).all()

assert (
    marketing["conversions"]
    <= marketing["clicks"]
).all()

# ------------------------------------------------------------
# 28. INDICADORES EXECUTIVOS
# ------------------------------------------------------------

total_revenue = items["net_revenue"].sum()

total_cost = items["line_cost"].sum()

total_profit = items["gross_profit"].sum()

gross_margin = (
    total_profit / total_revenue
)

total_customers = customers["customer_id"].nunique()

total_orders = sales["order_id"].nunique()

average_ticket = (
    total_revenue / total_orders
)

total_returns = len(returns)

return_rate = (
    total_returns / total_orders
)

canceled_orders = (
    sales["status"] == "Cancelado"
).sum()

cancellation_rate = (
    canceled_orders / total_orders
)

executive_kpis = pd.DataFrame({
    "indicator": [
        "Revenue",
        "Cost",
        "Gross Profit",
        "Gross Margin",
        "Customers",
        "Orders",
        "Average Ticket",
        "Returns",
        "Return Rate",
        "Canceled Orders",
        "Cancellation Rate"
    ],

    "value": [
        total_revenue,
        total_cost,
        total_profit,
        gross_margin,
        total_customers,
        total_orders,
        average_ticket,
        total_returns,
        return_rate,
        canceled_orders,
        cancellation_rate
    ]
})

print("\nEXECUTIVE KPIs")

print(executive_kpis.to_string(index=False))

# ------------------------------------------------------------
# 29. ANALISE MENSAL
# ------------------------------------------------------------

sales["year_month"] = (
    sales["order_date"].dt.to_period("M")
)

monthly_sales = (
    sales.groupby("year_month")
    .agg(
        revenue=("order_revenue", "sum"),
        profit=("order_profit", "sum"),
        orders=("order_id", "nunique")
    )
    .reset_index()
)

monthly_sales["revenue_mom"] = (
    monthly_sales["revenue"].pct_change()
)

monthly_sales["revenue_yoy"] = (
    monthly_sales["revenue"].pct_change(12)
)

monthly_sales["revenue_cumulative"] = (
    monthly_sales["revenue"].cumsum()
)

# ------------------------------------------------------------
# 30. ANALISE POR REGIAO
# ------------------------------------------------------------

regional_sales = sales.merge(
    customers[["customer_id", "region"]],
    on="customer_id",
    how="left"
)

regional_performance = (
    regional_sales.groupby("region")
    .agg(
        revenue=("order_revenue", "sum"),
        profit=("order_profit", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique")
    )
    .reset_index()
)

regional_performance["margin"] = (
    regional_performance["profit"]
    / regional_performance["revenue"].replace(0, np.nan)
)

# ------------------------------------------------------------
# 31. ANALISE POR CATEGORIA
# ------------------------------------------------------------

category_performance = items.merge(
    products[["product_id", "category_name"]],
    on="product_id",
    how="left"
)

category_performance = (
    category_performance.groupby("category_name")
    .agg(
        revenue=("net_revenue", "sum"),
        cost=("line_cost", "sum"),
        profit=("gross_profit", "sum"),
        quantity=("quantity", "sum")
    )
    .reset_index()
)

category_performance["margin"] = (
    category_performance["profit"]
    / category_performance["revenue"]
)

# ------------------------------------------------------------
# 32. ANALISE DE CLIENTES - RFM
# ------------------------------------------------------------

customer_sales = sales.merge(
    items.groupby("order_id")["net_revenue"]
    .sum()
    .reset_index(),
    on="order_id",
    how="left"
)

reference_date = END_DATE + pd.Timedelta(days=1)

rfm = (
    customer_sales.groupby("customer_id")
    .agg(
        last_purchase=("order_date", "max"),
        frequency=("order_id", "nunique"),
        monetary=("net_revenue", "sum")
    )
    .reset_index()
)

rfm["recency"] = (
    reference_date - rfm["last_purchase"]
).dt.days

# Pontuacao RFM
rfm["R"] = pd.qcut(
    rfm["recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F"] = pd.qcut(
    rfm["frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M"] = pd.qcut(
    rfm["monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["rfm_score"] = (
    rfm["R"] + rfm["F"] + rfm["M"]
)

rfm["customer_class"] = np.select(
    [
        rfm["rfm_score"] >= 13,
        rfm["rfm_score"] >= 9,
        rfm["rfm_score"] >= 6
    ],
    [
        "VIP",
        "Regular",
        "Em Desenvolvimento"
    ],
    default="Em Risco"
)

# ------------------------------------------------------------
# 33. IDENTIFICACAO DE ANOMALIAS
# ------------------------------------------------------------

monthly_sales["revenue_mean"] = (
    monthly_sales["revenue"].mean()
)

monthly_sales["revenue_std"] = (
    monthly_sales["revenue"].std()
)

monthly_sales["z_score"] = (
    (
        monthly_sales["revenue"]
        - monthly_sales["revenue_mean"]
    )
    / monthly_sales["revenue_std"]
)

monthly_sales["anomaly"] = (
    monthly_sales["z_score"].abs() > 2
)

# ------------------------------------------------------------
# 34. EXPORTACAO CSV
# ------------------------------------------------------------

all_tables = {
    **dimensions,
    **facts
}

all_tables.update({
    "executive_kpis": executive_kpis,
    "monthly_sales": monthly_sales,
    "regional_performance": regional_performance,
    "category_performance": category_performance,
    "customer_rfm": rfm
})

for name, df in all_tables.items():

    path = os.path.join(
        PROCESSED_DIR,
        f"{name}.csv"
    )

    df.to_csv(
        path,
        index=False
    )

# ------------------------------------------------------------
# 35. EXPORTACAO PARA SQLITE
# ------------------------------------------------------------

engine = create_engine(
    "sqlite:///data/analytics/vision_group.db"
)

for name, df in all_tables.items():

    df.to_sql(
        name,
        engine,
        if_exists="replace",
        index=False,
        chunksize=10000
    )

# ------------------------------------------------------------
# 36. RELATORIO DE EXECUCAO
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("VISION GROUP - DATA PIPELINE CONCLUIDO")
print("=" * 65)

print(f"Empresa: {COMPANY}")

print(f"Clientes: {total_customers:,}")

print(f"Pedidos: {total_orders:,}")

print(f"Receita: R$ {total_revenue:,.2f}")

print(f"Lucro bruto: R$ {total_profit:,.2f}")

print(f"Margem bruta: {gross_margin:.2%}")

print(f"Ticket medio: R$ {average_ticket:,.2f}")

print(f"Devolucoes: {total_returns:,}")

print(f"Taxa de devolucao: {return_rate:.2%}")

print(f"Cancelamentos: {canceled_orders:,}")

print(f"Taxa de cancelamento: {cancellation_rate:.2%}")

print("\nArquivos CSV exportados.")

print("Banco SQLite criado.")

print("\nTodas as validacoes executadas.")
3.2. Atenção: dois ajustes importantes antes de considerar a base definitiva
O código acima é uma primeira implementação funcional da estrutura, mas existem duas questões de consistência que precisam ser resolvidas antes de usar os números como resultados oficiais.

Ajuste 1 — Receita e cancelamentos

A receita de R$ 8,2 milhões é calibrada nos itens antes de zerar os pedidos cancelados. Portanto, a receita total de pedidos pode ficar abaixo da meta.

A solução correta é calibrar a receita considerando somente itens de pedidos não cancelados e, depois, recalcular os custos e os totais de fact_sales. Também devemos zerar a receita e o custo dos itens cancelados. Não basta zerar somente o cabeçalho do pedido.

Ajuste 2 — Relacionamento das devoluções

O modelo atual registra devoluções por pedido, não por produto. Para uma análise de rentabilidade após devoluções por categoria e produto, a evolução correta é criar fact_returns_items, contendo return_id, order_id, product_id, quantidade devolvida e valor reembolsado.

Assim, não distribuímos um reembolso arbitrariamente entre os produtos de um pedido.

Esses cuidados são importantes porque um dashboard executivo precisa apresentar números conciliados, não apenas gráficos visualmente bonitos.

4. Camada analítica SQL
Agora que temos o banco vision_group.db, vamos criar as seis views exigidas pelo projeto.

Crie o arquivo:

sql/01_create_views.sql

O banco SQLite permite executar essas consultas pelo DBeaver, DB Browser for SQLite ou Python.

4.1. View — Executive KPIs
Consolida os principais indicadores da empresa.


sql
CREATE VIEW IF NOT EXISTS vw_executive_kpis AS

WITH sales AS (
    SELECT
        SUM(order_revenue) AS revenue,
        SUM(order_profit) AS profit,
        COUNT(DISTINCT order_id) AS orders
    FROM fact_sales
),

customers AS (
    SELECT
        COUNT(DISTINCT customer_id) AS customers
    FROM dim_customers
),

returns AS (
    SELECT
        COUNT(DISTINCT order_id) AS returned_orders
    FROM fact_returns
)

SELECT
    s.revenue,
    s.profit,

    s.profit / NULLIF(s.revenue, 0) AS margin,

    s.orders,
    c.customers,

    s.revenue / NULLIF(s.orders, 0) AS average_ticket,

    r.returned_orders,

    r.returned_orders * 1.0
        / NULLIF(s.orders, 0) AS return_rate

FROM sales s
CROSS JOIN customers c
CROSS JOIN returns r;
4.2. View — Sales Performance
Analisa vendas mensais, crescimento e desempenho comercial.


sql
CREATE VIEW IF NOT EXISTS vw_sales_performance AS

WITH monthly AS (

    SELECT
        strftime('%Y-%m', order_date) AS year_month,

        SUM(order_revenue) AS revenue,

        COUNT(DISTINCT order_id) AS orders,

        AVG(order_revenue) AS average_ticket

    FROM fact_sales

    GROUP BY strftime('%Y-%m', order_date)
),

comparison AS (

    SELECT
        *,

        LAG(revenue, 1) OVER (
            ORDER BY year_month
        ) AS previous_month,

        LAG(revenue, 12) OVER (
            ORDER BY year_month
        ) AS previous_year

    FROM monthly
)

SELECT
    year_month,
    revenue,
    orders,
    average_ticket,

    (revenue - previous_month)
        / NULLIF(previous_month, 0) AS growth_mom,

    (revenue - previous_year)
        / NULLIF(previous_year, 0) AS growth_yoy,

    SUM(revenue) OVER (
        ORDER BY year_month
        ROWS BETWEEN UNBOUNDED PRECEDING
        AND CURRENT ROW
    ) AS cumulative_revenue

FROM comparison;
4.3. View — Customer Performance
Analisa frequência de compra e comportamento dos clientes.


sql
CREATE VIEW IF NOT EXISTS vw_customer_performance AS

WITH customer_orders AS (

    SELECT
        customer_id,

        COUNT(DISTINCT order_id) AS frequency,

        SUM(order_revenue) AS monetary,

        MAX(order_date) AS last_purchase,

        MIN(order_date) AS first_purchase

    FROM fact_sales

    WHERE status != 'Cancelado'

    GROUP BY customer_id
)

SELECT
    c.customer_id,

    c.customer_segment,

    c.acquisition_channel,

    co.frequency,

    co.monetary,

    co.first_purchase,

    co.last_purchase,

    julianday('2025-12-31')
        - julianday(co.last_purchase) AS recency

FROM dim_customers c

LEFT JOIN customer_orders co
    ON c.customer_id = co.customer_id;
4.4. View — Product Profitability
Identifica os produtos e categorias mais rentáveis.


sql
CREATE VIEW IF NOT EXISTS vw_product_profitability AS

SELECT
    p.product_id,

    p.product_name,

    p.category_name,

    SUM(i.quantity) AS units_sold,

    SUM(i.net_revenue) AS revenue,

    SUM(i.line_cost) AS cost,

    SUM(i.gross_profit) AS profit,

    SUM(i.gross_profit)
        / NULLIF(SUM(i.net_revenue), 0) AS margin

FROM fact_sales_items i

INNER JOIN dim_products p
    ON i.product_id = p.product_id

GROUP BY
    p.product_id,
    p.product_name,
    p.category_name;
4.5. View — Operational Performance
Analisa cancelamentos, devoluções, entregas e estoque.


sql
CREATE VIEW IF NOT EXISTS vw_operational_performance AS

WITH sales AS (

    SELECT
        COUNT(*) AS total_orders,

        SUM(
            CASE
                WHEN status = 'Cancelado'
                THEN 1 ELSE 0
            END
        ) AS canceled_orders,

        SUM(
            CASE
                WHEN status = 'Entregue'
                THEN 1 ELSE 0
            END
        ) AS delivered_orders,

        AVG(
            CASE
                WHEN status = 'Entregue'
                THEN julianday(delivered_date)
                   - julianday(order_date)
            END
        ) AS average_delivery_days,

        AVG(
            CASE
                WHEN status = 'Entregue'
                 AND delivered_date <= promised_delivery_date
                THEN 1.0
                WHEN status = 'Entregue'
                THEN 0.0
            END
        ) AS sla_rate

    FROM fact_sales
),

returns AS (

    SELECT
        COUNT(DISTINCT order_id) AS returned_orders
    FROM fact_returns
),

inventory AS (

    SELECT
        COUNT(*) AS inventory_records,

        SUM(
            CASE
                WHEN stock_status = 'Ruptura'
                THEN 1 ELSE 0
            END
        ) AS stockouts

    FROM fact_inventory
)

SELECT
    s.total_orders,
    s.canceled_orders,
    s.delivered_orders,
    s.average_delivery_days,
    s.sla_rate,

    r.returned_orders,

    i.inventory_records,
    i.stockouts,

    i.stockouts * 1.0
        / NULLIF(i.inventory_records, 0) AS stockout_rate

FROM sales s
CROSS JOIN returns r
CROSS JOIN inventory i;
4.6. View — Financial Performance
Consolida os resultados financeiros.


sql
CREATE VIEW IF NOT EXISTS vw_financial_performance AS

SELECT
    month,

    revenue,

    operating_expenses,

    depreciation_amortization,

    ebitda,

    cash_inflow,

    cash_outflow,

    cash_flow,

    SUM(cash_flow) OVER (
        ORDER BY month
        ROWS BETWEEN UNBOUNDED PRECEDING
        AND CURRENT ROW
    ) AS cumulative_cash_flow

FROM fact_financial;
Importante: as views financeiras devem ser conciliadas com as vendas e com as despesas efetivamente registradas. No modelo inicial, as despesas são simuladas e o EBITDA é uma estimativa sintética, não um demonstrativo contábil.

4.7. Outras consultas SQL para demonstrar conhecimento
Além das views, recomendo incluir consultas individuais demonstrando CTEs, Window Functions, Subqueries e agregações.

Consulta 1 — Top 10 produtos por lucro


sql
SELECT
    product_name,
    category_name,
    revenue,
    profit,
    margin

FROM vw_product_profitability

ORDER BY profit DESC

LIMIT 10;
Consulta 2 — Categorias abaixo da margem média


sql
WITH category_margin AS (

    SELECT
        category_name,

        SUM(gross_profit)
            / NULLIF(SUM(net_revenue), 0) AS margin

    FROM fact_sales_items i

    JOIN dim_products p
        ON i.product_id = p.product_id

    GROUP BY category_name
)

SELECT *
FROM category_margin

WHERE margin < (
    SELECT AVG(margin)
    FROM category_margin
)

ORDER BY margin;
Consulta 3 — Ranking de regiões


sql
WITH regional_sales AS (

    SELECT
        c.region,

        SUM(s.order_revenue) AS revenue

    FROM fact_sales s

    JOIN dim_customers c
        ON s.customer_id = c.customer_id

    GROUP BY c.region
)

SELECT
    region,
    revenue,

    RANK() OVER (
        ORDER BY revenue DESC
    ) AS regional_rank

FROM regional_sales;
Consulta 4 — Comparação de receita mensal com o mês anterior


sql
SELECT
    year_month,
    revenue,
    previous_month,

    revenue - previous_month AS revenue_variation

FROM vw_sales_performance

ORDER BY year_month;
Consulta 5 — Meta versus realizado


sql
SELECT
    t.month,

    t.revenue_target,

    f.revenue AS actual_revenue,

    f.revenue - t.revenue_target AS variance,

    (f.revenue - t.revenue_target)
        / NULLIF(t.revenue_target, 0) AS variance_percentage

FROM fact_targets t

JOIN fact_financial f
    ON t.month = f.month

ORDER BY t.month;
5. Construção do Dashboard Executivo no Power BI
Agora chegamos à parte visual do projeto.

Conecte o Power BI ao banco:

data/analytics/vision_group.db

Você pode utilizar o conector SQLite disponível no seu ambiente ou exportar os dados para CSV e importá-los.

Minha recomendação é carregar as tabelas dimensionais e fatos, mantendo o modelo estrela.

Página 1 — Executive Overview
Visão da diretoria
Power BI Executive Sales Dashboard Template | Editable Sales Analytics Report | Dark Theme - Etsy
 
Indicadores:

Receita.

Lucro bruto.

Margem.

Clientes.

Pedidos.

Ticket médio.

Crescimento.

Meta versus realizado.

Medidas DAX iniciais:


dax
Receita =
SUM(fact_sales[order_revenue])

dax
Lucro =
SUM(fact_sales[order_profit])

dax
Margem =
DIVIDE([Lucro], [Receita], 0)

dax
Pedidos =
DISTINCTCOUNT(fact_sales[order_id])

dax
Clientes =
DISTINCTCOUNT(fact_sales[customer_id])

dax
Ticket Medio =
DIVIDE([Receita], [Pedidos], 0)

dax
Meta Receita =
SUM(fact_targets[revenue_target])

dax
Atingimento Meta =
DIVIDE([Receita], [Meta Receita], 0)
Pergunta respondida: como está o negócio?

Página 2 — Sales Performance
Power Bi Dashboards & reports | AI-Powered Power BI Resources | Briqlab
 
Indicadores:

Receita por mês.

Receita por região.

Receita por categoria.

Receita por produto.

Ticket médio.

Crescimento MoM.

Crescimento YoY.

Gráficos recomendados:

Linha: evolução mensal da receita.

Mapa: vendas por estado brasileiro.

Barras: receita por categoria.

Ranking: 10 produtos mais vendidos.

Cartões: crescimento MoM e YoY.

Página 3 — Customer Analytics
The Value Of RFM Analysis In Ecommerce Customer Segmentation
 
Indicadores:

Clientes novos.

Clientes recorrentes.

Retenção.

Churn.

Cohort.

RFM.

LTV.

Aqui recomendo criar uma tabela de coortes por mês da primeira compra.

Uma definição operacional para começar:

Cliente novo: primeira compra no período.

Cliente recorrente: cliente que já comprou antes do período e voltou a comprar.

Churn: cliente que não realizou compra durante uma janela definida de inatividade.

Retenção: percentual de clientes de uma coorte que voltaram a comprar nos períodos seguintes.

Não confunda cliente sem compra em determinado mês com cliente definitivamente perdido.

Página 4 — Profitability
Financial Dashboard - Microsoft Fabric Community
 
Indicadores:

Receita.

Custos.

Lucro.

Margem.

Margem por produto.

Margem por categoria.

Gráficos:

Receita versus custo.

Margem por categoria.

Ranking de produtos rentáveis.

Produtos com receita alta e margem baixa.

Impacto dos descontos na margem.

Página 5 — Operations
Power BI Template for Inventory Analysis - Microsoft Fabric Community
 
Indicadores:

Pedidos.

Cancelamentos.

Devoluções.

SLA.

Estoque.

Rupturas.

Performance operacional.

Gráficos:

Taxa de cancelamento.

Evolução de devoluções.

Entregas dentro do prazo.

Produtos com estoque crítico.

Ruptura por loja.

Tempo médio de entrega.

Página 6 — Financial Performance
Power BI Financial KPI Dashboard
 
Indicadores:

Receita.

Despesas.

EBITDA.

Cash Flow.

Budget versus Actual.

Desvios orçamentários.

Gráficos:

Receita versus despesas.

Evolução do EBITDA.

Fluxo de caixa acumulado.

Meta versus realizado.

Desvios por mês.

6. Relatório de análise executiva
A última etapa é produzir um documento com linguagem empresarial, como se você fosse um analista apresentando os resultados à diretoria.

Crie o arquivo reports/executive_report.md.

Sugestão de estrutura:

Escrita


VISION GROUP
Relatório Executivo de Business Intelligence
Período analisado: Janeiro de 2021 a Dezembro de 2025

1. Situação geral da empresa
A Vision Group é uma rede varejista fictícia com atuação nacional e presença nas cinco regiões brasileiras.

A empresa possui uma base de 82.400 clientes e um histórico de 145.200 pedidos.

Os indicadores corporativos de referência são:

Receita líquida comercial: R$ 8.200.000,00.
Lucro bruto: R$ 2.100.000,00.
Margem bruta: aproximadamente 25,6%.
Histórico: cinco anos.
A análise executiva busca avaliar o desempenho comercial, financeiro, operacional e o comportamento dos clientes.

2. Principais resultados positivos
Nesta seção devem ser apresentados os resultados identificados nas análises:

Regiões com maior contribuição para a receita.
Categorias com maior rentabilidade.
Crescimento da receita ao longo dos períodos.
Produtos com maior contribuição para o lucro.
Segmentos de clientes com maior frequência de compra.
Campanhas com melhor retorno sobre investimento.
3. Principais problemas identificados
A análise deverá investigar:

Categorias com redução de margem.
Produtos com alto volume e baixa rentabilidade.
Regiões com desempenho inferior às metas.
Aumento de cancelamentos e devoluções.
Problemas de SLA.
Rupturas de estoque.
Clientes com risco de churn.
Despesas acima do orçamento.
4. Oportunidades de melhoria
As principais oportunidades a serem avaliadas incluem:

Revisão das políticas de desconto.
Otimização do mix de produtos.
Melhoria da retenção de clientes.
Reestruturação de campanhas de marketing.
Otimização dos níveis de estoque.
Melhoria dos processos logísticos.
Controle das despesas operacionais.
5. Recomendações para a diretoria
Acompanhar mensalmente a margem por categoria e produto.
Investigar os motivos de devoluções e cancelamentos.
Criar estratégias de retenção para clientes de alto valor.
Priorizar produtos com maior contribuição para o lucro.
Estabelecer metas operacionais de SLA e ruptura.
Acompanhar o orçamento e o fluxo de caixa.
Implantar reuniões mensais de acompanhamento dos KPIs.
Conclusão executiva
A Vision Group possui uma base comercial relevante e oportunidades de melhoria em rentabilidade, relacionamento com clientes e eficiência operacional.

A utilização de indicadores integrados permite identificar áreas que necessitam de atenção e direcionar decisões estratégicas com base em dados.

As recomendações devem ser priorizadas conforme os resultados efetivamente encontrados nas análises, evitando conclusões baseadas apenas em dados agregados ou indicadores isolados.

Esse documento é um modelo. Os resultados positivos e problemas devem ser substituídos pelos achados reais depois de executar e validar a base.

7. Regras de negócio e documentação técnica
Para que o projeto tenha aparência profissional, crie também um documento chamado README.md.

Registre estas definições:

Indicador

Regra de negócio

Receita bruta

Quantidade × preço antes de descontos

Desconto

Redução concedida ao cliente

Receita comercial

Receita bruta − descontos

Devolução

Valor efetivamente reembolsado

Receita ajustada

Receita comercial − reembolsos

Custo

Quantidade × custo unitário

Lucro bruto

Receita comercial − custo

Margem bruta

Lucro bruto / receita comercial

Ticket médio

Receita / pedidos válidos

Churn

Clientes inativos segundo janela definida

Retenção

Clientes da coorte que retornaram

SLA

Entregas realizadas até a data prometida

ROAS

Receita atribuída / investimento em anúncios

EBITDA

Resultado operacional antes de juros, impostos, depreciação e amortização

Uma regra particularmente importante: o modelo precisa distinguir pedidos criados, pedidos entregues, pedidos cancelados e pedidos devolvidos. Um pedido pode ser entregue e posteriormente devolvido; esses acontecimentos não devem ser tratados como estados mutuamente exclusivos.

8. Checklist da entrega final
Para apresentar esse projeto em um portfólio profissional, recomendo a seguinte organização:

Progresso do projeto

0%
0 de 27 atividades concluídas

1. Base de dados


82.400 clientes e 145.200 pedidos

Dimensões e tabelas fato

Cobertura das cinco regiões brasileiras

Cinco anos de histórico

Regras de negócio documentadas
2. Python


Geração e integração dos dados

Tratamento de dados ausentes e duplicados

Validação de chaves e relacionamentos

Análise exploratória

KPIs e comparação com metas

RFM, tendências e anomalias
3. SQL


Seis views executivas

CTEs e subqueries

Window Functions

Agregações e comparações temporais

Consultas de validação
4. Power BI


Executive Overview

Sales Performance

Customer Analytics

Profitability

Operations

Financial Performance
5. Entrega executiva


Relatório com cinco partes

Recomendações à diretoria

Documentação técnica

README e dicionário de dados

Apresentação do projeto
9. Ajustes que eu faria antes de publicar no portfólio
Há uma diferença entre criar uma base sintética e construir um projeto de BI realmente confiável. Eu faria estes aprimoramentos antes de considerar a entrega concluída:

Calibrar a receita depois dos cancelamentos: os R$ 8,2 milhões precisam ser atingidos na mesma granularidade e com os mesmos filtros utilizados nos indicadores executivos.

Criar devoluções por item: isso permitirá calcular a receita ajustada e a rentabilidade real por produto.

Criar uma tabela de movimentação de estoque: estoque mensal é uma fotografia; para medir entradas, saídas e giro, precisamos registrar movimentações.

Melhorar as metas: distribuir metas por mês, região, categoria e vendedor, com sazonalidade.

Melhorar o financeiro: incluir juros, impostos, depreciação, amortização, contas a pagar e receber e movimentações de caixa.

Calcular LTV e churn com critérios documentados: os valores precisam respeitar o período de observação e a definição de cliente ativo.

Criar testes automatizados: além de assert, registrar erros, alertas e resultados de cada execução.

Minha recomendação: não publique ainda os números das análises como resultados definitivos. O código apresentado estabelece a estrutura e os indicadores iniciais, mas precisa dos ajustes de conciliação descritos para que as métricas financeiras sejam confiáveis. Em especial, a meta de R$ 8,2 milhões não está garantida na tabela de pedidos após o tratamento dos cancelamentos.

Com esses ajustes, você terá um projeto de portfólio muito mais próximo do trabalho de um analista de BI dentro de uma empresa: geração e governança de dados, modelagem dimensional, ETL, SQL analítico, Power BI e recomendações para a diretoria.


refaça o script mantendo somente a parte de geração do dataset, n coloque validações ou analises. Quero somente criar o dataset. Ajuste Ajuste 1 — Receita e cancelamentos

A receita de R$ 8,2 milhões é calibrada nos itens antes de zerar os pedidos cancelados. Portanto, a receita total de pedidos pode ficar abaixo da meta.

A solução correta é calibrar a receita considerando somente itens de pedidos não cancelados e, depois, recalcular os custos e os totais de fact_sales. Também devemos zerar a receita e o custo dos itens cancelados. Não basta zerar somente o cabeçalho do pedido.

Ajuste 2 — Relacionamento das devoluções

O modelo atual registra devoluções por pedido, não por produto. Para uma análise de rentabilidade após devoluções por categoria e produto, a evolução correta é criar fact_returns_items, contendo return_id, order_id, product_id, quantidade devolvida e valor reembolsado.

Assim, não distribuímos um reembolso arbitrariamente entre os produtos de um pedido.

Esses cuidados são importantes porque um dashboard executivo precisa apresentar números conciliados, não apenas gráficos visualmente bonitos.


Divida em 3 partes

Pensou por 1m 14s
Vision Group — Geração do Dataset (Python)
Vamos refazer o script mantendo exclusivamente a geração dos dados fictícios, sem validações, análises exploratórias, KPIs ou relatórios.

O código será dividido em três partes, que deverão ser executadas na sequência, no mesmo arquivo Python.

Desta vez, vamos corrigir os dois problemas estruturais:

A receita de R$ 8.200.000,00 será atingida considerando somente pedidos não cancelados. Os itens de pedidos cancelados terão receita, desconto reconhecido, custo e lucro iguais a zero.

As devoluções serão separadas em duas tabelas:

fact_returns: informações gerais da devolução.

fact_returns_items: produtos e quantidades devolvidas, com reembolso individual por item.

1. Estrutura final do dataset
O script criará 16 arquivos CSV:

Dimensões

Tabelas fato

dim_customers

fact_sales

dim_products

fact_sales_items

dim_categories

fact_returns

dim_sellers

fact_returns_items

dim_stores

fact_inventory

dim_regions

fact_marketing

dim_date

fact_financial

fact_targets

Parâmetros da empresa
Indicador

Valor

Clientes

82.400

Pedidos

145.200

Produtos

500

Vendedores

120

Lojas

35

Receita comercial

R$ 8.200.000

Lucro bruto

R$ 2.100.000

Margem bruta

25,61%

Histórico

2021–2025

Cobertura

Brasil

Regra financeira: a receita de R$ 8,2 milhões será a receita comercial após descontos e antes das devoluções. O lucro bruto de R$ 2,1 milhões será calculado sobre essa mesma receita, descontando o custo dos produtos.

PARTE 1/3 — Configurações e dimensões
Crie o arquivo vision_group.py.

Nesta primeira parte, vamos gerar todas as dimensões da empresa.


python

# ============================================================
# VISION GROUP - EXECUTIVE BI DASHBOARD
# PARTE 1/3 - CONFIGURACOES E DIMENSOES
# ============================================================

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

Resultado da parte 1: sete dimensões preparadas para relacionar clientes, produtos, categorias, vendedores, lojas, regiões e calendário.

PARTE 2/3 — Vendas, pedidos e itens vendidos
Agora vamos criar as duas principais tabelas comerciais.

fact_sales: uma linha por pedido.

fact_sales_items: uma linha por produto vendido dentro de cada pedido.

Nesta etapa está a principal correção financeira do projeto.

Como será feito o ajuste da receita?
O processo será:

Gerar os 145.200 pedidos.

Definir os pedidos cancelados.

Gerar os itens de todos os pedidos.

Excluir os pedidos cancelados da base de receita elegível.

Distribuir os R$ 8,2 milhões entre os itens dos pedidos não cancelados.

Calibrar os custos para que o lucro bruto seja R$ 2,1 milhões.

Zerar os valores comerciais dos itens cancelados.

Consolidar os valores dos itens na tabela de pedidos.

Dessa maneira, os valores das duas tabelas ficarão conciliados.

Adicione o código abaixo ao final da primeira parte:


python

# ============================================================
# VISION GROUP
# PARTE 2/3 - VENDAS E ITENS VENDIDOS
# ============================================================

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

Como ficam os valores dos pedidos cancelados?
A tabela fact_sales continuará registrando o pedido cancelado, mas seus valores comerciais serão zero.

A tabela fact_sales_items também manterá os produtos originalmente associados ao pedido, mas terá:

Campo

Pedido cancelado

gross_revenue

0

discount_value

0

net_revenue

0

line_cost

0

gross_profit

0

gross_margin

0

Isso preserva o histórico de pedidos sem reconhecer vendas que não foram realizadas.

Importante: o custo unitário cadastrado no produto permanece como custo de referência. O campo unit_cost dos itens representa o custo unitário calibrado para o cenário financeiro sintético.

PARTE 3/3 — Devoluções, estoque, marketing, financeiro e exportação
Nesta última parte vamos gerar as demais tabelas fato e exportar o dataset completo.

3.1. Devoluções por produto
A estrutura será:

fact_returns

return_id

order_id

return_date

return_reason

refund_value

1

102

2022-05-10

Produto danificado

150,00

2

205

2023-08-15

Insatisfação

89,00

fact_returns_items

return_item_id

return_id

product_id

returned_quantity

refund_value

1

1

35

1

100,00

2

1

48

2

50,00

3

2

100

1

89,00

Assim, será possível identificar quais produtos e categorias apresentam maior volume de devoluções.

3.2. Código final
Adicione o código abaixo ao final da parte 2:


python

# ============================================================
# VISION GROUP
# PARTE 3/3 - DEVOLUCOES, ESTOQUE, MARKETING E FINANCEIRO
# ============================================================

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
4. Resultado final do projeto
Ao executar o arquivo vision_group.py, será criada a seguinte estrutura:


text
vision_group_dataset/
│
└── raw/
    │
    ├── dim_customers.csv
    ├── dim_products.csv
    ├── dim_categories.csv
    ├── dim_sellers.csv
    ├── dim_stores.csv
    ├── dim_regions.csv
    ├── dim_date.csv
    │
    ├── fact_sales.csv
    ├── fact_sales_items.csv
    ├── fact_returns.csv
    ├── fact_returns_items.csv
    ├── fact_inventory.csv
    ├── fact_marketing.csv
    ├── fact_financial.csv
    └── fact_targets.csv
15 arquivos CSV 7 dimensões 8 tabelas fato

5. Relacionamento entre as tabelas
O modelo de dados ficará organizado assim:

Star Schema vs Snowflake Schema and the 7 Critical Differences — Blog | Keboola
 
Os principais relacionamentos serão:

dim_customers.customer_id → fact_sales.customer_id

dim_sellers.seller_id → fact_sales.seller_id

dim_stores.store_id → fact_sales.store_id

fact_sales.order_id → fact_sales_items.order_id

dim_products.product_id → fact_sales_items.product_id

fact_returns.return_id → fact_returns_items.return_id

fact_returns_items.product_id → dim_products.product_id

fact_returns.order_id → fact_sales.order_id

dim_categories.category_id → dim_products.category_id

6. Um último ajuste recomendado no código
Na parte 2, recomendo arredondar o preço unitário antes de calcular a receita dos itens.

Substitua:


python
unit_prices = (
    unit_prices * revenue_factor
)
Por:


python
unit_prices = np.round(
    unit_prices * revenue_factor,
    2
)
Assim, o preço unitário exportado terá maior consistência com a receita bruta dos itens.

Observação sobre o resultado: o código calibra a receita e o lucro bruto nas tabelas comerciais, mas os valores financeiros de EBITDA e fluxo de caixa são simulados separadamente. Além disso, as devoluções são registradas como eventos posteriores à venda, sem apagar a receita comercial original.

Com isso, você terá somente a etapa de geração do dataset, pronta para ser utilizada posteriormente nos projetos de Python, SQL e Power BI.



atualize o script para consertar os seguintes problemas:
Pontos que precisam de atenção
Antes de utilizar os dados no projeto, precisamos documentar algumas características do gerador.
Ponto	Situação no script	Tratamento necessário
Receita	Calibrada em R$ 8,2 milhões	Validar soma dos itens elegíveis
Lucro	Calibrado em R$ 2,1 milhões	Identificar como lucro bruto
Cancelamentos	Valores dos itens zerados	Excluir pedidos cancelados dos KPIs comerciais
Devoluções	Geradas separadamente	Descontar reembolsos nas análises líquidas
Custo dos produtos	Custo ajustado nos itens	Utilizar line_cost como custo comercial realizado
Clientes	Todos cadastrados na dimensão	Diferenciar cadastrados de compradores
Marketing	Receita atribuída sinteticamente	Não tratar como atribuição comprovada
Metas	Distribuídas igualmente por mês	Documentar como metas uniformes
Estoque	Fotografia mensal sintética	Não interpretar como movimentação real
No script, cidade e estado são sorteados separadamente:
Vamos corrigir essa inconsistência usando uma tabela de cidades associadas aos respectivos estados.

Atualização do gerador de dados — Vision Group
Vamos corrigir o script para que a base seja mais coerente com a realidade de uma empresa varejista e possa ser utilizada posteriormente no Python, SQL e Power BI.

Vou preservar a estrutura de três partes e manter o escopo exclusivamente na geração do dataset. Não vou adicionar análises exploratórias, dashboards ou rotinas de validação.

1. Correções que serão implementadas
Problema

Correção

Receita de R$ 8,2 milhões

Calibrar considerando somente pedidos não cancelados e conciliar os itens.

Lucro de R$ 2,1 milhões

Identificar como lucro bruto, calculado após o custo dos produtos.

Cancelamentos

Manter pedidos cancelados para análise operacional, mas excluí-los dos valores comerciais.

Devoluções

Registrar reembolsos por produto e permitir o cálculo da receita líquida de devoluções.

Custo dos produtos

Utilizar line_cost como custo comercial realizado.

Clientes

Diferenciar clientes cadastrados de clientes que efetivamente compraram.

Marketing

Identificar a receita atribuída como estimativa sintética.

Metas

Documentar que as metas mensais são uniformes.

Estoque

Identificar como fotografia mensal sintética.

Geografia

Utilizar cidades associadas corretamente aos respectivos estados.

Também vou acrescentar campos que facilitam o trabalho posterior:

is_canceled: identifica pedidos cancelados.

is_commercial: identifica pedidos elegíveis para indicadores comerciais.

is_returned: identifica pedidos com devolução.

net_revenue_after_returns: receita após reembolsos.

gross_profit_after_returns: lucro bruto após reembolsos, mantendo o custo comercial dos itens vendidos como referência.

Uma distinção importante: a receita de R$ 8,2 milhões será a receita comercial antes das devoluções. Depois dos reembolsos, a receita líquida será menor. O lucro de R$ 2,1 milhões também será o lucro bruto antes das devoluções.

PARTE 1/3 — Configurações e dimensões
Substitua a primeira parte anterior por esta versão.

A principal mudança geográfica é a criação de um cadastro de municípios por UF. O estado e a cidade passam a ser selecionados a partir da mesma estrutura, impedindo combinações inválidas como Curitiba–SP ou Salvador–MG.

Também incluí parâmetros de documentação do cenário sintético diretamente no código.


python

# ============================================================
# VISION GROUP - EXECUTIVE BI DASHBOARD
# PARTE 1/3 - CONFIGURACOES E DIMENSOES
# ============================================================

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
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")

os.makedirs(RAW_DIR, exist_ok=True)

# Quantidades
N_CUSTOMERS = 82_400
N_ORDERS = 145_200
N_PRODUCTS = 500
N_SELLERS = 120
N_STORES = 35

# Parametros financeiros
TARGET_REVENUE = 8_200_000.00
TARGET_GROSS_PROFIT = 2_100_000.00

TARGET_COST = (
    TARGET_REVENUE - TARGET_GROSS_PROFIT
)

# Historico
START_DATE = pd.Timestamp("2021-01-01")
END_DATE = pd.Timestamp("2025-12-31")

# Parametros operacionais
RETURN_RATE = 0.08
CANCELLATION_RATE = 0.04

# ============================================================
# 2. DOCUMENTACAO DAS REGRAS DE NEGOCIO
# ============================================================

BUSINESS_RULES = {

    "company": COMPANY,

    "dataset_type": "Synthetic",

    "currency": "BRL",

    "historical_period": "2021-01-01 a 2025-12-31",

    "target_revenue": TARGET_REVENUE,

    "target_gross_profit": TARGET_GROSS_PROFIT,

    "revenue_definition":
        "Receita comercial apos descontos e antes de devolucoes",

    "profit_definition":
        "Lucro bruto antes de devolucoes, calculado pela receita comercial menos custo dos produtos",

    "product_cost_definition":
        "line_cost representa o custo comercial realizado dos itens elegiveis",

    "canceled_orders":
        "Pedidos cancelados permanecem registrados, mas seus valores comerciais sao zero",

    "commercial_kpis":
        "Pedidos cancelados devem ser excluidos dos indicadores comerciais",

    "returns":
        "Reembolsos devem ser descontados da receita comercial nas analises liquidas",

    "customers":
        "Clientes cadastrados nao representam necessariamente clientes compradores",

    "marketing":
        "Receita atribuida e ROAS sao estimativas sinteticas, nao atribuicoes comprovadas",

    "targets":
        "Metas mensais distribuidas uniformemente ao longo do historico",

    "inventory":
        "Estoque representa fotografias mensais sinteticas, nao movimentacoes reais",

    "geography":
        "Cidades selecionadas exclusivamente dentro de seus respectivos estados",

    "financial":
        "Despesas, EBITDA e fluxo de caixa sao valores sinteticos"

}

# ============================================================
# 3. CADASTRO DE ESTADOS E CIDADES
# ============================================================

STATE_CITIES = {

    "AC": ["Rio Branco", "Cruzeiro do Sul", "Sena Madureira"],

    "AL": ["Maceio", "Arapiraca", "Palmeira dos Indios"],

    "AP": ["Macapa", "Santana", "Laranjal do Jari"],

    "AM": ["Manaus", "Parintins", "Itacoatiara"],

    "BA": ["Salvador", "Feira de Santana", "Vitoria da Conquista",
           "Ilheus", "Barreiras"],

    "CE": ["Fortaleza", "Juazeiro do Norte", "Sobral",
           "Crato", "Maracanau"],

    "DF": ["Brasilia", "Ceilandia", "Taguatinga"],

    "ES": ["Vitoria", "Vila Velha", "Serra",
           "Cariacica", "Cachoeiro de Itapemirim"],

    "GO": ["Goiania", "Anapolis", "Rio Verde",
           "Aparecida de Goiania", "Jatai"],

    "MA": ["Sao Luis", "Imperatriz", "Caxias"],

    "MT": ["Cuiaba", "Varzea Grande", "Rondonopolis",
           "Sinop", "Tangara da Serra"],

    "MS": ["Campo Grande", "Dourados", "Tres Lagoas"],

    "MG": ["Belo Horizonte", "Uberlandia", "Contagem",
           "Juiz de Fora", "Montes Claros"],

    "PA": ["Belem", "Ananindeua", "Santarem",
           "Maraba", "Castanhal"],

    "PB": ["Joao Pessoa", "Campina Grande", "Patos"],

    "PR": ["Curitiba", "Londrina", "Maringa",
           "Cascavel", "Ponta Grossa"],

    "PE": ["Recife", "Caruaru", "Petrolina",
           "Olinda", "Jaboatao dos Guararapes"],

    "PI": ["Teresina", "Parnaiba", "Picos"],

    "RJ": ["Rio de Janeiro", "Niteroi", "Petropolis",
           "Volta Redonda", "Campos dos Goytacazes"],

    "RN": ["Natal", "Mossoro", "Parnamirim"],

    "RS": ["Porto Alegre", "Caxias do Sul", "Pelotas",
           "Santa Maria", "Passo Fundo"],

    "RO": ["Porto Velho", "Ji-Parana", "Ariquemes"],

    "RR": ["Boa Vista", "Rorainopolis", "Caracarai"],

    "SC": ["Florianopolis", "Joinville", "Blumenau",
           "Chapeco", "Criciuma"],

    "SP": ["Sao Paulo", "Campinas", "Santos",
           "Ribeirao Preto", "Sao Jose dos Campos",
           "Sorocaba", "Bauru"],

    "SE": ["Aracaju", "Itabaiana", "Lagarto"],

    "TO": ["Palmas", "Araguaina", "Gurupi"]
}

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

REGION_IDS = {

    "Norte": 1,
    "Nordeste": 2,
    "Centro-Oeste": 3,
    "Sudeste": 4,
    "Sul": 5

}

STATES = list(STATE_CITIES.keys())

# Cadastro geografico normalizado
geography_records = []

for state, cities_list in STATE_CITIES.items():

    for city in cities_list:

        geography_records.append({

            "state": state,

            "city": city,

            "region": STATE_REGION[state],

            "region_id": REGION_IDS[
                STATE_REGION[state]
            ]

        })

geography = pd.DataFrame(
    geography_records
)

# ============================================================
# 4. DIM_REGIONS
# ============================================================

regions = pd.DataFrame({

    "region_id": list(REGION_IDS.values()),

    "region_name": list(REGION_IDS.keys())

})

# ============================================================
# 5. DIM_CUSTOMERS
# ============================================================

customer_ids = np.arange(
    1,
    N_CUSTOMERS + 1
)

# Datas de cadastro anteriores ou durante o historico
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

# Selecionar uma cidade e seu estado simultaneamente
geo_positions = rng.integers(
    0,
    len(geography),
    N_CUSTOMERS
)

customer_geography = (
    geography.iloc[geo_positions]
    .reset_index(drop=True)
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

    "city": customer_geography["city"],

    "state": customer_geography["state"],

    "region": customer_geography["region"],

    "region_id": customer_geography["region_id"],

    "signup_date": signup_dates,

    "customer_segment": rng.choice(
        ["Novo", "Regular", "Fiel", "VIP"],
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

# ============================================================
# 6. DIM_CATEGORIES
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
# 7. DIM_PRODUCTS
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
# 8. DIM_SELLERS
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
# 9. DIM_STORES
# ============================================================

store_ids = np.arange(
    1,
    N_STORES + 1
)

store_geo_positions = rng.integers(
    0,
    len(geography),
    N_STORES
)

store_geography = (
    geography.iloc[store_geo_positions]
    .reset_index(drop=True)
)

stores = pd.DataFrame({

    "store_id": store_ids,

    "store_name": [
        f"Loja Vision {i:02d}"
        for i in store_ids
    ],

    "city": store_geography["city"],

    "state": store_geography["state"],

    "region": store_geography["region"],

    "region_id": store_geography["region_id"],

    "store_type": rng.choice(
        [
            "Loja Fisica",
            "Centro de Distribuicao"
        ],
        N_STORES,
        p=[0.85, 0.15]
    )

})

# ============================================================
# 10. DIM_DATE
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

dim_date["year"] = dim_date["date"].dt.year

dim_date["month"] = dim_date["date"].dt.month

dim_date["day"] = dim_date["date"].dt.day

dim_date["quarter"] = dim_date["date"].dt.quarter

dim_date["month_name"] = dim_date["date"].dt.month_name()

dim_date["year_month"] = dim_date["date"].dt.strftime("%Y-%m")

dim_date["day_of_week"] = dim_date["date"].dt.day_name()

dim_date["is_weekend"] = (
    dim_date["date"].dt.dayofweek >= 5
)

# ============================================================
# 11. DIMENSOES CONSOLIDADAS
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
PARTE 2/3 — Vendas, cancelamentos e calibração financeira
Aqui está a segunda parte, com uma mudança importante em relação ao script anterior: a calibração será feita em centavos, evitando diferenças causadas pelo arredondamento de valores monetários.

Também vamos separar três conceitos:

gross_revenue: receita bruta antes dos descontos.

net_revenue: receita após descontos, antes das devoluções.

net_revenue_after_returns: receita após os reembolsos.

Para garantir a conciliação, a receita e o custo serão distribuídos proporcionalmente entre os itens elegíveis, com ajuste de centavos no último item quando necessário.

Outro cuidado: o custo de referência em dim_products.unit_cost não será confundido com o custo efetivamente reconhecido. O custo comercial realizado estará em fact_sales_items.line_cost.


python

# ============================================================
# VISION GROUP
# PARTE 2/3 - VENDAS E ITENS
# ============================================================

# ============================================================
# 12. FUNCAO DE DISTRIBUICAO MONETARIA EM CENTAVOS
# ============================================================

def allocate_cents(total_value, weights):

    """
    Distribui um valor monetario em centavos.
    A soma dos valores retornados corresponde
    exatamente ao total informado.
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

        weights = np.ones(
            len(weights)
        )

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

        allocated[positions[:remaining]] += 1

    return allocated / 100


# ============================================================
# 13. FACT_SALES - PEDIDOS
# ============================================================

order_ids = np.arange(
    1,
    N_ORDERS + 1
)

random_days = rng.integers(
    0,
    (END_DATE - START_DATE).days + 1,
    N_ORDERS
)

random_order_dates = (
    START_DATE
    + pd.to_timedelta(
        random_days,
        unit="D"
    )
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

# Status dos pedidos
order_statuses = rng.choice(
    [
        "Entregue",
        "Enviado",
        "Processando",
        "Cancelado"
    ],
    N_ORDERS,
    p=[
        0.82,
        0.08,
        0.06,
        0.04
    ]
)

canceled_mask = (
    order_statuses == "Cancelado"
)

# Datas de entrega
delivery_days = rng.integers(
    2,
    15,
    N_ORDERS
)

delivered_dates = (
    order_dates
    + pd.to_timedelta(
        delivery_days,
        unit="D"
    )
)

delivered_dates = pd.Series(
    delivered_dates
)

delivered_dates.loc[
    ~np.isin(
        order_statuses,
        ["Entregue"]
    )
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

# Associar geografia da loja
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
# 14. FACT_SALES_ITEMS - GERACAO DOS ITENS
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

# Descontos comerciais
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

# Valores comerciais iniciais
initial_prices = rng.uniform(
    30,
    350,
    N_ITEMS
)

initial_gross = (
    quantities * initial_prices
)

initial_net = (
    initial_gross
    * (1 - discount_rates)
)

# Identificar pedidos cancelados nos itens
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

eligible_positions = np.flatnonzero(
    item_eligible_mask
)

# ============================================================
# 15. CALIBRAR RECEITA COMERCIAL
# ============================================================

# Distribuir R$ 8,2 milhoes entre os itens elegiveis
eligible_revenue = allocate_cents(
    TARGET_REVENUE,
    initial_net[eligible_positions]
)

# Inicializar receita dos itens
net_revenue = np.zeros(
    N_ITEMS
)

net_revenue[eligible_positions] = (
    eligible_revenue
)

# Reconstruir receita bruta e descontos
# para preservar a relacao comercial por item
gross_revenue = np.zeros(
    N_ITEMS
)

discount_value = np.zeros(
    N_ITEMS
)

for position in eligible_positions:

    net_cents = int(
        round(
            net_revenue[position] * 100
        )
    )

    rate = discount_rates[position]

    gross_cents = int(
        np.ceil(
            net_cents / (1 - rate)
        )
    )

    gross_revenue[position] = (
        gross_cents / 100
    )

    discount_value[position] = (
        gross_cents - net_cents
    ) / 100

# ============================================================
# 16. CALIBRAR CUSTO COMERCIAL
# ============================================================

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

# Distribuir custo total desejado
line_cost = np.zeros(
    N_ITEMS
)

line_cost[eligible_positions] = (
    allocate_cents(
        TARGET_COST,
        initial_item_cost[eligible_positions]
    )
)

# Custo unitario realizado
adjusted_unit_costs = np.divide(
    line_cost,
    quantities,
    out=np.zeros(N_ITEMS),
    where=quantities != 0
)

# Preco unitario comercial de referencia
unit_prices = np.divide(
    gross_revenue,
    quantities,
    out=np.zeros(N_ITEMS),
    where=quantities != 0
)

# ============================================================
# 17. ZERAR ITENS DE PEDIDOS CANCELADOS
# ============================================================

gross_revenue[item_canceled_mask] = 0.00

discount_value[item_canceled_mask] = 0.00

net_revenue[item_canceled_mask] = 0.00

line_cost[item_canceled_mask] = 0.00

adjusted_unit_costs[item_canceled_mask] = 0.00

unit_prices[item_canceled_mask] = 0.00

# Lucro bruto antes das devolucoes
gross_profit = (
    net_revenue - line_cost
).round(2)

# Margem bruta
gross_margin = np.divide(
    gross_profit,
    net_revenue,
    out=np.zeros(N_ITEMS),
    where=net_revenue != 0
)

# ============================================================
# 18. FACT_SALES_ITEMS
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

promotion_ids[discounted_positions] = (
    rng.choice(
        [
            "PROMO-001",
            "PROMO-002",
            "PROMO-003",
            "PROMO-004",
            "PROMO-005"
        ],
        len(discounted_positions)
    )
)

items = pd.DataFrame({

    "item_id": item_ids,

    "order_id": item_order_ids,

    "product_id": item_product_ids,

    "quantity": quantities,

    "unit_price": np.round(
        unit_prices,
        4
    ),

    "unit_cost": np.round(
        adjusted_unit_costs,
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
# 19. CONSOLIDAR ITENS EM FACT_SALES
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

# Campos comerciais adicionais
sales["is_returned"] = False

sales["total_refund"] = 0.00

sales["net_revenue_after_returns"] = (
    sales["order_revenue"]
)

sales["gross_profit_after_returns"] = (
    sales["order_gross_profit"]
)

# ============================================================
# 20. TABELAS COMERCIAIS
# ============================================================

facts_commercial = {

    "fact_sales": sales,

    "fact_sales_items": items

}
O que mudou financeiramente?
A distribuição agora utiliza valores monetários em centavos. Isso evita que a soma dos itens fique diferente da meta por causa de arredondamentos independentes.

O custo também é distribuído considerando somente os pedidos elegíveis.

Dessa forma, a relação financeira será:

8.200.000
−
6.100.000
=
2.100.000
8.200.000−6.100.000=2.100.000
E os indicadores serão calculados assim:

M
a
r
g
e
m
 
B
r
u
t
a
=
2.100.000
8.200.000
=
25
,
61
%
Margem Bruta= 
8.200.000
2.100.000
​
 =25,61%
Os pedidos cancelados continuam na base, mas não entram no faturamento comercial.

PARTE 3/3 — Devoluções, clientes, estoque, marketing e financeiro
Esta última parte corrige também a interpretação das informações geradas.

As principais mudanças são:

O reembolso será registrado por produto, sem rateio arbitrário do valor total do pedido.

A receita líquida de devoluções será calculada no pedido e no item.

A dimensão de clientes terá informações para distinguir compradores de clientes apenas cadastrados.

O marketing terá campos explícitos indicando que a atribuição é sintética.

O estoque será identificado como fotografia mensal.

O financeiro terá receitas, custos e reembolsos organizados por competência.

Será criado um arquivo business_rules.csv documentando as premissas do gerador.

4. Dois pequenos ajustes finais importantes
Antes de executar, faça estas duas substituições na parte 3.

Ajuste A — Identificação de pedidos devolvidos
Atualmente, is_returned é definido pelo valor do reembolso. Isso pode deixar uma devolução em processamento identificada incorretamente como não devolvida.

Substitua:


python
sales["is_returned"] = (
    sales["total_refund"] > 0
)
Por:


python
returned_order_set = set(
    returns["order_id"]
)

sales["is_returned"] = (
    sales["order_id"].isin(returned_order_set)
)
Ajuste B — Fluxo de caixa
Para evitar contabilizar o reembolso duas vezes, substitua:


python
financial["cash_inflow"] = (
    financial["net_revenue_after_returns"]
    * rng.uniform(
        0.90,
        1.00,
        len(financial)
    )
).round(2)
Por:


python
financial["cash_inflow"] = (
    financial["revenue"]
    * rng.uniform(
        0.90,
        1.00,
        len(financial)
    )
).round(2)
O reembolso já está registrado em cash_outflow, portanto não deve ser descontado novamente das entradas.

5. Estrutura final dos arquivos
vision_group_dataset/raw/

Dimensões

dim_customers.csv

dim_products.csv

dim_categories.csv

dim_sellers.csv

dim_stores.csv

dim_regions.csv

dim_date.csv

Tabelas fato

fact_sales.csv

fact_sales_items.csv

fact_returns.csv

fact_returns_items.csv

fact_inventory.csv

fact_marketing.csv

fact_financial.csv

fact_targets.csv

Documentação

business_rules.csv

dataset_metadata.csv

6. Regras de interpretação para o futuro dashboard
Estas serão as definições que recomendo preservar nas próximas etapas do projeto:

Indicador

Campo correto

Receita comercial

fact_sales.order_revenue

Custo comercial

fact_sales.order_cost

Lucro bruto

fact_sales.order_gross_profit

Reembolsos

fact_sales.total_refund

Receita líquida de devoluções

fact_sales.net_revenue_after_returns

Lucro após devoluções

fact_sales.gross_profit_after_returns

Clientes compradores

dim_customers.is_buyer

Pedidos cancelados

fact_sales.is_canceled

Pedidos comerciais

fact_sales.is_commercial

Quantidade devolvida

fact_returns_items.returned_quantity

Estoque

fact_inventory.stock_quantity

Atenção a uma premissa: o lucro após devoluções mantém o custo original dos produtos vendidos, sem presumir recuperação do custo dos itens devolvidos. Isso é uma aproximação conservadora, não uma apuração contábil completa.

7. Como executar
Salve as três partes no mesmo arquivo vision_group.py, na ordem apresentada, incluindo os dois ajustes finais.

Execute:


bash
pip install pandas numpy
python vision_group.py
O resultado será um dataset sintético documentado, com receita e lucro bruto calibrados, devoluções identificadas por produto, geografia coerente e distinção entre clientes cadastrados e compradores.

Uma ressalva importante: o gerador define 82.400 clientes cadastrados, mas não garante 82.400 compradores únicos. Essa distinção é intencional e permite analisar a conversão da base de clientes.

A próxima etapa natural será construir o modelo SQL em estrela e as seis views executivas utilizando essas tabelas como fonte.


Isso também pode interessar a você.
Saiba mais


Bluertec
Consultoria em Dados, BI e Data Lake
Engenharia de dados, data warehouse e dashboards sob medida
Anúncio

refaça o script com ele todo em ingles

Pensou por 48s
Vou reescrever o gerador completo em inglês, mantendo a estrutura do projeto Vision Group e os ajustes financeiros e operacionais que definimos anteriormente.

A padronização será aplicada a:

Nomes das variáveis Python.

Nomes das tabelas e colunas.

Valores categóricos (status, canais, regiões etc.).

Nomes dos arquivos CSV.

Comentários e documentação técnica.

As premissas principais serão mantidas:

Parâmetro

Valor

Company

Vision Group

Registered customers

82,400

Orders

145,200

Target revenue

R$ 8,200,000

Target gross profit

R$ 2,100,000

Target gross margin

25.61%

Historical period

2021–2025

Country

Brazil

O código abaixo será dividido em três partes sequenciais. Salve tudo no mesmo arquivo vision_group.py, respeitando a ordem.

PART 1/3 — Configuration, dimensions and reference data
Nesta parte criamos as configurações, cidades brasileiras vinculadas corretamente aos estados, regiões comerciais, clientes, produtos, vendedores, lojas, categorias e calendário.


python

# ============================================================
# VISION GROUP - EXECUTIVE BI DATASET
# PART 1/3 - CONFIGURATION AND DIMENSIONS
# ============================================================

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

print("Part 1 completed: dimensions created.")
PART 2/3 — Sales, items and financial calibration
Nesta parte criamos os pedidos e seus respectivos itens.

A receita e o lucro bruto serão calibrados exclusivamente nos pedidos não cancelados. Os itens cancelados terão receita, desconto e custo iguais a zero.

Também criaremos os campos necessários para calcular o desempenho comercial posteriormente.


python

# ============================================================
# VISION GROUP - EXECUTIVE BI DATASET
# PART 2/3 - SALES AND FINANCIAL CALIBRATION
# ============================================================

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

print("Part 2 completed: sales and items created.")
PART 3/3 — Returns, inventory, marketing and financial tables
Nesta parte finalizamos o banco de dados consolidado.

As devoluções terão identificação individual por produto e pedido. Os reembolsos serão descontados da receita comercial para produzir a receita líquida após devoluções.

Também serão criadas as tabelas de estoque, marketing, financeiro e metas.

Como executar o projeto
1. Instale as bibliotecas

bash
pip install pandas numpy
2. Salve o arquivo
Nome:

vision_group.py

Cole as três partes na sequência, sem criar três arquivos separados.

3. Execute

bash
python vision_group.py
4. Estrutura de saída
vision_group_dataset/

raw/

DIMENSIONS

dim_customers.csv

dim_products.csv

dim_categories.csv

dim_sellers.csv

dim_stores.csv

dim_regions.csv

dim_date.csv

FACT TABLES

fact_sales.csv

fact_sales_items.csv

fact_returns.csv

fact_returns_items.csv

fact_inventory.csv

fact_marketing.csv

fact_financial.csv

fact_targets.csv

DOCUMENTATION

business_rules.csv

dataset_metadata.csv

Observações importantes sobre os números
O gerador mantém exatamente:

82.400 clientes cadastrados.

145.200 pedidos.

R$ 8.200.000 de receita comercial antes das devoluções.

R$ 6.100.000 de custo comercial.

R$ 2.100.000 de lucro bruto.

Margem bruta de aproximadamente 25,61%.

Os valores de receita e lucro são metas sintéticas calibradas, não resultados de uma operação real.

Importante: o script gera receita e custo exatos na soma dos itens, mas unit_price e unit_cost são campos arredondados para quatro casas decimais. Para conciliações financeiras, utilize sempre net_revenue e line_cost, não tente reconstruir esses totais multiplicando os campos unitários.

Também vale lembrar que o estoque é uma fotografia mensal sintética e o EBITDA é uma estimativa gerencial, não um demonstrativo contábil auditado.

Com isso, você terá a base em inglês pronta para avançar para a camada SQL, as seis views analíticas e o dashboard executivo no Power BI.



me mande a parte 3/3 dividida em 2 partes

Claro! Vou dividir a Parte 3/3 em duas subpartes, mantendo todo o código em inglês e preservando a sequência de execução.

Parte 3A/3: Devoluções, atualização das vendas e clientes, estoque e marketing.

Parte 3B/3: Dados financeiros, metas, regras de negócio, metadados e exportação dos arquivos.

Cole a Parte 3B imediatamente depois da Parte 3A no mesmo arquivo Python.

Parte 3A/3 — Returns, Inventory and Marketing
Primeira metade

python
# ============================================================
# VISION GROUP - EXECUTIVE BI DATASET
# PART 3A/3 - RETURNS, INVENTORY AND MARKETING
# ============================================================

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
Parte 3B/3 — Financial Data, Targets and Export
Segunda metade

python
# ============================================================
# VISION GROUP - EXECUTIVE BI DATASET
# PART 3B/3 - FINANCIAL DATA, TARGETS AND EXPORT
# ============================================================

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
Atenção antes de executar
As duas subpartes dependem das variáveis criadas nas Partes 1 e 2, incluindo:

TARGET_REVENUE

TARGET_GROSS_PROFIT

TARGET_MARGIN

N_ORDERS

N_CUSTOMERS

N_PRODUCTS

N_STORES

N_SELLERS

RETURN_RATE

CANCELLATION_RATE

BUSINESS_RULES

dimensions

RAW_DIR

Uma observação técnica importante: o código acima mantém a lógica do gerador anterior. Para uma versão definitiva, há dois pontos que merecem correção antes de usar os dados no BI:

O estoque mensal de 16 anos, 5.000 produtos e várias lojas pode gerar milhões de registros.

O EBITDA e o fluxo de caixa são simulações, não demonstrações financeiras reconciliadas.

Além disso, a geração de devoluções deve limitar a quantidade acumulada devolvida por item à quantidade originalmente comprada caso um mesmo item possa aparecer em mais de uma devolução.


Saiba mais


SENSEDIA S.A.
Governe APIs, Integrações e agentes
Governança para APIs, integrações e IA. Mais controle, segurança e escala para sua arquitetura.
Anúncio

faça as seguintes correções no codigo: 3.1. Atenção: os valores comerciais são calibrados
Seu script utiliza a função:
def allocate_money(total_value, weights):    total_cents = int(        round(total_value * 100)    )    weights = np.asarray(        weights,        dtype=float    )    weights = np.maximum(        weights,        0    )    if weights.sum() == 0:        weights = np.ones(len(weights))    exact_values = (        weights / weights.sum()    ) * total_cents    allocated = np.floor(        exact_values    ).astype(np.int64)    remaining = (        total_cents - allocated.sum()    )    fractions = (        exact_values - allocated    )    if remaining > 0:



Essa função distribui um valor monetário entre os registros, respeitando os centavos.

Ela garante que a soma distribuída seja igual ao valor original.

Por exemplo:
values = allocate_money(    1000.00,    [10, 20, 30, 40])print(values)print(values.sum())



O resultado será uma distribuição proporcional, totalizando exatamente R$ 1.000,00.

Isso é útil para construir uma base sintética controlada.

Entretanto: os valores de receita e custo são artificialmente calibrados para atingir os objetivos definidos. Portanto, a base é adequada para demonstrar técnicas analíticas, mas não representa uma simulação independente de vendas reais.
3.2. Correção importante: custo dos produtos
Existe um ponto no seu script que merece atenção.

Os custos originais dos produtos são gerados em:
products["unit_cost"]



Mas, posteriormente, o custo comercial é distribuído globalmente:
line_cost[    eligible_positions] = allocate_money(    TARGET_COST,    initial_item_cost[        eligible_positions    ])



Isso faz com que o custo realizado de cada item seja ajustado proporcionalmente para atingir R$ 6,1 milhões.

Consequentemente, o unit_cost da tabela de produtos não corresponde necessariamente ao custo efetivo dos itens vendidos.

Para o dashboard de rentabilidade, utilizaremos:
fact_sales_items["line_cost"]



como custo comercial efetivamente registrado na venda.

E utilizaremos:
fact_sales_items["unit_cost"]



como custo unitário realizado, quando estivermos trabalhando com a tabela de itens.

Não devemos substituir esse custo pelo custo de referência da dimensão de produtos.
3.3. Correção: devoluções e margem
Seu script calcula:
items["net_revenue_after_returns"] = (    items["net_revenue"]    - items["refund_value"])items["gross_profit_after_returns"] = (    items["net_revenue_after_returns"]    - items["line_cost"])



Isso significa que o custo original continua integralmente reconhecido mesmo quando o produto é devolvido.

Essa é uma hipótese simplificadora.

Na contabilidade real, precisamos saber se o produto devolvido voltou ao estoque, se foi perdido ou se precisou ser descartado.

Para este projeto, documentaremos que o custo não é revertido automaticamente nas devoluções.
3.4. Correção: taxa de devolução
O código faz:
N_RETURNS = int(    round(N_ORDERS * RETURN_RATE))



Como RETURN_RATE = 0.10, são geradas aproximadamente 14.520 ocorrências de devolução.

Entretanto, elas são selecionadas apenas entre pedidos entregues.

Portanto, a taxa efetiva sobre pedidos entregues será maior que 10%.

Para apresentar corretamente o indicador, usaremos:
\[
\text{Taxa de devolução} =
\frac{\text{Pedidos devolvidos}}
{\text{Pedidos entregues}}
\times100
\]

E não simplesmente o parâmetro RETURN_RATE.
3.5. Correção: metas mensais
O seu script distribui uniformemente as metas:
targets["revenue_target"] = (    TARGET_REVENUE / number_of_months)targets["gross_profit_target"] = (    TARGET_GROSS_PROFIT / number_of_months)



Isso significa que cada mês recebe a mesma meta.

É uma hipótese válida para demonstrar o funcionamento do dashboard, mas não considera a sazonalidade.

Em uma empresa real, dezembro, por exemplo, poderia ter uma meta maior que fevereiro.

Também é importante observar que a meta total de receita representa R$ 8,2 milhões, enquanto os valores efetivamente realizados depois das devoluções serão menores.

Vamos manter a distinção entre:
- Receita comercial antes das devoluções.
- Reembolsos aprovados.
- Receita comercial líquida de devoluções.
- Meta de receita comercial.
Assim, a comparação será feita entre indicadores equivalentes.
3.6. Correção: datas de devolução
Seu script permite que a devolução aconteça até 30 dias depois da entrega.

Como consequência, algumas devoluções de pedidos de dezembro de 2025 podem acontecer em janeiro de 2026.

Isso não é necessariamente um erro.

Mas precisamos considerar isso na análise temporal: a receita é reconhecida pela data do pedido, enquanto o reembolso é registrado pela data da devolução.

Não devemos misturar as duas datas sem documentar o critério.
3.7. Marketing e financeiro
O script possui duas regras importantes:
marketing["is_verified_attribution"] = False



E:
financial["data_type"] = (    "Synthetic Financial Simulation")



Portanto:
- O ROAS é uma estimativa sintética, não uma atribuição comprovada.
- O EBITDA é simulado.
- O fluxo de caixa é simulado.
- As despesas não representam lançamentos contábeis auditados.
Essas informações deverão aparecer na documentação do projeto.

Vou incorporar as sete correções ao gerador da Vision Group, mantendo o código integralmente em inglês e preservando a estrutura das três partes anteriores.

A ideia é manter a base adequada para um projeto de Executive BI, mas deixar explícitas as hipóteses de simulação, principalmente nas áreas comercial, financeira e de devoluções.

O que será corrigido
Item

Correção

3.1

Distribuição monetária com precisão de centavos por allocate_money()

3.2

Separação entre custo de referência e custo comercial realizado

3.3

Custo original mantido após devoluções, sem reversão automática

3.4

Taxa de devolução calculada sobre pedidos entregues

3.5

Metas mensais uniformes, comparadas com receita comercial antes dos reembolsos

3.6

Devoluções registradas pela data efetiva da devolução

3.7

Marketing, EBITDA e fluxo de caixa identificados como dados sintéticos

Também vou incluir essas regras na tabela business_rules, para que sejam exportadas junto com o dataset.

1. Correção da função allocate_money
Coloque esta função na Parte 1, junto às funções auxiliares do gerador.

Ela distribui valores monetários proporcionalmente e garante que a soma final seja exatamente igual ao valor solicitado, respeitando os centavos.


python
# ============================================================
# MONEY ALLOCATION FUNCTION
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

    remaining = (
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
        np.asarray(values, dtype=np.int64)
        / 100
    ).round(2)
Importante: a função retorna valores inteiros em centavos. Por isso, utilize cents_to_currency() quando precisar transformar os valores em reais.

2. Correção da calibração comercial
Substitua o bloco antigo de calibração de receita e custo por este.

O objetivo é:

Calibrar a receita em R$ 8,2 milhões.

Calibrar o custo comercial em R$ 6,1 milhões.

Calibrar o lucro bruto em R$ 2,1 milhões.

Excluir pedidos cancelados da distribuição.

Zerar receita e custo dos itens cancelados.

Manter os custos de referência dos produtos separados dos custos realizados.


python
# ============================================================
# COMMERCIAL FINANCIAL CALIBRATION
# ============================================================

# Identify canceled orders
canceled_order_ids = set(
    sales.loc[
        sales["status"] == "Canceled",
        "order_id"
    ]
)

# Identify eligible sales items
eligible_mask = ~items["order_id"].isin(
    canceled_order_ids
)

eligible_positions = np.flatnonzero(
    eligible_mask.to_numpy()
)

canceled_positions = np.flatnonzero(
    ~eligible_mask.to_numpy()
)

# Initialize commercial values
items["net_revenue"] = 0.00
items["line_cost"] = 0.00

# ------------------------------------------------------------
# REVENUE CALIBRATION
# ------------------------------------------------------------

initial_item_revenue = (
    items["quantity"].to_numpy()
    * items["price"].to_numpy()
    - items["discount"].to_numpy()
    + items["freight_value"].to_numpy()
)

allocated_revenue_cents = allocate_money(
    TARGET_REVENUE,
    initial_item_revenue[eligible_positions]
)

items.loc[
    eligible_mask,
    "net_revenue"
] = cents_to_currency(
    allocated_revenue_cents
)

# ------------------------------------------------------------
# COST CALIBRATION
# ------------------------------------------------------------

initial_item_cost = (
    items["quantity"].to_numpy()
    * items["unit_cost_reference"].to_numpy()
)

allocated_cost_cents = allocate_money(
    TARGET_COST,
    initial_item_cost[eligible_positions]
)

items.loc[
    eligible_mask,
    "line_cost"
] = cents_to_currency(
    allocated_cost_cents
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

# ------------------------------------------------------------
# REALIZED UNIT COST
# ------------------------------------------------------------

items["unit_cost"] = np.divide(
    items["line_cost"],
    items["quantity"],
    out=np.zeros(len(items)),
    where=items["quantity"].to_numpy() != 0
).round(2)

# ------------------------------------------------------------
# GROSS PROFIT
# ------------------------------------------------------------

items["gross_profit"] = (
    items["net_revenue"]
    - items["line_cost"]
).round(2)

items["gross_margin"] = np.divide(
    items["gross_profit"],
    items["net_revenue"],
    out=np.zeros(len(items)),
    where=items["net_revenue"].to_numpy() != 0
)

# ------------------------------------------------------------
# COMMERCIAL CLASSIFICATION
# ------------------------------------------------------------

items["is_canceled"] = (
    items["order_id"].isin(
        canceled_order_ids
    )
)

items["is_commercial"] = (
    ~items["is_canceled"]
)
Atenção à dimensão de produtos: mantenha products["unit_cost"] como custo de referência. No código acima, unit_cost_reference representa esse custo original previamente associado aos itens por meio de um relacionamento com a dimensão de produtos.

Caso sua Parte 2 ainda não tenha criado essa coluna, faça o relacionamento antes da calibração:


python
items = items.merge(
    products[
        [
            "product_id",
            "unit_cost"
        ]
    ].rename(
        columns={
            "unit_cost": "unit_cost_reference"
        }
    ),
    on="product_id",
    how="left",
    validate="many_to_one"
)
Assim teremos duas informações distintas:

unit_cost_reference: custo original cadastrado no produto.

unit_cost: custo unitário comercial realizado.

line_cost: custo total realizado do item vendido.

O dashboard de rentabilidade deverá utilizar line_cost como custo comercial efetivo.

3. Correção do cabeçalho de vendas
Depois da calibração dos itens, substitua a geração dos totais de fact_sales por:


python
# ============================================================
# UPDATE FACT_SALES
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

sales["is_canceled"] = (
    sales["status"] == "Canceled"
)

sales["is_commercial"] = (
    ~sales["is_canceled"]
)

sales["is_returned"] = False

sales["total_refund"] = 0.00

sales["net_revenue_after_returns"] = (
    sales["order_revenue"]
)

sales["gross_profit_after_returns"] = (
    sales["order_gross_profit"]
)
Dessa forma, os valores do cabeçalho são derivados dos itens, e não gerados independentemente.

4. Correção da quantidade de devoluções
Substitua o cálculo de N_RETURNS na Parte 3A.

A taxa de devolução será calculada sobre pedidos efetivamente entregues.


python
# ============================================================
# RETURN RATE CALCULATION
# ============================================================

delivered_orders = sales.loc[
    sales["status"] == "Delivered",
    "order_id"
].to_numpy()

N_DELIVERED_ORDERS = len(
    delivered_orders
)

# RETURN_RATE represents the target rate
# among delivered orders.

N_RETURNS = int(
    round(
        N_DELIVERED_ORDERS
        * RETURN_RATE
    )
)

N_RETURNS = min(
    N_RETURNS,
    N_DELIVERED_ORDERS
)

selected_return_orders = rng.choice(
    delivered_orders,
    size=N_RETURNS,
    replace=False
)
Agora, se 138.940 pedidos forem entregues e a taxa desejada for 10%, teremos aproximadamente 13.894 pedidos devolvidos.

A fórmula utilizada no dashboard será:

Return Rate
=
Returned Orders
Delivered Orders
×
100
Return Rate= 
Delivered Orders
Returned Orders
​
 ×100
5. Correção das devoluções por produto
Substitua a geração anterior de fact_returns e fact_returns_items por esta versão.

Ela mantém o relacionamento entre devolução, pedido e produto, sem distribuir arbitrariamente o reembolso entre produtos.

Também impede que a quantidade devolvida ultrapasse a quantidade comprada, inclusive quando um item tiver mais de um registro de devolução.


python
# ============================================================
# FACT_RETURNS AND FACT_RETURNS_ITEMS
# ============================================================

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

# Track quantities already returned
returned_quantity_tracker = {}

for return_id, order_id in zip(
    return_ids,
    selected_return_orders
):

    order_items = items_by_order[
        order_id
    ]

    available_items = []

    for _, item in order_items.iterrows():

        item_id = int(
            item["item_id"]
        )

        purchased_quantity = int(
            item["quantity"]
        )

        already_returned = returned_quantity_tracker.get(
            item_id,
            0
        )

        remaining_quantity = (
            purchased_quantity
            - already_returned
        )

        if remaining_quantity > 0:

            available_items.append({
                "item_id": item_id,
                "product_id": int(item["product_id"]),
                "remaining_quantity": remaining_quantity,
                "purchased_quantity": purchased_quantity,
                "net_revenue": float(item["net_revenue"])
            })

    if not available_items:
        continue

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

    delivery_date = pd.Timestamp(
        delivery_map.loc[order_id]
    )

    # Return dates can extend into January 2026
    return_date = (
        delivery_date
        + pd.Timedelta(
            days=int(rng.integers(1, 31))
        )
    )

    return_reason = rng.choice(
        RETURN_REASONS
    )

    return_status = rng.choice(
        ["Approved", "Processing"],
        p=[0.85, 0.15]
    )

    total_refund = 0.00

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
            rng.uniform(0.70, 1.00)
        )

        refund_value = 0.00

        if return_status == "Approved":

            refund_value = round(
                original_item_revenue
                * refund_percentage,
                2
            )

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

returns = pd.DataFrame(
    return_records
)

returns_items = pd.DataFrame(
    return_item_records
)
Atualização dos itens

python
# ============================================================
# UPDATE FACT_SALES_ITEMS AFTER RETURNS
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

# Revenue after approved refunds
items["net_revenue_after_returns"] = (
    items["net_revenue"]
    - items["refund_value"]
).round(2)

# Original commercial cost remains recognized.
# No automatic inventory recovery is assumed.

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
Atualização do cabeçalho de vendas

python
# ============================================================
# UPDATE SALES AFTER RETURNS
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

sales["is_returned"] = (
    sales["order_id"].isin(
        returns["order_id"]
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
Regra contábil documentada: o custo comercial original não é revertido automaticamente quando ocorre uma devolução. O projeto não simula a recuperação física do produto no estoque.

6. Correção das datas financeiras
Aqui temos uma distinção importante:

A receita é reconhecida pela data do pedido.

O reembolso é reconhecido pela data da devolução.

Uma devolução de dezembro de 2025 pode gerar um reembolso em janeiro de 2026.

Por isso, substitua a criação do calendário financeiro da Parte 3B pelo seguinte:


python
# ============================================================
# FINANCIAL CALENDAR
# ============================================================

last_return_date = (
    returns["return_date"].max()
    if not returns.empty
    else END_DATE
)

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

# Revenue recognized by order date
monthly_revenue = (
    sales.loc[
        sales["is_commercial"]
    ]
    .groupby(
        sales.loc[
            sales["is_commercial"],
            "order_date"
        ].dt.to_period("M")
    )["order_revenue"]
    .sum()
)

# Product costs recognized by order date
monthly_cost = (
    sales.loc[
        sales["is_commercial"]
    ]
    .groupby(
        sales.loc[
            sales["is_commercial"],
            "order_date"
        ].dt.to_period("M")
    )["order_cost"]
    .sum()
)

# Refunds recognized by return date
monthly_refunds = (
    returns.loc[
        returns["return_status"] == "Approved"
    ]
    .groupby(
        returns.loc[
            returns["return_status"] == "Approved",
            "return_date"
        ].dt.to_period("M")
    )["refund_value"]
    .sum()
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
As demais colunas de despesas, EBITDA e fluxo de caixa podem permanecer na sequência do bloco financeiro anterior.

Observação: janeiro de 2026 poderá aparecer na tabela financeira apenas com reembolsos, sem receita comercial correspondente. Isso é intencional e representa o critério temporal definido.

7. Correção das metas mensais
Mantenha as metas uniformes, mas deixe explícito que são metas de receita comercial antes das devoluções.

Substitua o bloco de fact_targets por:


python
# ============================================================
# FACT_TARGETS - MONTHLY TARGETS
# ============================================================

targets = pd.DataFrame({
    "month": pd.date_range(
        START_DATE,
        END_DATE,
        freq="MS"
    )
})

number_of_months = len(targets)

targets["revenue_target"] = (
    TARGET_REVENUE / number_of_months
).round(2)

targets["gross_profit_target"] = (
    TARGET_GROSS_PROFIT / number_of_months
).round(2)

targets["orders_target"] = (
    N_ORDERS / number_of_months
).round(2)

targets["margin_target"] = TARGET_MARGIN

targets["revenue_target_type"] = (
    "Commercial Revenue Before Returns"
)

targets["target_method"] = (
    "Uniform Monthly Distribution"
)

targets["seasonality_adjusted"] = False
Dessa maneira, as comparações do dashboard serão:

Indicador

Realizado

Meta

Receita comercial

order_revenue

revenue_target

Lucro bruto

order_gross_profit

gross_profit_target

Margem bruta

gross_profit / revenue

margin_target

Não compare a meta de receita comercial com a receita líquida de devoluções.

8. Correção do marketing
Mantenha o ROAS como estimativa sintética, sem atribuição comprovada.

Acrescente estas colunas ao bloco de marketing:


python
# ============================================================
# MARKETING DATA CLASSIFICATION
# ============================================================

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
Isso evita que a diretoria interprete o ROAS como resultado de uma mensuração real de conversões.

9. Correção do financeiro
Acrescente ao final da geração de fact_financial:


python
# ============================================================
# FINANCIAL DATA CLASSIFICATION
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
Isso deixa explícito que:

EBITDA é simulado.

Fluxo de caixa é simulado.

Despesas são estimativas.

Os números não representam lançamentos contábeis auditados.

10. Atualização das regras de negócio
Na Parte 1, acrescente as seguintes regras ao dicionário BUSINESS_RULES:


python
# ============================================================
# ADDITIONAL BUSINESS RULES
# ============================================================

BUSINESS_RULES.update({

    "commercial_revenue_calibration":
        "Commercial revenue is synthetically calibrated to the configured target using proportional monetary allocation in cents.",

    "commercial_cost_calibration":
        "Commercial product costs are synthetically calibrated to the configured target. Actual transaction costs are stored in fact_sales_items.line_cost.",

    "reference_product_cost":
        "dim_products.unit_cost represents the original reference cost and does not necessarily match the realized commercial unit cost.",

    "realized_unit_cost":
        "fact_sales_items.unit_cost represents the realized unit cost calculated from line_cost divided by quantity.",

    "canceled_orders":
        "Canceled orders are excluded from commercial KPIs. Their sales item revenue and commercial costs are zero.",

    "gross_profit":
        "Gross profit is defined as commercial revenue minus realized commercial product cost, before returns.",

    "returns_methodology":
        "Returns are recorded at product-item level through fact_returns_items.",

    "return_rate":
        "Return rate is calculated as returned orders divided by delivered orders.",

    "refund_recognition":
        "Approved refunds are recognized using the actual return date, while commercial revenue is recognized using the order date.",

    "return_cost_treatment":
        "Original product costs are not automatically reversed when a return occurs. Inventory recovery and product disposal are not simulated.",

    "monthly_targets":
        "Monthly revenue and gross profit targets are distributed uniformly across the historical period, without seasonal adjustment.",

    "target_comparison":
        "Commercial revenue targets are compared against commercial revenue before returns. Refunds are reported separately.",

    "marketing_attribution":
        "Marketing attributed revenue and ROAS are synthetic estimates and do not represent verified marketing attribution.",

    "financial_data":
        "Financial expenses, EBITDA and cash flow are synthetic simulations and are not audited accounting records.",

    "inventory_methodology":
        "Inventory data represents synthetic monthly stock snapshots, not actual stock movements.",

    "customer_classification":
        "Registered customers are distinguished from customers with at least one commercial purchase.",

    "financial_period":
        "Commercial revenue and product costs are recognized by order date, while approved refunds are recognized by return date.",

    "dataset_nature":
        "The dataset is synthetic and designed for analytical demonstration, not as an independent representation of actual retail transactions."

})
11. Atualização do resumo do dataset
Recomendo substituir o resumo anterior por este:


python
# ============================================================
# FINAL DATASET SUMMARY
# ============================================================

commercial_sales = sales.loc[
    sales["is_commercial"]
]

total_revenue = (
    commercial_sales["order_revenue"].sum()
)

total_cost = (
    commercial_sales["order_cost"].sum()
)

total_gross_profit = (
    commercial_sales["order_gross_profit"].sum()
)

total_refunds = (
    returns.loc[
        returns["return_status"] == "Approved",
        "refund_value"
    ].sum()
)

delivered_count = (
    (sales["status"] == "Delivered").sum()
)

returned_count = (
    sales["is_returned"].sum()
)

effective_return_rate = (
    returned_count / delivered_count
    if delivered_count > 0
    else 0
)

print("\n" + "=" * 65)
print("VISION GROUP - DATASET GENERATION COMPLETED")
print("=" * 65)

print(f"Registered customers: {len(customers):,}")

print(
    f"Purchasing customers: "
    f"{customers['is_buyer'].sum():,}"
)

print(f"Total orders: {len(sales):,}")

print(
    f"Commercial orders: "
    f"{sales['is_commercial'].sum():,}"
)

print(
    f"Canceled orders: "
    f"{sales['is_canceled'].sum():,}"
)

print(
    f"Delivered orders: "
    f"{delivered_count:,}"
)

print(
    f"Returned orders: "
    f"{returned_count:,}"
)

print(
    f"Effective return rate: "
    f"{effective_return_rate:.2%}"
)

print("\nCOMMERCIAL PERFORMANCE")

print(
    f"Commercial revenue: "
    f"R$ {total_revenue:,.2f}"
)

print(
    f"Commercial product cost: "
    f"R$ {total_cost:,.2f}"
)

print(
    f"Gross profit: "
    f"R$ {total_gross_profit:,.2f}"
)

print(
    f"Gross margin: "
    f"{total_gross_profit / total_revenue:.2%}"
    if total_revenue > 0
    else "0.00%"
)

print(
    f"Approved refunds: "
    f"R$ {total_refunds:,.2f}"
)

print(
    f"Commercial revenue after refunds: "
    f"R$ {total_revenue - total_refunds:,.2f}"
)

print("\nDATA CLASSIFICATION")

print("Dataset: Synthetic")

print("Marketing attribution: Unverified")

print("Financial statements: Simulated")

print("Inventory: Monthly synthetic snapshots")

print("Targets: Uniform monthly distribution")

print("\nOUTPUT DIRECTORY")

print(os.path.abspath(RAW_DIR))

print("\nDATASET GENERATION FINISHED!")
12. Estrutura final do dataset
Após essas alterações, a estrutura continuará organizada em dimensões e fatos.

Tabela

Conteúdo

dim_customers

Cadastro e classificação de clientes

dim_products

Produtos e custos de referência

dim_categories

Categorias

dim_sellers

Vendedores

dim_stores

Lojas

dim_regions

Regiões

dim_date

Calendário

fact_sales

Cabeçalho das vendas

fact_sales_items

Itens e custos realizados

fact_returns

Devoluções por pedido

fact_returns_items

Devoluções por produto

fact_inventory

Fotografias mensais do estoque

fact_marketing

Campanhas sintéticas

fact_financial

Simulação financeira

fact_targets

Metas mensais

13. Resultado esperado para o projeto
Regras fundamentais da Vision Group

Receita comercial calibrada

R$ 8,2 milhões
Custo comercial calibrado

R$ 6,1 milhões
Lucro bruto calibrado

R$ 2,1 milhões
Pedidos cancelados não entram nos indicadores comerciais.

Devoluções são relacionadas aos produtos efetivamente devolvidos.

Reembolsos são reconhecidos pela data da devolução.

Custos não são revertidos automaticamente.

Metas mensais não consideram sazonalidade.

Marketing e financeiro são explicitamente sintéticos.

Uma observação importante antes de executar: como a calibração agora é feita em centavos, TARGET_REVENUE - TARGET_COST deve corresponder exatamente a TARGET_GROSS_PROFIT. No exemplo, R$ 8,2 milhões menos R$ 6,1 milhões resulta em R$ 2,1 milhões.

Também mantenha a ordem de execução: primeiro dimensões e itens, depois calibração comercial, cabeçalho de vendas, devoluções e, por último, financeiro e metas.

Com essas alterações, o gerador fica conceitualmente mais consistente e a documentação passa a explicar as limitações dos dados, algo essencial para apresentar o projeto como trabalho profissional de Business Intelligence.


Saiba mais


MARKETPLACE4U NEGOCIOS DIGITAIS LTDA
Gestão Mil: ERP por módulos
Financeiro, notas, estoque e ponto. Só o que você usa.
Anúncio

refaça o scritp completo com essas atualizações e me mande separadp em 4 partes

Vou reconstruir o gerador completo da Vision Group, incorporando todas as correções anteriores e organizando o código em quatro partes sequenciais.

O objetivo é que você consiga copiar as quatro partes para um único arquivo Python e executar tudo em sequência, sem precisar misturar trechos das versões anteriores.

Estrutura do script
Parte

Conteúdo

1/4

Configurações, funções auxiliares, cidades brasileiras e dimensões

2/4

Pedidos, itens, custos e calibração comercial

3/4

Devoluções por produto, clientes, estoque e marketing

4/4

Financeiro, metas, regras de negócio, metadados e exportação

Premissas do gerador
82.400 clientes cadastrados.

145.200 pedidos.

R$ 8,2 milhões de receita comercial antes das devoluções.

R$ 6,1 milhões de custo comercial.

R$ 2,1 milhões de lucro bruto.

25,61% de margem bruta aproximada.

Cinco anos de histórico, de 2021 a 2025.

Operações distribuídas pelo Brasil.

Dados financeiros, marketing, estoque e metas identificados como sintéticos.

Importante: os valores comerciais são calibrados artificialmente. O dataset é destinado à demonstração de técnicas de BI, não à reprodução estatística independente de uma empresa real.

PARTE 1/4 — Configuration, Functions and Dimensions
Crie um arquivo chamado vision_group_dataset.py.

Instale as bibliotecas necessárias, caso ainda não tenha:


bash
pip install pandas numpy
Nesta primeira parte, vamos criar as configurações, a função de distribuição monetária, as referências geográficas brasileiras e todas as dimensões.


python
# ============================================================
# VISION GROUP - EXECUTIVE BI DATASET
# PART 1/4 - CONFIGURATION AND DIMENSIONS
# ============================================================

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
PARTE 2/4 — Sales, Items and Commercial Calibration
Nesta parte construiremos os pedidos e os itens vendidos.

A receita e o custo serão calibrados exclusivamente nos pedidos não cancelados. O custo de referência do produto continuará preservado na dimensão, enquanto o custo comercial realizado será registrado nos itens.


python
# ============================================================
# VISION GROUP - EXECUTIVE BI DATASET
# PART 2/4 - SALES AND COMMERCIAL CALIBRATION
# ============================================================

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
