# Synthetic E-commerce Dataset Generator (Enhanced)
import os
import numpy as np
import pandas as pd

SEED=42
TOTAL_ROWS=1_000_000
OUTPUT_DIR='synthetic_dataset'
rng=np.random.default_rng(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_CUSTOMERS=100_000
N_PRODUCTS=5_000
AVG_ITEMS_PER_ORDER=2
START_DATE=pd.Timestamp('2010-01-01')
END_DATE=pd.Timestamp('2025-12-31')

states=['California','Texas','Florida','New York','Illinois','Pennsylvania','Ohio','Georgia','North Carolina','Michigan','New Jersey','Virginia','Washington','Arizona','Massachusetts']
regions={'California':'West','Texas':'South','Florida':'South','New York':'Northeast','Illinois':'Midwest','Pennsylvania':'Northeast','Ohio':'Midwest','Georgia':'South','North Carolina':'South','Michigan':'Midwest','New Jersey':'Northeast','Virginia':'South','Washington':'West','Arizona':'West','Massachusetts':'Northeast'}

signup_dates=START_DATE+pd.to_timedelta(rng.integers(0,(END_DATE-START_DATE).days+1,size=N_CUSTOMERS),unit='D')
customers=pd.DataFrame({
'customer_id':np.arange(1,N_CUSTOMERS+1),
'customer_unique_id':[f'CUST-{i:07d}' for i in range(1,N_CUSTOMERS+1)],
'state':rng.choice(states,size=N_CUSTOMERS),
'signup_date':signup_dates,
'customer_segment':rng.choice(['Consumer','Small Business','Corporate'],size=N_CUSTOMERS,p=[0.75,0.20,0.05]),
'acquisition_channel':rng.choice(['Organic Search','Paid Search','Social Media','Referral','Marketplace','Email'],size=N_CUSTOMERS)
})
customers['signup_date']=pd.to_datetime(customers['signup_date']).dt.strftime('%Y-%m-%d')

geography=pd.DataFrame({'state':states,'region':[regions[s] for s in states]})

categories={'Electronics':['Smartphones','Laptops','Headphones','Accessories','Cameras'],'Home':['Furniture','Kitchenware','Decor','Bedding','Storage'],'Fashion':['Clothing','Shoes','Bags','Watches','Accessories'],'Sports':['Fitness','Outdoor','Cycling','Team Sports','Camping'],'Beauty':['Skincare','Haircare','Makeup','Fragrances','Personal Care'],'Books':['Fiction','Business','Technology','Education','Children']}
product_ids=np.arange(1,N_PRODUCTS+1)
prod_cat=rng.choice(list(categories.keys()),N_PRODUCTS)
products=pd.DataFrame({'product_id':product_ids,'category':prod_cat,'subcategory':[rng.choice(categories[c]) for c in prod_cat]})
costs=pd.DataFrame({'product_id':product_ids,'unit_cost':np.round(rng.lognormal(3.0,0.8,N_PRODUCTS),2)})

fixed=len(customers)+len(products)+len(costs)+len(geography)
avail=TOTAL_ROWS-fixed
N_ORDERS=int(avail/(1+AVG_ITEMS_PER_ORDER+1+0.1))
while True:
    item_counts=rng.choice([1,2,3,4],size=N_ORDERS,p=[0.2,0.45,0.25,0.1])
    total=fixed+N_ORDERS+item_counts.sum()+N_ORDERS+round(N_ORDERS*0.1)
    if total<=TOTAL_ROWS: break
    N_ORDERS-=max(1,(total-TOTAL_ROWS)//5)

order_ids=np.arange(1,N_ORDERS+1)
order_dates=START_DATE+pd.to_timedelta(rng.integers(0,(END_DATE-START_DATE).days+1,size=N_ORDERS),unit='D')
statuses=rng.choice(['Delivered','Shipped','Processing','Canceled'],size=N_ORDERS,p=[0.77,0.10,0.08,0.05])
promised=order_dates+pd.to_timedelta(rng.integers(2,10,size=N_ORDERS),unit='D')
channels=rng.choice(['Website','Mobile App','Marketplace','B2B Portal'],size=N_ORDERS,p=[0.5,0.25,0.2,0.05])

delivered=[]; canceled=[]
for d,s in zip(order_dates,statuses):
    if s=='Canceled': delivered.append(pd.NaT); canceled.append(d+pd.Timedelta(days=int(rng.integers(0,5))))
    else: delivered.append(d+pd.Timedelta(days=int(rng.integers(1,12)))); canceled.append(pd.NaT)

orders=pd.DataFrame({'order_id':order_ids,'customer_id':rng.integers(1,N_CUSTOMERS+1,size=N_ORDERS),'order_date':order_dates,'status':statuses,'promised_delivery_date':promised,'delivered_date':delivered,'canceled_at':canceled,'sales_channel':channels})

item_order_ids=np.repeat(order_ids,item_counts)
actual=len(item_order_ids)
order_items=pd.DataFrame({'order_id':item_order_ids,'product_id':rng.integers(1,N_PRODUCTS+1,size=actual),'quantity':rng.integers(1,6,size=actual),'price':np.round(rng.lognormal(3.3,0.7,size=actual),2),'freight':np.round(rng.uniform(2,35,size=actual),2)})
pp=rng.random(actual)
dpct=np.where(pp<0.25,rng.uniform(0.05,0.30,actual),0)
order_items['discount']=np.round(order_items['quantity']*order_items['price']*dpct,2)
order_items['promotion_id']=np.where(dpct>0,rng.integers(1000,1100,actual).astype(str),np.nan)
order_items=order_items.merge(costs,on='product_id',how='left')
order_items['gross_merchandise_value']=order_items['quantity']*order_items['price']
order_items['freight_value']=order_items['freight']
order_items['line_revenue']=order_items['gross_merchandise_value']-order_items['discount']+order_items['freight_value']
order_items['line_cost']=order_items['quantity']*order_items['unit_cost']

order_totals=order_items.groupby('order_id')['line_revenue'].sum()
payments=pd.DataFrame({'order_id':order_ids,'payment_value':np.round(order_totals.reindex(order_ids).to_numpy(),2)})
ship=order_items.groupby('order_id')['freight'].sum()
orders['shipping_cost']=(orders['order_id'].map(ship)*rng.uniform(0.65,1.10,len(orders))).round(2)

eligible=orders.loc[orders.status=='Delivered','order_id'].to_numpy()
N_RET=min(int(round(N_ORDERS*0.10)),len(eligible))
ret_ids=rng.choice(eligible,size=N_RET,replace=False)
pmap=payments.set_index('order_id')['payment_value']
odmap=orders.set_index('order_id')['order_date']
ret_dates=[odmap.loc[o]+pd.Timedelta(days=int(rng.integers(1,61))) for o in ret_ids]
returns=pd.DataFrame({'order_id':ret_ids,'return_date':ret_dates,'refund_value':np.round([pmap.loc[o]*rng.uniform(0.25,1.0) for o in ret_ids],2),'return_reason':rng.choice(['Damaged Product','Wrong Product','Customer Dissatisfaction','Late Delivery','Product Defect'],size=N_RET),'return_status':rng.choice(['Approved','Pending','Rejected'],size=N_RET,p=[0.8,0.15,0.05]),'returned_quantity':rng.integers(1,5,size=N_RET)})
orders.loc[orders.order_id.isin(returns.order_id),'status']='Returned'

for c in ['order_date','promised_delivery_date','delivered_date','canceled_at']:
    orders[c]=pd.to_datetime(orders[c]).dt.strftime('%Y-%m-%d')
returns['return_date']=pd.to_datetime(returns['return_date']).dt.strftime('%Y-%m-%d')

tables={'customers':customers,'geography':geography,'products':products,'costs':costs,'orders':orders,'order_items':order_items,'payments':payments,'returns':returns}
for n,df in tables.items(): df.to_csv(os.path.join(OUTPUT_DIR,f'{n}.csv'),index=False)
print('Generated',sum(len(df) for df in tables.values()),'rows')
