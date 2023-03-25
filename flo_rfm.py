
##################################
# Customer Segmentation with RFM
#################################

###################
# Business Problem
###################

# FLO wants to segment its customers and determine marketing strategies according to these segments.
# For this, the behavior of the customers will be defined and groups will be formed according to these behavior clusters.

#######################
# The story of dataset
#######################

# The dataset is based on the past shopping behavior of customers who made their last purchases from OmniChannel (both online and offline) in 2020 - 2021.
# consists of the information obtained.

# master_id: Unique customer number
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : The channel where the last purchase was made
# first_order_date : The date of the customer's first purchase
# last_order_date : The date of the last purchase made by the customer
# last_order_date_online : The date of the last purchase made by the customer on the online platform
# last_order_date_offline : The date of the last purchase made by the customer on the offline platform
# order_num_total_ever_online : The total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : Total number of purchases made by the customer offline
# customer_value_total_ever_offline : The total price paid by the customer for offline purchases
# customer_value_total_ever_online : The total price paid by the customer for their online shopping
# interested_in_categories_12 : List of categories the customer has purchased from in the last 12 months



# TASK 1: Data Understanding
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

# display settings
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 500)

df_ = pd.read_csv(r"datasets\flo_data_20k.csv")
df = df_.copy()

def check_df(dataframe, head=5):
    print("INFO".center(70,'='))
    print(dataframe.info())

    print("SHAPE".center(70,'='))
    print('Rows: {}'.format(dataframe.shape[0]))
    print('Columns: {}'.format(dataframe.shape[1]))

    print("TYPES".center(70,'='))
    print(dataframe.dtypes)

    print("HEAD".center(70, '='))
    print(dataframe.head(head))

    print("TAIL".center(70,'='))
    print(dataframe.tail(head))

    print("NULL".center(70,'='))
    print(dataframe.isnull().sum())

    print("QUANTILES".center(70,'='))
    print(dataframe.describe().T)

check_df(df)


# 3. Omnichannel means that customers shop from both online and offline platforms.
# Create new variables for each customer's total purchases and spend.
df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_price"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]


# 4. Examine the types of variables.
df.dtypes
df.loc[:,df.columns.str.contains("date")]=df.loc[:,df.columns.str.contains("date")].apply(pd.to_datetime)


# 5. Look at the distribution of the number of customers, average number of products purchased, and average spend in shopping channels.
df.groupby("order_channel").agg({"master_id":"count",
                                 "total_order":["mean","sum"],
                                 "total_price":["mean","sum"]})

# 6. Rank the top 10 paying customers.
df.sort_values(by="total_price",ascending=False).head(10)

# 7. Rank the top 10 customers with the most orders.
df.sort_values(by="total_order",ascending=False).head(10)


# 8. Functionalize data preprocessing

def data_processing(dataframe):
    dataframe["total_order"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_price"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]

    dataframe.loc[:, dataframe.columns.str.contains("date")] = dataframe.loc[:, dataframe.columns.str.contains("date")].apply(pd.to_datetime)

    return dataframe

df = data_processing(df)

# Visualization
def visualization(df, col, target, func):
    sns.barplot(x=df[col], y=df[target], estimator=func, palette="Blues")
    plt.show(block=True)

visualization(df,"order_channel","total_order",sum)


# TASK 2: Calculating RFM Metrics

df["last_order_date"].max()

#df["last_order_date"].max() + dt.timedelta(days=2)

today_date = dt.datetime(2021,6,1)

df['last_order_date'] = pd.to_datetime(df['last_order_date'])
df["last_date"] = df["last_order_date"].apply(lambda x: (today_date - x).days)

rfm = df[["master_id","total_order","last_date","total_price","interested_in_categories_12"]]
rfm.head()

rfm.columns = ["master_id","frequency","recency","monetary","category"]

rfm["recency"]=rfm["recency"].astype("int")
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5])


# TASK 3: Calculation of RF and RFM Scores
rfm["RFM_SCORE"] = rfm["recency_score"].astype(str)+ rfm["frequency_score"].astype(str)

rfm[["recency","recency_score","frequency","frequency_score","RFM_SCORE"]].head()



# TASK 4: Definition of RF Scores as Segments
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)


# TASK 5: Action Time
# 1. Examine the recency, frequency and monetary averages of the segments.
rfm[["segment","recency","frequency","monetary"]].groupby("segment").agg(["mean","count"])

list = ["recency", "frequency", "monetary"]
for i in list:
    visualization(rfm, i, "segment",np.mean)

# 2. With the help of RFM analysis, find the customers in the relevant profile for 2 cases and save the customer IDs to the csv.
# a. FLO includes a new women's shoe brand. The product prices of the brand it includes are above the general customer preferences.
# For this reason, customers in the profile who will be interested in the promotion of the brand and product sales are requested to be contacted privately.
# Those who shop from their loyal customers (champions, loyal_customers), on average over 250 TL and from the women category,
# are the customers to contact privately. Save the id numbers of these customers in the csv file as new_brand_target_customer_id.cvs.

rfm_new=rfm[((rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers")) & ((rfm["category"].str.contains("KADIN")) & (rfm["monetary"]>250))]
rfm_new.reset_index(inplace=True)
rfm_new.drop("index",axis=1)
rfm_new["master_id"].to_csv("rfm_new.csv")
rfm_new.head()

rfm_new.value_counts()

# b. Up to 40% discount is planned for Men's and Children's products.
# It is aimed to specifically target customers who are good customers in the past, but who have not shopped for a long time,
# who are interested in the categories related to this discount, who should not be lost, who are asleep and new customers.
# Save the ids of the customers in the appropriate profile to the csv file as discount_target_customer_ids.csv.

rfm_new_2 = rfm[((rfm["segment"] == "about_to_sleep") | (rfm["segment"] == "new_customers")) &
                ((rfm["category"].str.contains("ERKEK")) | (rfm["category"].str.contains("COCUK")))]
rfm_new_2.reset_index(inplace=True)
rfm_new_2.drop("index",axis=1)
rfm_new_2["master_id"].to_csv("rfm_new_2.csv")


# TASK 6: Functionalize the whole process.
def create_rfm(dataframe):
    dataframe["total_order"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_price"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]

    dataframe.loc[:, dataframe.columns.str.contains("date")] = dataframe.loc[:,dataframe.columns.str.contains("date")].apply(pd.to_datetime)

    dataframe["last_order_date"].max()
    # df["last_order_date"].max() + dt.timedelta(days=2)
    today_date = dt.datetime(2021, 6, 1)

    dataframe['last_order_date'] = dataframe['last_order_date']
    dataframe["last_date"] = dataframe["last_order_date"].apply(lambda x: (today_date - x).days)

    rfm = dataframe[["master_id", "total_order", "last_date", "total_price", "interested_in_categories_12"]]
    rfm.columns = ["master_id", "frequency", "recency", "monetary", "category"]

    rfm["recency"] = rfm["recency"].astype("int")
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)

    return rfm

df_new = create_rfm(df)
