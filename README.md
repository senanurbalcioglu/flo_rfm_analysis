## WHAT IS RFM ANALYSIS ?
<img src="https://user-images.githubusercontent.com/105918751/227719690-b3849554-e0b8-4a8c-8521-6c4280dd73b1.png" width="500" height="300">
RFM analysis is a simple and technique method to segment customers according to their recency, frequency and monetary values. <br />
Recency value means when a customer last made a purchase. If first customer shopped one day ago and second customer shopped ten days ago, first customer is more valuable. <br />
Frequency value is how often a customer has made purchase.<br />
Monetary value means how much a customer has spent overall.

### RFM Scores
It is standardized to express RFM metric values in the same gender. RFM scores are needed in order to better compare and analyze RFM metric values both within themselves and with each other. Recency, frequency and monetary values are combined. There is a problem. There will be too many combinations as all RFM metrics range from 1 to 5. This situation makes it difficult for the segmentation process to be efficient. This chart shows segments according to recency and frequency values.

![image](https://user-images.githubusercontent.com/105918751/227720862-959897f6-8ee5-47fe-8e35-6bfdb567e75e.png)

### BUSINESS PROBLEM
FLO wants to set a roadmap for sales and marketing activities.In order for the company to make a medium-long-term plan, it is necessary to estimate the potential value that existing customers will provide to the company in the future.

*****

### THE STORY OF DATASET
The dataset is based on the past shopping behavior of customers who made their last purchases from OmniChannel (both online and offline) in 2020 - 2021 consists of the information obtained.

master_id: Unique customer number<br />
order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)<br />
last_order_channel : The channel where the last purchase was made<br />
first_order_date : The date of the customer's first purchase<br />
last_order_date : The date of the last purchase made by the customer<br />
last_order_date_online : The date of the last purchase made by the customer on the online platform<br />
last_order_date_offline : The date of the last purchase made by the customer on the offline platform<br />
order_num_total_ever_online : The total number of purchases made by the customer on the online platform<br />
order_num_total_ever_offline : Total number of purchases made by the customer offline<br />
customer_value_total_ever_offline : The total price paid by the customer for offline purchases<br />
customer_value_total_ever_online : The total price paid by the customer for their online shopping<br />
interested_in_categories_12 : List of categories the customer has purchased from in the last 12 months<br />
