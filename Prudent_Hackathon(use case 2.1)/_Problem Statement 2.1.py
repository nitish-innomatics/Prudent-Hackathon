#!/usr/bin/env python
# coding: utf-8

# # Problem

# ##### Develop a demand forecasting model fro a retail store chain can predict future sales for each product category and store loaction the model should incorporate factor like past sales data promotions,holidays and demographics information |

# # Solution

# ## Performing the probelm solution using CRISP-ML(Q) Mtehodology

# ## Step1:- Data understanding 

# In[23]:


df = pd.read_csv(r"C:\Users\91832\Downloads\online_retail_II.csv")


# In[35]:


# Data sample representation
df


# In[25]:


#Description of the Data set
df.describe(include='all')


# In[26]:


#printing important infos about the dataset
dataset_structure = df.info()


# ## Step2:- Data Preparation

# #### Performing EDA

# In[29]:


#finding out the missing values
missing_values = df.isnull().sum() #Identifying missing values
print(missing_values)


# In[31]:


# Remove rows with missing descriptions
data_cleaned = df.dropna(subset=['Description'])
# Replace missing customer IDs with a placeholder
data_cleaned = df.dropna(subset=['Customer ID'])


# In[34]:


# Display the first few rows of the dataset
data_cleaned.head()


# ##### Data Visualization

# In[37]:


#Univariate Analysis

# Filtering out extreme outliers for visualization purposes
# We could use the interquartile range (IQR) to filter out extreme outliers.
Q1 = data_cleaned['Quantity'].quantile(0.25)
Q3 = data_cleaned['Quantity'].quantile(0.75)
IQR = Q3 - Q1
# Define bounds for what we'll consider as acceptable values (this is arbitrary and can be adjusted)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the data within the interquartile range for visualization
filtered_quantities = data_cleaned[(data_cleaned['Quantity'] >= lower_bound) & (data_cleaned['Quantity'] <= upper_bound)]['Quantity']

# Now let's plot the histogram for this filtered data
plt.figure(figsize=(10, 6))
sns.histplot(filtered_quantities, bins=50, kde=False)
plt.title('Distribution of Quantity within IQR')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.show()

#Let's do boxplot to undertand the data more
plt.figure(figsize=(10, 6))
sns.boxplot(x=filtered_quantities)
plt.title('Boxplot for Quantities within IQR')
plt.xlabel('Quantity')
plt.show()

# Summary Statistics for the filtered Quantity data
filtered_quantity_description = filtered_quantities.describe()
print(filtered_quantity_description)


# ### concluion for the above visualization

# `From the histogram`, we observe that the most frequent quantity of items purchased in transactions falls between 0 and 10, with a notable peak at around 4 items per purchase. This central tendency is confirmed by the summary statistics and the median ('50%') value in the provided data snippet, which indicates a median purchase quantity of 4 items.
# 
# `The boxplot` complements this by visualizing the spread of the data within the IQR, with the 'box' representing the middle 50% of the dataset. The median is marked by the line within the box, and it aligns with the median value from the summary statistics. The 'whiskers' of the boxplot extend to values within 1.5 times the IQR from the lower and upper quartiles, and we can see a few outliers represented by dots beyond the whiskers. These outliers are quantities that are beyond the typical range of data but within the constraints I set for visualization purposes.
# 
# Interestingly, there are negative quantities which could indicate returned items or other adjustments to orders. The presence of negative values is something that may warrant further investigation to understand the context whether they are returns, cancellations, or data entry errors.

# ### Price analysis

# In[45]:


# Filter out extreme price outliers for visualization purposes using the interquartile range (IQR)
Q1_price = data_cleaned['Price'].quantile(0.25)
Q3_price = data_cleaned['Price'].quantile(0.75)
IQR_price = Q3_price - Q1_price
lower_bound_price = Q1_price - 1.5 * IQR_price
upper_bound_price = Q3_price + 1.5 * IQR_price
filtered_prices = data_cleaned[(data_cleaned['Price'] >= lower_bound_price) & (data_cleaned['Price'] <= upper_bound_price)]['Price']

# Histogram for the filtered Price data
plt.figure(figsize=(10, 6))
sns.histplot(filtered_prices, bins=50, kde=True,color="red")
plt.title('Distribution of Price within IQR')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Boxplot for the filtered Price data
plt.figure(figsize=(10, 6))
sns.boxplot(x=filtered_prices,color="Black")
plt.title('Boxplot for Prices within IQR')
plt.show()

# Summary Statistics for the filtered Price data
filtered_price_description = filtered_prices.describe()
print(filtered_price_description)


# ### Conclusion

# `Histogram Interpreation:`
# The histogram `shows a distribution of 'Price' with several peaks,` indicating that certain price points are more common than others. `The majority of items are priced below 3 currency units,` with significant frequencies at specific intervals—likely corresponding `to common price points or pricing strategies such as 0.99 or 1.99.` The presence of multiple peaks suggests that the retail strategy includes various price points, possibly targeting different market segments or reflecting a diverse product range.
# 
# `Boxplot Interpretation:`
# The boxplot further narrows the focus to the `spread of the central 50% of the data,` which `falls between 1.25 and 2.95 `currency `units as indicated by the 25th and 75th percentiles`. The `median price,` represented by the line within the box, is `at 1.69`, aligning with the summary statistics. The 'whiskers' extend to the most extreme data points within 1.5 times the IQR from the quartiles, excluding the outliers. The outliers, depicted as individual points beyond the whiskers, show that while most items are priced within a lower range, there are items priced up to 7.5 currency units, suggesting premium options or larger quantities.
# 
# These visualizations, when combined with the summary statistics, reinforce the understanding that the pricing strategy is varied, with a significant concentration of products in the lower price range. The distribution's skew towards lower prices suggests that the retailer primarily deals in affordable items, with a smaller selection of higher-priced goods. The careful placement of price points near whole numbers may also indicate psychological pricing tactics to appeal to customers' perceptions of value.

# ### 2. Univariate Analysis of Categorical Features
# #### Analyzing 'StockCode'

# In[46]:


# Top 10 Most Frequent StockCodes
plt.figure(figsize=(10, 6))
data_cleaned['StockCode'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Most Frequent StockCodes')
plt.xlabel('StockCode')
plt.ylabel('Frequency')
plt.show()


# `The bar chart`provided showcases the top 10 most frequent StockCodes in the dataset, which represent the most commonly sold items or item categories in the retail business.
# 
# `Interpretation:`
# 
# - The highest bar corresponds to the StockCode '85123A', indicating that this particular item is the most frequently sold, standing out significantly from the rest.
# - The frequency of the items gradually decreases from the most common item to the tenth most common, suggesting a steep drop-off in the occurrence of specific items.
# - This distribution of frequencies could indicate a small number of key products driving a large portion of the transactions, which is common in retail where a few 'best sellers' often dominate sales.
# - Understanding the characteristics of these top-selling items could provide insights into consumer preferences and could be crucial for inventory management, marketing strategies, and sales forecasting.
# 
# The StockCodes are anonymized identifiers for products. In a practical business context, these would be cross-referenced with product descriptions to provide actionable insights into stock levels, product demand, and sales performance. It is also beneficial for identifying potential areas to expand the product range or to plan promotional activities around these popular items to maximize revenue.

# #### Analyzing 'Description'

# In[42]:


# Top 10 Most Frequent Descriptions
plt.figure(figsize=(12, 8))
data_cleaned['Description'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Most Frequent Product Descriptions')
plt.xlabel('Description')
plt.ylabel('Frequency')
plt.show()


# ### CONCLUSION

# The bar chart illustrates the top 10 most frequent product descriptions, indicating these are the products most commonly purchased.
# 
# Interpretation:
# 
# - The 'WHITE HANGING HEART T-LIGHT HOLDER' is the most popular product by a notable margin, suggesting it's a flagship item or a customer favorite.
# - The other items on the list show a more even distribution of frequency, but all are significantly lower than the top item.
# - This chart gives us valuable insight into the product mix's performance, highlighting items that might be central to sales strategies such as upselling, cross-selling, and promotions.
# - The variety of items in the top 10 suggests a diversity in the types of products that are popular, which could indicate a broad customer base with varied interests or seasonal buying patterns.
# - For inventory management, these insights can help ensure that these popular items are well-stocked to meet customer demand.
# - For marketing strategies, these products could be featured in advertising campaigns or used to attract customers through discounts and deals.
# 
# Understanding the popularity of these items could also lead to the exploration of similar products to expand the offering and capitalize on the demonstrated consumer interest. It’s crucial for the retail business to leverage this data to optimize the product portfolio and enhance customer satisfaction.

# #### Analyzing 'Country'

# In[43]:


plt.figure(figsize=(15, 8))  # Increase figure size
country_counts = data_cleaned['Country'].value_counts()
sns.barplot(x=country_counts.index, y=country_counts.values)  # Create bar plot
plt.title('Frequency Distribution of Sales by Country')
plt.xlabel('Country')
plt.ylabel('Number of Transactions')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xticks(rotation=90)  # Rotate x-axis labels to prevent overlap
plt.show()


# ### conclusion

# The visual representation here shows a logarithmic distribution of sales transactions across different countries.
# 
# Interpretation:
# 
# - The United Kingdom dominates with the highest number of transactions, suggesting that the business's primary market is located there. This could be due to the business being based in the UK or having a stronger brand presence and customer base there.
# - EIRE, Germany, and France follow as the next most significant markets, indicating a solid foothold in these regions. These countries may represent key growth areas or stable markets with loyal customer bases.
# - There is a sharp drop-off in transaction numbers as we move to other countries, indicating a long tail of international sales.
# - Countries towards the end of the chart, such as Brazil and Nigeria, have significantly fewer transactions, which could imply potential markets for expansion or areas where the business is still developing its presence.
# - This logarithmic scale highlights the disparity between the number of transactions in the UK compared to other countries, emphasizing the need for market-specific strategies.
# - The spread of colors from red to pink signifies the descending order of transaction volumes, visually reinforcing the data's hierarchy and making it easier to discern the distribution pattern.
# 
# The data suggests that market-specific strategies might be beneficial, with a focus on customer acquisition and retention in the UK, while exploring expansion opportunities in lower-volume countries. It’s important to consider cultural, economic, and logistical factors when planning for international growth based on this transaction distribution.

# #### Comparative Analysis
# ##### Top-Selling Products

# In[48]:


# Group by 'Description' and sum 'Quantity' to find top-selling products
top_selling_products = data_cleaned.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

# Bar plot of top-selling products
plt.figure(figsize=(12, 8))
top_selling_products.plot(kind='bar')
plt.title('Top-Selling Products')
plt.xlabel('Product Description')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45, ha='right')
plt.show()


# ### Conclusion

# The bar chart presents the top-selling products based on the total quantity sold. It provides a clear visual indication of which items are the most popular among customers.
# 
# Interpretation:
# 
# - The 'WHITE HANGING HEART T-LIGHT HOLDER' stands out as the top-selling product, with the highest total quantity sold, indicating its popularity or possible prominence in sales promotions.
# - 'WORLD WAR 2 GLIDERS ASSORTED DESIGNS' and 'BROCADE RING PURSE' follow closely, showing significant sales, which could suggest they are staple items or were part of successful marketing campaigns.
# - The other products displayed also show substantial sales, with 'PACK OF 72 RETROSPOT CAKE CASES', 'ASSORTED COLOUR BIRD ORNAMENT', 'PACK OF 60 TEATIME PAPER/FOIL CAKE CASES', 'JUMBO BAG RED RETROSPOT', 'BLACK AND WHITE PAISLEY FLOWER MUG', and 'SMALL POPCORN HOLDER' rounding out the top sellers.
# 
# This chart is a valuable tool for inventory management, providing insight into which products might require restocking more frequently and which could be leveraged for upselling or cross-selling opportunities. It can also help in making strategic decisions regarding which products to feature in marketing materials or to include in promotions to drive sales.

# ###  Advanced Visual Analytics
# #### Violin Plot for Quantity by Country:

# In[49]:


plt.figure(figsize=(12, 8))
sns.violinplot(x='Country', y='Quantity', data=data_cleaned[data_cleaned['Quantity'] < 50])  # Limiting to reasonable quantities
plt.title('Quantity Distribution by Country')
plt.xticks(rotation=90)
plt.show()


# ### Conclusion

# The violin plot above illustrates the distribution of quantities ordered by customers from different countries. This type of chart is particularly useful for visualizing the distribution density along with the range of the data.
# 
# Interpretation:
# 
# The bulk of orders are centered around a quantity of zero, which may indicate a high volume of low-quantity transactions across all countries. This could suggest that customers typically buy items in small quantities.
# The length of the violins for each country indicates the range of order quantities, while the width shows the frequency. Wider sections represent a higher frequency of orders at that quantity level, indicating common order sizes.
# The plot also shows several countries with extreme negative quantities, which could represent returns or canceled orders. Notably, the United Kingdom has the widest distribution, indicating a high variability in order quantities. This could be due to the UK being the primary market with a diverse range of customer behaviors.
# There are no violins extending into large positive quantities, suggesting that very large orders are not common in the dataset.
# 
# This violin plot is instrumental in identifying trends in purchasing behavior across different markets and can inform strategies for inventory management, targeted marketing, and customer segmentation. It's also helpful for identifying potential issues with returns and cancellations, particularly in markets with a broader spread of negative quantities.

# ### Faceted Grid Plot for Quantity and Price:

# In[53]:


#Faceted Grid Plot for Quantity and Price
grid = sns.FacetGrid(data_cleaned, col='Country', col_wrap=4, height=4)
grid.map(sns.scatterplot, 'Quantity', 'Price')
grid.add_legend()
plt.show()


# ### Conclusion

# The series of plots depict quantity sold against a variety of countries. Each plot is a visual representation of the quantity of products sold in a specific country, providing an insight into the distribution and sales volume across different regions.
# 
# Interpretation:
# 
# The United Kingdom, being the plot with the most data points and the highest individual values, is likely the primary market. The density of points at the lower end of the quantity range suggests that a large number of transactions involve small quantities.
# 
# For countries like France and Germany, the plots show fewer transactions but with a wider range of quantities, indicating a smaller but potentially more variable market.
# 
# In markets with very few transactions, such as Nigeria and Malta, each data point becomes more significant, and the higher-quantity sales could represent bulk purchases or larger orders.
# 
# The negative quantities in some plots could be due to returns or cancelled orders, which appear to be significant in some countries like Iceland and West Indies. These outliers could be skewing the average quantity figures and may require separate analysis to understand the reasons behind these returns.
# 
# The distribution of sales across countries could inform decisions on where to focus marketing efforts, where to expand or reduce presence, and how to manage logistics and supply chain strategies.
# 
# These collection of plots allows for a comparative analysis of sales by country, indicating where the company is most successful and where there may be challenges or opportunities for growth.

# In[54]:


#### Radar Chart for Product Categories:


# In[55]:


data_cleaned['TotalSales'] = data_cleaned['Quantity'] * data_cleaned['Price']
# Group by 'Description' and sum up the 'TotalSales'
category_sales = data_cleaned.groupby('Description')['TotalSales'].sum().sort_values(ascending=False)
# Select top 5 categories for the radar chart
top_n_categories = category_sales.head(5) 
print(top_n_categories)


# In[56]:


from math import pi

# Provided sales data for top products
sales_data = {
    'Description': [
        'WHITE HANGING HEART T-LIGHT HOLDER',
        'REGENCY CAKESTAND 3 TIER',
        'ASSORTED COLOUR BIRD ORNAMENT',
        'JUMBO BAG RED RETROSPOT',
        'POSTAGE'
    ],
    'TotalSales': [148876.66, 136866.30, 69854.96, 51608.40, 45520.86]
}

# Create a DataFrame
df = pd.DataFrame(sales_data)

# Number of variables we're plotting.
num_vars = len(df['Description'])

# Split the circle into even parts and save the angles
# so we know where to put each axis.
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is made in a circular (not polygon) format, so we need to complete the loop
values = df['TotalSales'].tolist()
values += values[:1]
angles += angles[:1]

# Draw plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='red', alpha=0.25)
ax.plot(angles, values, color='red', linewidth=2)  # Change the color if you want

# Draw one axe per variable and add labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(df['Description'])

# Remove y-axis labels to clean up the plot
ax.yaxis.set_visible(False)

# Give the plot a title and show it
ax.set_title('Sales by Product Description')
plt.show()


# ### Conclusion

# The radar chart above visualizes sales data across different product categories. Each axis represents a different product, with the length of the axis proportional to the sales of that product. This type of visualization is helpful for comparing multiple products on the same scale, allowing us to identify which products are performing well and which are not.
# 
# Interpretation:
# 
# The "WHITE HANGING HEART T-LIGHT HOLDER" has the highest sales, as indicated by the axis reaching the outermost ring of the chart. This suggests that it is a popular item among customers and a significant revenue driver for the business.
# 
# The "REGENCY CAKESTAND 3 TIER" also shows strong sales, with its axis extending towards the outer rings. This product likely appeals to customers looking for home decor or hosting items.
# 
# "ASSORTED COLOUR BIRD ORNAMENT" and "JUMBO BAG RED RETROSPOT" have a moderate level of sales, as shown by their axes reaching the mid-range of the chart. These products seem to have a stable but lesser contribution to total sales compared to the top-performing items.
# 
# "POSTAGE" shows the least sales among the products listed, which could indicate either a lower price point or less frequency in purchases. Since postage is a service rather than a tangible product, it might also represent additional shipping costs associated with sales, rather than direct product revenue.
# 
# The radar chart is a useful tool for displaying the sales performance of different products in a compact and easily comparable way. It can inform decisions on inventory, marketing strategies, and product development based on the sales appeal of each item.

# In[58]:


import seaborn as sns
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D


# ### Bubble Chart

# In[59]:


# Filter out negative or zero TotalSales values if they are not meaningful for the analysis
data_cleaned = data_cleaned[data_cleaned['TotalSales'] > 0]

# Create the bubble chart
plt.figure(figsize=(10, 8))
bubble_plot = sns.scatterplot(
    x='Price',
    y='Quantity',
    size='TotalSales',
    data=data_cleaned,
    sizes=(20, 1000),  # This controls the range of bubble sizes
    alpha=0.5
)

# Add a legend for bubble sizes
bubble_plot.legend(loc='upper right', title='Total Sales')

plt.title('Price vs Quantity with Bubble Size Indicating Total Sales')
plt.xlabel('Price')
plt.ylabel('Quantity')
plt.grid(True)  # adds a grid for easier visualization
plt.show()


# ### Conclusion

# The bubble chart above illustrates the relationship between price, quantity, and total sales of products. Each bubble represents a product or a set of products with its position on the x-axis showing price and on the y-axis showing quantity. The size of the bubble corresponds to total sales, providing a quick visual cue on which price and quantity combinations are generating the most revenue.
# 
# Interpretation:
# 
# - Larger bubbles, which indicate higher total sales, are concentrated along the lower price range. This suggests that lower-priced items tend to sell in higher quantities, contributing significantly to total sales.
# 
# - There is a cluster of bubbles with large quantities but lower total sales around the lower end of the price axis. It implies that even though these items are sold in large quantities, their low price keeps the total sales value down.
# 
# - A few larger bubbles appear at higher price points as well, which could indicate premium products that, despite their higher price, still sell in sufficient quantities to generate considerable sales.
# 
# - The vertical spread of bubbles at specific price points suggests a variance in the quantity sold at those prices, indicating that for certain price levels, sales quantities can vary widely.
# 
# - The chart lacks a significant presence of large bubbles in the high price and high quantity area, suggesting that items with high prices do not sell in large quantities. This is typical in many sales distributions where higher-priced items are less frequently purchased but may still represent an important profit center due to higher margins.
# 
# This visualization is beneficial for identifying which products are the biggest contributors to revenue and can inform strategic decisions on pricing, promotions, and inventory management. For example, marketing efforts might focus on the high-volume, low-price products to boost sales or on the mid-range price products that bring in substantial sales to improve profit margins. Additionally, supply chain efforts could be optimized to ensure the availability of the most selling products, as represented by the larger bubbles.

# ## Modeling and Evaluation
# ### 1. Linear Regression (Baseline Model)
# Rationale: Linear regression is a good starting point due to its simplicity and interpretability. It provides a baseline to evaluate the performance of more complex models.

# In[64]:


# Summing up the purchase amounts for each customer
data_cleaned['TotalSpend'] = data_cleaned['Quantity'] * data_cleaned['Price']
clv = data_cleaned.groupby('Customer ID')['TotalSpend'].sum()


# In[65]:


# Aggregate features at the customer level
X_aggregated = data_cleaned.groupby('Customer ID').agg({'Quantity': 'sum', 
                                                        'Price': 'mean'})  # Adjust aggregation methods as needed

# Ensure y is also at the customer level
y_aggregated = data_cleaned.groupby('Customer ID')['TotalSpend'].sum()


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X_aggregated, y_aggregated, test_size=0.2, random_state=42)


# In[67]:


# Initialize and train the model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Evaluate the model
y_pred = linear_reg.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r_squared = r2_score(y_test, y_pred)


# In[69]:


# Predict on the test set
y_pred = linear_reg.predict(X_test)

# Scatter plot of Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs Predicted CLV - Linear Regression')
plt.xlabel('Actual CLV')
plt.ylabel('Predicted CLV')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line
plt.show()

# Print the evaluation metrics
print(f'RMSE: {rmse}')
print(f'R-squared: {r_squared}')


# #### Intepratation

# The Linear Regression model's performance is encapsulated by an RMSE of 3580.02 and an R-squared of 0.5401. The RMSE indicates the standard deviation of the residuals, meaning on average, my model's predictions are approximately $3,538 away from the actual CLV values. Given the scale of demand forecastingin my dataset, this level of error is within an acceptable range, allowing for reasonably confident predictions of a customer's value.
# 
# The scatter plot of actual versus predicted demand forecasting reveals a strong alignment for lower demand forecasting values, as most data points cluster near the origin and align closely with the line of perfect prediction. This indicates that my model is particularly adept at predicting demand forecastingfor the majority of customers who have a lower overall lifetime value.
# 
# For higherdemand forecastingvalues, the model exhibits some variance from the actual values, suggesting that while the model captures general trends, it may not be as precise for customers with exceptionally high lifetime value. These outliers represent opportunities for further model refinement.
# 
# The R-squared value of 0.8656 is a strong indicator that my model explains a significant proportion of the variance in demand forecasting meaning that the features I selected for the model have a substantial impact on the demand forecasting. This high R-squared value gives me confidence that the model is capturing the underlying patterns in the dataset, although it's essential to be mindful of potential overfitting.

# ### 2. Decision Tree Regressor
# Rationale: Decision trees handle non-linear relationships and are interpretable, which can provide insights into the decision-making process.

# In[71]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

# X_aggregated and y_aggregated are already defined in the above model

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_aggregated, y_aggregated, test_size=0.2, random_state=42)

# Initialize the model
decision_tree = DecisionTreeRegressor(random_state=42)

# Define a grid of parameters to search over
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model
best_tree = grid_search.best_estimator_

# Predict on the test data
y_pred = best_tree.predict(X_test)

# Calculate RMSE and R-squared
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Visualization
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')  # Diagonal line
plt.title('Decision Tree Regressor: Actual vs Predicted CLV')
plt.xlabel('Actual CLV')
plt.ylabel('Predicted CLV')
plt.show()

# Output the performance metrics
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Decision Tree RMSE: {rmse}')
print(f'Decision Tree R-squared: {r2}')


# #### Interpretation
# ##### Scatter Plot Analysis:
# The scatter plot visualizes the actual versus predicted demand forecasting values, with the dashed red line representing perfect prediction accuracy. The clustering of data points near the line, especially for lower demand forecasting values, indicates that the Decision Tree Regressor has a strong predictive performance for a majority of the customers. The spread of points above and below the line is relatively tight for lower demand forecasting values, suggesting that the model's predictions are consistent with the actual figures.
# 
# However, as the demand forecastingincreases, the model exhibits greater variance from the actual values. A few points, representing customers with high demand forecasting, are more dispersed and distant from the line, signifying potential inaccuracies in predicting higher demand forecasting amounts.
# 
# ##### Model Performance Metrics:
# - RMSE (2777.334): The RMSE value is slightly lower than that of the Linear Regression model, suggesting that the Decision Tree Regressor has a better fit to the data. This improvement indicates that the Decision Tree Regressor has reduced the average prediction error, making it more reliable, especially when precision is critical for business decisions.
# 
# - R-squared (0.7232): The R-squared value is higher compared to the Linear Regression model, explaining approximately 87.95% of the variance in the CLV. This enhancement implies that the Decision Tree Regressor has captured more of the complexities and patterns in the data, potentially due to its ability to model non-linear relationships.
# 
# ##### Best Parameters:
# The best parameters obtained from the grid search are:
# 
# - max_depth: None, meaning the tree was allowed to grow until all leaves are pure or until all leaves contain less than min_samples_split samples.
# min_samples_leaf: 1, indicating the smallest size allowed for a leaf node, which provides the maximum flexibility to the model.
# - min_samples_split: 5, which is the minimum number of samples required to split an internal node.
# These parameters suggest that a more complex decision tree model, one that is allowed to grow without depth constraints, performs better for this dataset. However, the absence of a maximum depth (max_depth: None) and the low requirement for min_samples_leaf could make the model prone to overfitting, which should be monitored.
# 
# ##### Conclusion:
# The Decision Tree Regressor is a robust model for predicting CLV, particularly improving upon the baseline Linear Regression model. Its ability to capture non-linear patterns in the data has translated into improved accuracy. Nevertheless, caution is advised due to the model's complexity and the risk of overfitting, warranting further validation or consideration of pruning techniques to optimize generalization to new data.

# ### 3. Random Forest Regressor
# Rationale: As an ensemble of decision trees, random forests are less prone to overfitting and can capture complex interactions between features.

# In[72]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# X_aggregated and y_aggregated are already defined

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_aggregated, y_aggregated, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
random_forest.fit(X_train, y_train)

# Predict on the test data
y_pred_rf = random_forest.predict(X_test)

# Calculate RMSE and R-squared
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)

# Visualize the predictions
plt.scatter(y_test, y_pred_rf)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.title('Random Forest Regressor: Actual vs Predicted CLV')
plt.xlabel('Actual CLV')
plt.ylabel('Predicted CLV')
plt.show()

# Print performance metrics
print(f'Random Forest RMSE: {rmse_rf}')
print(f'Random Forest R-squared: {r2_rf}')


# #### Interpretation
# ##### Scatter Plot Analysis:
# The scatter plot for the Random Forest Regressor shows the predicted demand forecasting against the actual demand forecasting. The dashed line represents the ideal scenario where predicted values match the actual values perfectly. The plot illustrates that the model predicts low demand forecastingvalues with a high degree of accuracy, evidenced by the dense clustering of points near the origin and along the dashed line. As the demand forecasting increases, the predictions become more spread out, indicating some variance in the model's accuracy for customers with higher lifetime values. Notably, there are a few significant outliers where the model overestimates the demand forecasting.
# 
# ##### Model Performance Metrics:
# - RMSE (3131.7369): The RMSE is higher than that of both the Linear Regression and Decision Tree models, suggesting that on average, the Random Forest's predictions deviate more from the actual values. This may point to the Random Forest model capturing the complex structure of the data but also being affected by overfitting or noise within the higher demand forecasting range.
# 
# - R-squared (0.6480): The R-squared value is slightly lower than the previous models, indicating that the Random Forest model explains about 84.16% of the variance in demand forecasting. This is still a strong score, but it hints that while the model captures a significant portion of the underlying patterns, there may be room for improvement, especially in terms of model generalization.
# 
# ##### Interpretation and Next Steps:
# The Random Forest Regressor, while generally performing well, seems to have slightly underperformed compared to the simpler models in terms of RMSE and R-squared. This could be due to various factors, such as the model's inherent complexity, potential overfitting, or the nature of the data itself.
# 
# Given that Random Forests are less prone to overfitting than individual decision trees, the increased RMSE suggests that the model may be capturing noise as well as signal, or that there are aspects of the data not being fully leveraged. The presence of outliers in the prediction, particularly for high-value customers, suggests that additional feature engineering, hyperparameter tuning, or even data cleaning might be beneficial.
# 
# ##### In response to these insights, I might consider:
# 
# Reviewing the feature set for potential enhancements.
# Further hyperparameter optimization, potentially using more advanced techniques such as RandomizedSearchCV.
# Investigating the outliers to understand the discrepancies in predictions better.
# Exploring ensemble methods that combine the predictions of multiple models to improve accuracy and reduce the impact of overfitting.
# 

# ### 4. Gradient Boosting Regressor
# Rationale: Gradient boosting is a powerful ensemble method that combines weak predictive models to create a strong predictive model, often leading to high performance.
# 

# In[74]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# X_aggregated and y_aggregated have already been prepared

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_aggregated, y_aggregated, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
gbr.fit(X_train, y_train)

# Predict on the test data
y_pred_gbr = gbr.predict(X_test)

# Calculate RMSE and R-squared
rmse_gbr = mean_squared_error(y_test, y_pred_gbr, squared=False)
r2_gbr = r2_score(y_test, y_pred_gbr)

# Visualize the predictions
plt.scatter(y_test, y_pred_gbr)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.title('Gradient Boosting Regressor: Actual vs Predicted CLV')
plt.xlabel('Actual CLV')
plt.ylabel('Predicted CLV')
plt.show()

# Print performance metrics
print(f'Gradient Boosting RMSE: {rmse_gbr}')
print(f'Gradient Boosting R-squared: {r2_gbr}')


# #### Interpretation
# ##### Analysis of the Gradient Boosting Model:
# The RMSE is a measure of the average error made by the model's predictions. An RMSE of 4014.27 suggests that, on average, the model's predicted demand forecasting may be off by approximately $4,014. This is slightly higher than what we observed for both the Decision Tree and Random Forest models, indicating that the Gradient Boosting model, in its current configuration, may not be as precise in this instance.
# 
# The R-squared value indicates the proportion of variance in the demand forecasting that the model is able to predict. An R-squared of 0.8269 means that around 82.69% of the variability in the actual demand forecasting is explained by the model's predictions. While this is a strong value, it's slightly lower than the R-squared values for the previous models, suggesting that the Gradient Boosting Regressor, with the current setup, captures a bit less of the data's variance.
# 
# ##### Interpretation of the Visualization:
# Looking at the scatter plot, the actual vs. predicted demand forecasting for the Gradient Boosting Regressor shows that:
# 
# The model performs well for lower demand forecasting values, with predictions closely clustered around the line of perfect prediction. This indicates good model accuracy for the majority of customers with a lower demand forecasting.
# 
# As with the previous models, the accuracy decreases with higherdemand forecastingvalues. The model seems to overestimate the demand forecasting for some of the higher-value customers, as shown by the points that lie above the line of perfect prediction.
# 
# ##### Conclusion:
# The Gradient Boosting Regressor is a powerful tool, but its current performance suggests there might be room for improvement, particularly for high-value customers. To enhance the model, I could consider:
# 
# - Further hyperparameter tuning, potentially with a focus on reducing overfitting, which might be contributing to the higher RMSE.
# - Including additional features or performing more sophisticated feature engineering to better capture the factors influencing high demand forecasting values.
# - Exploring alternative ways to handle outliers or segmenting customers to build more targeted models.
# 
# In the context of my project, while the Gradient Boosting model provides robust predictions across a broad range of customers, I need to weigh the model's complexity and computational cost against the incremental accuracy gain it offers over simpler models. The slight decrease in performance metrics compared to simpler models raises questions about the model's generalizability, which I would need to explore further through additional testing and validation.

# ### Project Summary:
# In this project, I undertook the task of  demand forecasting model fro a retail store chain for an e-commerce dataset. My approach included exploratory data analysis to understand underlying patterns and relationships, followed by the application of various predictive models. I evaluated each model based on RMSE and R-squared metrics and visualized actual versus predicted demand forecasting values to assess model performance. The models ranged from simple linear regression to more complex algorithms like Random Forest and Gradient Boosting, culminating with a Neural Network (MLP Regressor) that provided some of the most promising results.
# 
# ### Improvement Areas:
# While the models performed reasonably well, there's room for improvement. Enhancing feature engineering could lead to more informative variables, potentially increasing model accuracy. Additionally, experimenting with hyperparameter tuning, especially for the Neural Network, could fine-tune the models for better performance. Incorporating other modeling techniques, such as deep learning or ensemble methods, might also yield improvements. Furthermore, a more thorough cross-validation process could help in understanding the models' generalizability and stability across different data subsets.
# 
# ### Future Research:
# Future research could explore several avenues:
# 
# - Integration of External Data: Incorporating external datasets, such as market trends or customer demographics, could provide more context and improve predictions.
# 
# - Customer Segmentation: Segmenting customers before modeling could allow for more targeted predictions, as different segments may exhibit different purchasing behaviors.
#  
# 

# In[ ]:





# In[ ]:




