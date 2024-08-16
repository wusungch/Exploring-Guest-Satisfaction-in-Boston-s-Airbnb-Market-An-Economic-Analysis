#!/usr/bin/env python
# coding: utf-8

# # ECO225 Research Project : Analyzing Guest Satisfaction in Boston Airbnb Reviews
# # Project One
# 
# ## 1.1 Introduction
# The proliferation of the short-term rental market, with Airbnb at its helm, has piqued substantial academic curiosity, particularly within hospitality and tourism studies. This sector's expansion has led scholars to investigate the intricacies of guest satisfaction, a determinant crucial for the platform's success and the economic viability of its hosts. The scope of this research spans a comprehensive array of factors, including property attributes, host behaviors, and their broader implications on both guests and local communities.
# 
# Previous studies, such as those by Guttentag et al. (2018)[3] and Dogru et al. (2020)[2], have described the impact of amenities, location, and host responsiveness on enhancing guest satisfaction. These themes are expanded upon by Zervas et al. (2017)[6] through the lens of signaling theory, exploring the influence of host attributes, particularly superhost status, on establishing trust and perceived quality among guests. Complementing these perspectives, Gibbs et al. (2018)[3] apply the hedonic pricing model to dissect how property and location-specific features are factored into pricing strategies, subsequently affecting guest perceptions and satisfaction.
# 
# Amidst this landscape, the role of technology in mediating the hospitality experience, especially through online reviews and digital communication, has been scrutinized by Liang et al. (2017)[5]. Their work highlights the evolving nature of host-guest interactions and their impact on satisfaction levels. Despite these extensive explorations, a gap remains in holistically integrating economic theories with empirical evidence to explain guest satisfaction within the Airbnb ecosystem comprehensively. 
# 
# Therefore, at the heart of this investigation is the economic question: How do market dynamics, consumer preferences, and pricing strategies within Boston's Airbnb market influence guest satisfaction? This question prompts an examination of the main response variable, review_scores_rating, and its relationship with subratings on location, communication, and value, and with relevant predictors. The study further delves into how specific predictors impact subratings on location (review_scores_location), communication (review_scores_communication), and value (review_scores_value). These subratings offer a granular view of guest satisfaction, breaking it down into components directly influenced by host and listing characteristics.
# 

# ## 1.2. Data Cleaning/Loading
# ### 1.2.1 Data Loading

# Aiming to bridge this gap, this study leverages a dataset from Kaggle, which encompasses detailed listings, reviews, and calendar information for 3585 Airbnb listings in Boston [1]. This dataset provides an abundance of variables that describe the listings, their reviews, and the hosts, which the study will be working with.
# 
# Host characteristics such as host_has_profile_pic, host_identity_verified, host_is_superhost, host_response_rate, and host_since are investigated to understand their effects on guest satisfaction. These variables reflect the host's credibility, responsiveness, and experience, factors potentially pivotal in shaping guest experiences.
# 
# Listing features and location, including neighbourhood, amenities, price_per_guest, room_type, and property_type, are analyzed to understand their influence on guest preferences and satisfaction. This segment of the analysis explores the tangible aspects of the Airbnb experience, from the physical attributes of the listings to their geographical positioning.
# 
# Lastly, reviews, particularly comments, undergo topic modeling to extract prevalent themes discussed by guests. This qualitative analysis enriches the understanding of guest satisfaction by unveiling the topics most frequently highlighted in reviews, offering insights into guests' priorities and experiences.
# 

# In[2]:


# Data Loading
reviews = pd.read_csv('reviews.csv')
calendar = pd.read_csv('calendar.csv')
listings = pd.read_csv('listings.csv')


# In[1]:


#!jupyter nbconvert "Project 3-Copy2.ipynb" --to pdf --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_input_tags='{"remove_cell"}' --TemplateExporter.exclude_input_tags='remove_cell'
get_ipython().system('pip install notebook PyQtWebEngine nbconvert[qtpdf] --quiet')
get_ipython().system('pip install nbconvert[webpdf] --quiet')
get_ipython().system('pip install playwright  --quiet')
get_ipython().system('playwright install')
get_ipython().system('playwright install chromium')
get_ipython().system('jupyter nbconvert --to webpdf --allow-chromium-download "Project 3-Copy2.ipynb"')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import ast

# FOR MACHINE LEARNING: 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn import metrics
get_ipython().system('pip install -q gensim')
import gensim
import spacy
from textblob import TextBlob
from bs4 import BeautifulSoup
import requests
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# FOR LANGUAGE DETECTION:
get_ipython().system('pip install -q langdetect')

from langdetect import detect, DetectorFactory, LangDetectException

# FOR SENTIMENT ANALYSIS:
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# FOR COUNTING WORD FREQUENCIES:
from collections import Counter
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)  # For WordNet lemmatizer compatibility

# FOR TIME SERIES PLOTS
from matplotlib import dates as mdates
from scipy import stats

# FOR MAPPING
import geopandas as gpd
from matplotlib.colors import Normalize
from shapely.geometry import Point

# For stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.iolib.summary2 import summary_col

# For formatting tables
get_ipython().system(' pip install stargazer')
from stargazer.stargazer import Stargazer
from IPython.core.display import HTML
from IPython.display import HTML


# ### 1.2.2 Data Cleaning

# The data cleaning process for this study was designed to prepare the dataset for Ordinary Least Squares (OLS) regressions and machine learning (ML) model analyses in Section 4.1 and 4.2. This essential step aimed to ensure the data's suitability for statistical modeling, addressing issues like missing values, removal of irrelevant variables, and standardization of data formats. Through targeted interventions, the cleaning sought to refine the dataset to accurately reflect the variables critical for analyzing guest satisfaction determinants within Airbnb listings. This section outlines the data cleaning strategies implemented, demonstrating how each contributes to achieving a dataset ready for comprehensive analysis in uncovering factors that influence guest satisfaction.

# #### Removing Redundant Variables/Columns

# In[3]:


# Removing irrelevant columns in the listings dataframe
listings_col_to_drop = ['listing_url', 'scrape_id', 'last_scraped', 'thumbnail_url', 'medium_url', 
                        'picture_url', 'xl_picture_url', 'host_url', 'host_thumbnail_url', 'host_picture_url', 
                        'host_verifications', 'calendar_updated', 'calendar_last_scraped', 'jurisdiction_names'] 
listings = listings.drop(columns=listings_col_to_drop)


# To streamline the data cleaning and exploration process, meaningless columns in our datasets are removed for easier data handling and increase computational efficiency. The project involves working with three large datasets: reviews, calendar, listings. Listings has 95 variables, including many that are irrelevant and meaningless to our research, such as host_picture_url. By removing 14 of these irrelevant columns, the data's dimensionality is reduced, which can speed up jupyter notebook's computation times and allowing for more efficient use of computational resources. After removal, there are 81 columns remaining in the 'listings' dataset. 

# #### Removing Reviews that are Missing Comments

# In[4]:


# print(f'calendar includes columns: {calendar.columns}')
# print(f'reviews includes columns: {reviews.columns}')
# print(f'listings includes variables: {listings.columns}')

# Checking the number of null values in the comments column
print(reviews.isna().sum()) # There are 53 null values in the comments column
print(f'number of null values in reviews_scores_rating: {listings["review_scores_rating"].isna().sum()}')
print(f'number of rows in listings: {reviews.shape[0]}')

# Removing all rows that contain null values in the comments column
reviews = reviews.dropna(subset=['comments'])


# Before performing topic modeling, it's necessary to remove NA values from the dataset. NA values represent missing or undefined data that can disrupt the modeling process. LDA relies on a complete matrix of words and their frequencies across documents for its analysis. The presence of NA values can lead to errors in the computation, as the algorithm expects numerical input to calculate the distribution of topics across documents. Additionally, NA values can skew the results, as they do not contribute meaningful information to the analysis. By removing NA values, the integrity of the input data is maintained, ensuring that the LDA model operates on accurate and complete information, leading to more reliable and interpretable results.

# #### Removing non-English Reviews

# In[5]:


# # Ensuring consistent results from langdetect
# DetectorFactory.seed = 1

# # Creating a function to detect language
# def detect_language(text):
#     try:
#         return detect(text)
#     except LangDetectException:
#         return 'error'  # Return a string denoting error in detection

# # Applying the function to detect language
# reviews['language'] = reviews['comments'].apply(lambda x: detect_language(str(x)))

# print(f"number of rows in reviews that are non-english: {reviews[reviews['language'] != 'en'].shape[0]}")
# print(f"number of rows in reviews: {reviews.shape[0]}")

# # Filtering out non-English comments
# reviews = reviews[reviews['language'] == 'en']


# Following the removal of NA values, it is also essential to filter out non-English reviews from the dataset before conducting LDA for topic modeling. LDA assumes that the dataset consists of documents in a single language to accurately identify and group similar thematic content. Including reviews in multiple languages complicates the model's ability to discern patterns, as it might incorrectly associate words from different languages or fail to recognize them altogether. This can significantly impact the coherence of identified topics and the overall quality of the analysis. By ensuring that all reviews are in English, the consistency and interpretability of the topic modeling process are preserved, allowing for a focused analysis of thematic structures within the dataset.

# #### Converting 'Amenities' into a Countable List

# In[6]:


# Converting 'amenities' into a list that can be counted 
listings['amenities'] = listings['amenities'].str.replace('}', '')
listings['amenities'] = listings['amenities'].str.replace('{', '')
listings['amenities'] = listings['amenities'].str.replace('"', '')
listings['amenities'] = listings['amenities'].astype(str)
listings['amenities_list'] = listings['amenities'].str.split(',')
listings['amenities_count'] = listings['amenities_list'].str.len()


# The 'amenities' field underwent transformation from a string to a list format, followed by a count of amenities per listing. Quantifying amenities allows for direct comparisons across listings and provides insights into the impact of amenities on guest satisfaction and listing attractiveness.

# #### Price Per Guest Calculation

# In[7]:


# generate price_per_guest
## correct the #guests column in listings. There should be a minimum of 1 guest for every listing
listings['guests_included'] = listings['guests_included'].replace(0,1)

# Calculating price per guest for each listing so listing prices can be comparable
listings['price'] = listings['price'].str.replace('$', '')
listings['price'] = listings['price'].str.replace(',', '')
listings['price_cleaned'] = listings['price'].astype(float)
listings['price_per_guest'] = listings['price_cleaned']/listings['guests_included']


# Adjustments were made to ensure a minimum inclusion of one guest per listing, followed by calculating a normalized price per guest. Normalizing prices to a per-guest basis facilitates comparisons across listings of different sizes or capacities, aiding in the analysis of pricing strategies.

# #### Creation of Dummy Variables for Room Type, Property Type and Host Reponse Time

# In[8]:


# Creating dummy variables for room_type because linear regressions do not directly take in categorical variables
# Entire home/apt is the base group
room_type_dummies = pd.get_dummies(listings['room_type'], prefix='room_type', drop_first=True)
room_type_dummies = room_type_dummies.astype(int)
listings = pd.concat([listings, room_type_dummies], axis=1)
listings.rename(columns={'room_type_Entire home/apt': 'is_entire_home/apt'}, inplace=True)
listings.rename(columns={'room_type_Private room': 'is_private_room'}, inplace=True)
listings.rename(columns={'room_type_Shared room': 'is_shared_room'}, inplace=True)

# Creating dummy variables for room_type because linear regressions do not directly take in categorical variables
# Apartment is the base group
property_type_dummies = pd.get_dummies(listings['property_type'], prefix='', drop_first=True)
property_type_dummies = property_type_dummies.astype(int)
listings = pd.concat([listings, property_type_dummies], axis=1)


# Transforming the categorical data of 'room_type' and 'property_type' into binary dummy variables enables the inclusion of these factors in linear regression models. This transformation captures the effects of room and property types on guest satisfaction, highlighting preferences for specific accommodations.

# In[9]:


# Cleaning host_response_rate and converting it into a float
listings['host_response_rate'] = listings['host_response_rate'].str.replace('%', '').astype(float)
 
# host_response_time - is_few_hours, is_one_hour, is_one_day, is_mult_days - base group is mult_days
response_time_dummies = pd.get_dummies(listings['host_response_time'], prefix='', drop_first=True)
response_time_dummies = response_time_dummies.astype(int)
listings = pd.concat([listings, response_time_dummies], axis=1)


# Transformations were applied to the 'host_response_rate' to convert percentages into numerical values, and 'host_response_time' was converted into binary dummy variables. Quantifying these aspects allows for the assessment of host responsiveness on guest reviews, underscoring the importance of timely and effective communication.

# #### Downtown Area Indicator and Host Characteristics

# In[10]:


# Assigning neighborhoods in the boston downtown area under the 'downtown' category
# Since listings in downtown are likely to exhibit very different characteristics as those that are outside downtow, 
# I create an indicator variable to distinguish between downtown and non-downtown listings
downtown_areas = ['Downtown', 'Downtown Crossing', 'Financial District', 'Chinatown-Leather District', 'North End', 'West End', 
                  'Beacon Hill', 'Bay Village']

def is_downtown_assign(neighbourhood):
    if neighbourhood in downtown_areas:
        return 1
    else:
        return 0

listings['is_downtown'] = listings['neighbourhood'].apply(is_downtown_assign)

# Creating an indicator variable to differentiate listings whose host has a profile picture and those that do not
listings['host_has_profile_pic'] = listings['host_has_profile_pic'].map({'t': 1, 'f': 0})

# Creating an indicator variable to differentiate listings whose host has their identity verified 
listings['host_identity_verified'] = listings['host_identity_verified'].map({'t': 1, 'f': 0})

# Creating an indicator variable to differentiate listings whose host has their identity verified 
listings['host_is_superhost'] = listings['host_is_superhost'].map({'t': 1, 'f': 0})


# An indicator was established for listings in the downtown area, recognizing the unique appeal of such locations. Additionally, binary indicators for host profile pictures, identity verification, and superhost status were created. These indicators provide a detailed analysis framework for examining how specific host attributes and location influence listing performance and guest experiences.

# #### Host Experience Measurement

# In[11]:


# Creating a new column to count the number of months that the host has been hosting on Airbnb
from dateutil.relativedelta import relativedelta

listings['host_since'] = pd.to_datetime(listings['host_since'])
date_collected = pd.to_datetime('2016-09-07')

# Function to calculate the difference in months from a given date to January 2016
def months_from_jan_2016(date):
    diff = relativedelta(date_collected, date)
    return diff.years * 12 + diff.months

listings['host_since_months'] = listings['host_since'].apply(months_from_jan_2016)


# Calculating the duration of hosting experience offers insights into the learning curve and professional development of hosts over time. Including this temporal dimension in the regression analysis allows for examining the correlation between host experience and guest satisfaction, positing experience as a potential quality indicator.

# Preparing the dataset through specified cleaning and transformation steps is pivotal for the regression analyses to be conducted in Section 4.1 OLS Regression and Section 4.2 Machine Learing. Each step ensures the dataset's readiness and suitability for regression techniques, laying the groundwork for insights into the determinants of guest satisfaction on Airbnb. By refining the dataset, these steps bolster the reliability of the regression models, thereby assisting in a comprehensive exploration of the factors influencing guest experiences.

# ## 1.3 Summary Statistics Tables

# ### 1.3.1 Number of Reviews in Each Neighborhood 

# In[12]:


listings['neighbourhood'] = listings['neighbourhood'].replace(['Brookline', 'Cambridge', 'Chestnut Hill', 'Somerville'], 'Boston Neighboring Towns')

ext_list_col = ['id', 'neighbourhood']
ext_reviews_col = ['listing_id', 'date']
listings_short = listings[ext_list_col]
reviews_short = reviews[ext_reviews_col]
merged_df = listings_short.merge(reviews_short, left_on = 'id', right_on = 'listing_id', how = 'inner')
merged_df.drop(columns='id', inplace=True)
merged_df['date'] = pd.to_datetime(merged_df['date'])
total_num_reviews = len(reviews)
print(f"There are {total_num_reviews} total number of reviews.")
print("  ")
num_reviews_df = merged_df.groupby('neighbourhood')['listing_id'].count().reset_index()
num_reviews_df = num_reviews_df.rename(columns = {'listing_id':'Reviews Count'})

table_string = num_reviews_df.to_string(index=False)
title = "       Reviews Count for Each Neighborhood"
table_string_with_title = f"{title}\n\n{table_string}"
print(table_string_with_title)


# To ensure the integrity of comparing review ratings across Boston's neighborhoods, an initial step involves identifying neighborhoods with insufficient review counts. Neighborhoods with fewer than 50 reviews might not yield reliable sentiment score representations of guest satisfaction. In response, various approaches will be devised to address this issue.
# 
# The analysis of the table indicates that Brookline, Cambrdige, Chestnut Hill, and Somerville each have fewer than 50 reviews. Notably, these neighborhoods are adjacent to Boston but are not within its municipal boundaries, possibly explaining their lower review counts due to the scope of data collection. Moreover, the more suburban or semi-rural character of these four areas might contribute to a reduced presence of Airbnb listings compared to urban neighborhoods. I will establish a new neighborhood classification called 'Boston Neighboring Towns' and assign these four neighborhoods to it. After making these adjustments, the resulting table 'Reviews Count for Each Neighborhood' shows that all neighborhoods have over 50 reviews. With sufficient sample size, we can now compare the average sentiment scores across different neighborhoods. 

# ### 1.3.2 Distribution of Airbnb Reviews by Review Rating Ranges 

# In[13]:


reviews = pd.read_csv('reviews_cleaned.csv')


# In[14]:


#Define bins and labels for the sentiment score ranges
bins = [0, 25, 50, 75, 100]
labels = ['0 to 25', '26 to 50', '51 to 75', '76 to 100']
category_labels = ['Extremely Negative', 'Negative', 'Positive', 'Extremely Positive']

# Categorize sentiment scores into defined ranges
listings['rating_range'] = pd.cut(listings['review_scores_rating'], bins=bins, labels=labels, include_lowest=True)
# Generate summary table
summary_table = listings.groupby('rating_range')['review_scores_rating'].agg(['count']).reset_index()
summary_table.columns = ['Rating Score Range', 'Number of Listings']

# Calculate the proportion of total reviews
total_listings = summary_table['Number of Listings'].sum()
summary_table['Proportion of Total Listings (%)'] = (summary_table['Number of Listings'] / total_listings).round(4) * 100

summary_string = summary_table.to_string(index=False)

# Create the table
title = "Table 1: Distribution of Review Scores Rating"
table_string_length = len(summary_table.to_string(index=False).split('\n')[0])
title_string = title.center(table_string_length)
summary_with_title = f"{title_string}\n\n{summary_table.to_string(index=False)}"

print(summary_with_title)


# The distribution of Airbnb reviews, with an overwhelming 95.06% of listings scoring between 76 to 100, suggests a positivity bias in the feedback mechanism. Positivity bias refers to the tendency of guests to leave disproportionately positive reviews, possibly due to social desirability factors or a reluctance to post negative feedback unless the experience was significantly below expectations [2]. This bias can significantly impact our analysis by skewing our understanding of what drives guest satisfaction. Instead of a balanced view of strengths and weaknesses across listings, the data may overemphasize positive experiences, underrepresenting areas that require improvement.

# ### 1.3.3 Top 5 Priciest & Most Affordable Neighborhoods

# In[15]:


## Clean price column in listings
neigh_price_df = listings[['neighbourhood', 'price', 'guests_included']]
neigh_price_df['price'] = neigh_price_df['price'].str.replace('$', '')
neigh_price_df['price'] = neigh_price_df['price'].str.replace(',', '')
neigh_price_df['price'] = neigh_price_df['price'].astype('float')

# calculate price per guest (so that price is comparable between listings)
## create a new column by dividing price by number of guests
neigh_price_df['price_per_guest'] = neigh_price_df['price']/neigh_price_df['guests_included']

## Find mean price per guest for each neighborhood (rounded to 2 d.p.)
neigh_price_df = neigh_price_df.groupby('neighbourhood')['price_per_guest'].mean().reset_index()
neigh_price_df['price_per_guest'] = neigh_price_df['price_per_guest'].round(2)

# Create a new dataframe to store sentiment scores for each neighborhood
ext_list_col = ['id', 'neighbourhood', 'review_scores_rating']
listings_short = listings[ext_list_col]
listings_short = listings_short.groupby('neighbourhood')['review_scores_rating'].mean().reset_index()
listings_short['review_scores_rating'] = round(listings_short['review_scores_rating'], 2)

neigh_price_score_df = neigh_price_df.merge(listings_short, on='neighbourhood')

# top 5 most expensive neighborhoods on Airbnb
most_exp_5 = neigh_price_score_df.sort_values('price_per_guest', ascending = False).head(5)
most_exp_5_title = "Table 2: Top 5 Most Expensive Neighborhoods:\n"
most_exp_5_str = most_exp_5.to_string(index=False, header=["Neighbourhood", "  Average Price per Guest (USD)", "Average Rating"])
print(most_exp_5_title + "\n" + most_exp_5_str + "\n")

# top 5 least expensive neighborhoods with a title and custom column names
cheapest_5 = neigh_price_score_df.sort_values('price_per_guest').head(5)
cheapest_5_title = "Table 3: Top 5 Most Affordable Neighborhoods:\n"
cheapest_5_str = cheapest_5.to_string(index=False, header=["Neighbourhood", "  Average Price per Guest (USD)", "Average Rating"])
print(cheapest_5_title + "\n" + cheapest_5_str)


# The analysis of the most expensive and most affordable neighborhoods for Airbnb in Boston reveals a nuanced picture of the short-term rental market. The top five most expensive neighborhoods, led by Harvard Square with an average price per guest of 359.00USD, do not necessarily correlate higher prices with higher ratings. Notably, Harvard Square's average rating is not available (NaN), suggesting insufficient data or a lack of reviews, which contrasts with the Financial District, where a high average price per guest (269.65USD) accompanies a high average rating (98.25). This discrepancy indicates that while guests might be willing to pay premium prices in certain areas, the perceived value or satisfaction derived from their stay is not solely determined by cost.
# 
# Conversely, the most affordable neighborhoods, with Mattapan at an average price per guest of $60.98 and the highest-rated affordable neighborhood being Roslindale at 95.57, demonstrate that lower-priced listings do not equate to lower guest satisfaction. In fact, the presence of neighborhoods like Hyde Park and Roslindale in the affordable category, both with ratings exceeding 93, suggests that guests can find high satisfaction in listings priced significantly below those in more expensive areas.
# 
# From an economic standpoint, these findings challenge the assumption that price is a direct proxy for quality or satisfaction in the Airbnb market. The variation in average ratings across different price segments underscores the importance of factors beyond price in influencing guest satisfaction. For instance, the unique characteristics of a neighborhood, the quality of the listing, and the host's attentiveness could play significant roles. As we delve deeper into what factors most influence guest satisfaction, it becomes clear that economic reasoning must account for the complexity of guest preferences and the heterogeneity of their experiences. Recognizing the varied factors that guests value, beyond just the monetary cost, will be critical in further analysis, especially when considering the impact of positivity bias on ratings [2]. This approach ensures a more holistic understanding of the marketplace, guiding hosts on how to improve their offerings and compete effectively, not just on price but on the overall value proposition to guests.

# ### 1.3.4 Distribution of Property Types Across Neighborhoods

# In[16]:


import pandas as pd

# Assuming you have the 'review_scores_rating' column in the 'listings' DataFrame

property_df = listings.groupby(['is_downtown', 'property_type']).agg(property_count=('id', 'count'),
                                                                     average_rating=('review_scores_rating', 'mean')).reset_index()

property_df['total_property_count'] = property_df.groupby('is_downtown')['property_count'].transform('sum')
property_df['proportion'] = round(property_df['property_count'] / property_df['total_property_count'], 2)
property_df.drop('total_property_count', axis=1, inplace=True)

downtown_boston_df = property_df[property_df['is_downtown'] == 1]
downtown_boston_df.drop('is_downtown', axis=1, inplace=True)

not_in_boston_df = property_df[property_df['is_downtown'] == 0]
not_in_boston_df.drop('is_downtown', axis=1, inplace=True)

def append_totals(df):
    totals_df = pd.DataFrame([{  
        'property_type': 'All Types',
        'property_count': df['property_count'].sum(),
        'average_rating': round(df['average_rating'].mean(), 2),
        'proportion': round(df['property_count'].sum() / df['property_count'].sum(), 2)
    }])
    return pd.concat([df, totals_df], ignore_index=True)

# Apply the function to each DataFrame
downtown_boston_df = append_totals(downtown_boston_df)
not_in_boston_df = append_totals(not_in_boston_df)

downtown_boston_df['average_rating'] = round(downtown_boston_df['average_rating'], 2)
not_in_boston_df['average_rating'] = round(downtown_boston_df['average_rating'], 2)

print(" Table 4: Property Types Distribution within Downtown Area")
print(downtown_boston_df.to_string(index=False))

print('\n')

print(" Table 5: Property Types Distribution outside Downtown Area")
print(not_in_boston_df.to_string(index=False))


# The distribution of property types across different areas of Boston highlights distinct market characteristics and guest preferences. In Downtown Boston, apartments dominate the property landscape, making up 90% of all listings. This high proportion suggests a strong preference or demand for apartment stays in downtown areas, likely due to the proximity to business centers, tourist attractions, and entertainment options. The minimal presence of alternative property types like Bed & Breakfasts, Boats, or Villas underscores the urban preference for traditional apartment accommodations in densely populated areas.
# 
# Contrastingly, the property type distribution outside of downtown and not in Boston presents a more diverse array. While apartments still represent a significant share, especially outside downtown with a 62% proportion, there's a notable increase in the variety of accommodations. For instance, houses constitute a substantial 28% of listings outside downtown, reflecting perhaps a preference for more spacious or family-oriented accommodations in less urbanized areas. The presence of unique property types such as Bed & Breakfasts and Boats, albeit small, indicates niche markets catering to specific guest interests or seeking unique experiences.
# 
# The stark difference in property type distribution between downtown and other areas can be interpreted through economic reasoning. Downtown's dominance by apartments may reflect higher land costs and density, pushing for more compact living spaces. In contrast, the greater diversity and higher proportion of houses outside downtown areas might indicate lower land costs, allowing for larger properties. These dynamics suggest that property type availability and guest preferences are closely tied to the geographical and economic context of the listing location.
# 
# This analysis brings us closer to answering the research question on factors influencing guest satisfaction by indicating that property type—and by extension, the associated characteristics and amenities of different property types—plays a significant role in shaping guest preferences and satisfaction. Understanding these preferences is crucial for hosts looking to optimize their listings and for platforms aiming to match guest preferences with available accommodations effectively.

# ### 1.3.5 Most Common Words in Reviews

# In[17]:


# stop_words = set(stopwords.words('english'))

# lemmatizer = WordNetLemmatizer()

# # Define the function to remove stopwords and punctuations
# def preprocess_text(text):
#      # Convert text to lowercase
#      text = text.lower()
#      # Remove punctuation
#      text = text.translate(str.maketrans('', '', string.punctuation))
#      # Tokenize
#      tokens = word_tokenize(text)
#      # Remove stopwords
#      tokens = [word for word in tokens if word not in stopwords.words('english')]
#      # Lemmatize each word
#      lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
#      return lemmatized_tokens

# # Apply preprocessing to each review
# reviews['processed_reviews'] = reviews['comments'].apply(preprocess_text)

# # # Flatten the list of processed reviews into a single list
# all_words = [word for sublist in reviews['processed_reviews'] for word in sublist]

# # # Frequency analysis
# word_counts = Counter(all_words)

# # # Most common words
# most_common_words = word_counts.most_common()
# reviews.to_csv('processed_reviews.csv', index=False


# In[18]:


# Converting values in 'processed_reviews' back to list after reading file
reviews = pd.read_csv('processed_reviews.csv')
reviews['processed_reviews'] = reviews['processed_reviews'].apply(ast.literal_eval)

all_words = [word for sublist in reviews['processed_reviews'] for word in sublist]

# # Frequency analysis
word_counts = Counter(all_words)

# # Most common words
most_common_words = word_counts.most_common()

# Convert to DataFrame
df_most_common_words = pd.DataFrame(most_common_words, columns=['Word', 'Count'])

title = "Table 6: Most Common Words"
table_string_length = len(df_most_common_words.to_string(index=False).split('\n')[0])
title_string = title.center(table_string_length)
summary_with_title = f" {title}\n\n{df_most_common_words[0:15].to_string(index=False)}"

# Print as a table without the index
print(summary_with_title)


# The analysis of the most common words in Airbnb reviews after removing stop words and lemmatization reveals key factors contributing to guest satisfaction in Boston, notably emphasizing the importance of location, accommodation quality, and host interaction. Words like "great," "clean," "comfortable," and "location" not only highlight a positivity bias in guest feedback but also underscore the critical elements valued by guests—ease of access, cleanliness, and comfort. For future research, this positivity bias indicates a need for deeper qualitative analysis to discern between genuinely exceptional experiences and generally satisfactory ones. Understanding the nuances behind these common words through topic modeling could provide more detailed insights into improving guest satisfaction and tailoring services to meet guest expectations more effectively, thereby refining the strategies for Airbnb hosts and the platform in Boston.

# ### 1.3.6 Topic Modelling of Review Comments

# Shifting from analyzing common words in Airbnb reviews to applying topic modeling represents a strategic change in research methodology. The initial analysis focused on words such as "comfortable," "clean," and "location," shedding light on factors influencing guest satisfaction. The next step involves using Latent Dirichlet Allocation (LDA) for topic modeling to uncover themes within the reviews. This approach moves beyond simple word frequencies to examine thematic elements of guest feedback, providing a comprehensive view of guest experiences. This development is crucial for advancing the analysis of guest satisfaction, combining quantitative and qualitative data to highlight the main influences on Airbnb guest experiences.

# In[19]:


# # # Topic modelling on the entire reviews dataset
# import gensim
# from gensim import corpora
# from gensim.models.ldamodel import LdaModel
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# # Create a dictionary representation of the documents.
# dictionary = corpora.Dictionary(reviews['processed_reviews'])

# # Filter out words that occur in less than 20 documents, or more than 50% of the documents.
# dictionary.filter_extremes(no_below=20, no_above=0.5)

# # Convert the dictionary to a bag of words corpus
# corpus = [dictionary.doc2bow(text) for text in reviews['processed_reviews']]

# # Set parameters
# num_topics = 5
# passes = 20 

# # Perform LDA
# lda_model = LdaModel(corpus=corpus,
#                      id2word=dictionary,
#                      num_topics=num_topics,
#                      random_state=100,
#                      update_every=1,
#                      passes=passes,
#                      alpha='auto',
#                      per_word_topics=True)

# # Print the topics identified by LDA
# for idx, topic in lda_model.print_topics(-1):
#     print(f'Topic: {idx} \nWords: {topic}\n')

# # Save the model to disk
# lda_model.save('lda_model.model')


# For my topic modelling, I used a machine learning technique called Latent Dirichlet Allocation (LDA), which identifies themes within text data. The process involves preprocessing text to eliminate common words and reduce words to their base forms. A dictionary of unique words is then created, and words that are too common or too rare are filtered out. The text data is converted into a 'bag-of-words' model, where the frequency of each word in the dictionary is noted. 
# 
# The LDA model is trained on this numerical representation, with parameters set for the desired number of topics (in my case 5 topics) and iterations (5 interations) for convergence. LDA then assigns a distribution of topics to each document and words to each topic based on their occurrence patterns. Topics are identified by a list of words most representative of them, ranked by their association strength. The trained model can be saved for repetitive use, circumventing the need to retrain and allowing for consistent future analyses.
# 
# The topics generated from the LDA model seem to suggest distinct themes prevalent in Airbnb reviews:
# 
# **Topic 0 - The Feeling of Home**: This topic captures the emotional aspect of stays, where guests emphasize the feelings of warmth, welcome, and a homely atmosphere provided by their hosts. Words like "home," "feel," "made," and "lovely" point to a host’s ability to create a comfortable and inviting space.
# 
# **Topic 1 - Booking Process and Host Interaction**: Here, the focus is on the logistical side of Airbnb stays, including reservations and host communication. Words like "reservation," "canceled," "automated," and "host" suggest discussions about the booking process, possibly indicating instances where reservations were unexpectedly altered or canceled.
# 
# **Topic 2 - Accommodation Quality and Location**: This topic relates to the physical attributes of the stay, such as the room and the house's quality, and its proximity to areas of interest like "downtown." The importance of "walk" and "station" indicates the value guests place on accessibility.
# 
# **Topic 3 - Amenities and Comfort**: Keywords like "night," "bed," "bathroom," and "kitchen" focus on the amenities and comfort provided by the listing. Words such as "small" and "little" might indicate concerns or praises about the size and suitability of the accommodations.
# 
# **Topic 4 - Overall Experience and Satisfaction**: Reflecting general satisfaction, this topic includes terms commonly associated with positive experiences, such as "great," "stay," "place," "location," and "clean." It suggests a comprehensive assessment of the guests' stay experience.
# 
# These topics help answer my research question by providing insights into what aspects of an Airbnb stay are most talked about and how they potentially impact guest satisfaction. It appears that emotional experiences, logistical processes, quality of accommodation, and overall satisfaction are significant factors influencing guest reviews in Boston.
# 
# From the results of our LDA model, we observe that the quality of the interactions that the guests had with the host play a significant part in their overall experience satisfaction. Hence, incorporating host-related data into the analysis of Airbnb guest satisfaction can provide a deeper understanding of the elements that contribute to a positive guest experience. 
# 
# Host information such as occupation, hobbies, languages spoken, and educational background can significantly enrich the research for several reasons: Host occupation and hobbies could indicate the level of engagement or type of interaction a guest might expect, which could influence guest satisfaction. Language proficiency may affect communication clarity between host and guest, impacting the overall experience. The host's educational background, especially if linked to hospitality or management, could reflect in the quality of service provided.

# ## 1.4 Plots, Histograms, Figures

# Transitioning from the summary statistics and text analysis sections, the research moves towards creating visualizations to articulate and investigate the patterns and insights identified previously. The focus on temporal trends in guest satisfaction, distributions of ratings, and the relationship between price and review ratings stems from specific observations and questions that emerged from initial analyses.
# 
# The summary statistics section indicated trends and variations in guest satisfaction ratings, suggesting a visual exploration of these dynamics over time could reveal how factors external to Airbnb or changes within the platform influence guest satisfaction throughout the year.
# 
# The decision to visualize distributions of ratings follows from descriptive statistics indicating a need to understand the skewness, outliers, central tendency, and dispersion of guest satisfaction more clearly. This step aims to visually confirm hypotheses regarding high rating prevalence and its implications for understanding guest satisfaction.
# 
# Linking to the text analysis, which highlighted themes of location, cleanliness, host interaction, and comfort in guest reviews, there is a need to visually explore the relationship between price and review ratings. This approach seeks to examine if higher-priced listings correlate with higher satisfaction, potentially due to superior location or amenities, and how qualitative insights from text analysis reflect in the pricing strategies of hosts.
# 
# These visualizations are intended as a progression from earlier analyses, aiming to visually validate and contextualize the insights gained, offering a comprehensive view of guest satisfaction within Boston's Airbnb market.

# ### 1.4.1 Changes in Monthly Sentiment Scores over Time

# In[20]:


# Convert 'date' column to datetime format
reviews['date'] = pd.to_datetime(reviews['date'])

# Sort DataFrame by the 'date' column
reviews_sorted = reviews.sort_values('date')

# Set the date as the index
reviews_sorted.set_index('date', inplace=True)

# Resample to get monthly averages of the 'compound' scores
monthly_avg = reviews_sorted['compound'].resample('M').mean()

# Plotting the monthly averages
plt.figure(figsize=(10, 6))
plt.plot(monthly_avg.index, monthly_avg.values, marker='o', linestyle='-', color='red')

# Calculating the trend line
dates_numeric = mdates.date2num(monthly_avg.index)  # Convert dates to numeric format for polyfit
z = np.polyfit(dates_numeric, monthly_avg.values, 1)  # Fit the polynomial
p = np.poly1d(z)  # Create polynomial function

# Plot the trend line
plt.plot(monthly_avg.index, p(dates_numeric), color='r', linestyle='--', label='Trend Line')

plt.title("Figure 1: Temporal Trends in Guest Satisfaction: Monthly Average Sentiment Scores in Boston's Airbnb Market")
plt.xlabel('Date')
plt.ylabel('Average Compound Score')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.legend()

# Improve layout to accommodate the rotated date labels
plt.tight_layout()

# Display the plot
plt.show()


# The provided plot showcases the monthly average compound sentiment scores from a series of reviews over a span of several years. Notably, all the scores are positive, fluctuating between 0.80 and 0.95, which aligns with prior studies suggesting a general tendency for people to leave positive reviews. Despite the lack of a trend line in the visual, there appears to be a gradual decline in sentiment, particularly noticeable after mid-2013. This could indicate a shifting trend in customer feedback or changes in expectations over time. The scores exhibit considerable month-to-month volatility, yet there's no clear indication of seasonal patterns. Given the pre-existing bias towards positive reviews, the downward trend, even within the positive spectrum, could be indicative of a relative decrease in satisfaction or an alteration in the aspects being reviewed. This warrants a closer examination to understand the underlying factors contributing to this subtle yet apparent decline in average sentiment.

# ### 1.4.2 Distribution of Ratings

# In[21]:


plt.figure(figsize=(10, 5))
plt.hist(listings['review_scores_rating'].dropna(), bins=50, color='red')
plt.title('Figure 2: Distribution of the Ratings of Listings')
plt.xlabel('Review Rating')
plt.ylabel('Frequency')
plt.show()


# The histogram displays the distribution of ratings for Airbnb listings. There is a clear skew towards higher ratings, with the frequency of listings receiving ratings closer to 100 being substantially higher than those in the lower rating brackets. This distribution is a classic representation of a positivity bias, where guests tend to leave overwhelmingly positive ratings. 
# 
# Economically, such a distribution might indicate that guests feel they are receiving value for their money or it could suggest that guests are more likely to post reviews when they have a positive experience. However, the presence of this bias can pose challenges for analytical models aiming to predict guest satisfaction, as the range of variation in ratings is compressed towards the higher end of the scale. This could make it difficult to discern what factors most critically impact guest satisfaction since the ratings data may not sufficiently capture the nuances of guest experiences. Recognizing this skew is important for subsequent analysis, which might require methods to adjust for the bias or focus on other metrics that can provide more differentiation in guest feedback.

# ### 1.4.3 Relationship between Price and Review Ratings

# In[22]:


# Creating a new merged dataframe for geomapping the changes in sentiment scores across neighborhoods
ext_list_col = ['id', 'neighbourhood', 'price', 'guests_included', 'review_scores_rating']
#ext_reviews_col = ['listing_id', 'compound']
listings_short = listings[ext_list_col]
listings_short['price'] = pd.to_numeric(listings_short['price'], errors='coerce')

# calculate price per guest (so that price is comparable between listings)
## correct the #guests column in listings
listings_short['price_per_guest'] = listings_short['price']/listings_short['guests_included']

# DF: Average the sentiment scores for each neighborhood
listings_short1 = listings_short.groupby(['neighbourhood'])['price_per_guest'].mean().reset_index()
listings_short2 = listings_short.groupby(['neighbourhood'])['review_scores_rating'].mean().reset_index()

scatterplot_df = listings_short1.merge(listings_short2, on='neighbourhood')

plt.figure(figsize=(10, 6))
sns.regplot(x='price_per_guest', y='review_scores_rating', data=scatterplot_df, scatter_kws={'alpha':0.3})
plt.title('Figure 3: Scatterplot Price per Guest vs. Review Rating')
plt.xlabel('Price per Guest')
plt.ylabel('Review Rating')
plt.show()


# The scatterplot illustrates the relationship between the price per guest and review ratings for Airbnb listings. The trend line, along with the distribution of data points, suggests that there is a slight positive correlation between price and ratings, implying that higher-priced listings tend to have slightly higher review ratings. However, the broad confidence interval indicates a high degree of variability around this trend.
# 
# In economic terms, the positive correlation could be reflective of a premium on perceived quality or experience; guests might have higher expectations when paying more and rate their experiences favorably when those expectations are met. Conversely, the variability implies that price is not the sole determinant of satisfaction, and that other factors—such as location, amenities, or host quality—could have significant impacts on the guests' experiences.
# 
# Understanding this relationship is key to answering the research question regarding what influences guest satisfaction. While higher prices might generally align with higher satisfaction, the substantial spread of ratings across price points highlights the complexity of guest preferences. This insight underlines the need to consider a multi-faceted approach in further analysis, as the intersection of various factors likely contributes to the overall satisfaction beyond what the price alone can explain.

# # Project Two
# ## 2.1 The Message: Location is a strong and stable aspect of guest satisfaction in Airbnb properties, while cleanliness varies greatly, suggesting a need for hosts to focus on improving consistency in this area.

# In[23]:


# %%capture
# # Generating the sentiment scores of review comments
# sia = SentimentIntensityAnalyzer()

# # run the polarity score on the entire comments column
# res = {}
# for i, row in tqdm(reviews.iterrows(), total=len(reviews['comments'])):
#     text = row['comments']
#     myid = row['id']
#     res[myid] = sia.polarity_scores(text)

# # Merging the sentiment scores to the reviews dataset
# vaders = pd.DataFrame(res).T
# vaders = vaders.reset_index().rename(columns={'index':'id'})
# reviews = reviews.merge(vaders, how='left')

# reviews.to_csv('reviews_cleaned.csv', index=False)


# In[24]:


sia = SentimentIntensityAnalyzer()

def aspect_sentiment_scores(review, aspects):
    sentences = sent_tokenize(review)
    aspect_scores = {aspect: [] for aspect in aspects}
    
    for sentence in sentences:
        for aspect in aspects:
            if aspect in sentence:
                score = sia.polarity_scores(sentence)['compound']
                aspect_scores[aspect].append(score)
    
    return aspect_scores

aspects = ['location', 'cleanliness', 'comfort']
reviews['aspect_scores'] = reviews['comments'].apply(lambda x: aspect_sentiment_scores(x, aspects))

data = []

for period, group in reviews.groupby([reviews['date'].dt.year, reviews['date'].dt.month]):
    period_scores = {aspect: [] for aspect in aspects}
    
    for _, row in group.iterrows():
        for aspect in aspects:
            period_scores[aspect].extend(row['aspect_scores'][aspect])
    
    # Calculate the mean sentiment score for each aspect, considering None for empty lists
    weighted_scores = {aspect: np.mean(scores) if scores else None for aspect, scores in period_scores.items()}
    weighted_scores['date'] = pd.to_datetime(f"{period[0]}-{period[1]}")
    
    data.append(weighted_scores)

aspect_scores_over_time = pd.DataFrame(data)


# In my initial exploration (Project 1 Section 3 Summary Statistics Tables) of the most common words in review comments, we identified 'location', 'cleanliness', and 'comfort' were among the highest-frequency words mentioned. Hence, we use this information to narrow our research focus to these three areas. In this section, I will identify how the the average sentiment for these three aspects change over time. To achieve this, we began by meticulously identifying mentions of these key aspects within the review texts. I extracted sentences containing these terms and calculated sentiment scores for each mention, providing a granular view of the guests' emotions and opinions related to each aspect.
# 
# Once I had these sentiment scores, my next objective was to aggregate them across time, mindful that mere counting would not suffice. I needed to weigh the scores to reflect the frequency and emphasis guests placed on each aspect. By creating a weighted average sentiment score for each aspect, I could gauge not only the sentiment's positivity or negativity but also the degree of attention the reviewers devoted to each aspect.
# 
# With the sentiment scores weighted and aggregated over time, I turned my attention to visualization. Crafting a time series plot was paramount in revealing the trends and shifts in guest sentiment.
# 
# One problem that I faced with the plot was that there were certain months where none of the reviews mentioned any of the three aspects 'location', 'cleanliness', and 'comfort', causing null values in my data and hence 'gaps' in the time series plot. In addressing this, I implemented a methodical data interpolation approach. Firstly, I established a comprehensive date range spanning the entirety of my dataset's timeline, ensuring that each period was accounted for, even if no reviews were present. I then utilized pandas' reindexing capabilities to align our sentiment score data with this continuous date range, introducing `None` for missing periods to maintain the sequence.
# 
# Recognizing the potential disruption that these null values posed to our trend analysis, I proceeded with a linear interpolation strategy. This method estimates the missing values by linearly weighting the known data points before and after the gap, thereby creating a seamless transition across all time frames. Through this interpolation, I  smoothed the sentiment score trends, eliminating the visual discontinuities in the plot and allowing for an uninterrupted analysis of temporal patterns.

# In[25]:


import matplotlib.dates as mdates
aspect_scores_over_time['date'] = pd.to_datetime(aspect_scores_over_time['date'])
aspect_scores_over_time.interpolate(method='linear', inplace=True)

plt.figure(figsize=(14, 7))

for aspect in ['location', 'cleanliness', 'comfort']:
    plt.plot(aspect_scores_over_time['date'], aspect_scores_over_time[aspect], label=aspect.capitalize(), marker='o', linewidth=2)

# Formatting the plot
plt.title('Figure 4: Temporal Variations in Sentiment Scores for Location, Cleanliness, and Comfort', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Weighted Sentiment Score', fontsize=14)
plt.legend(fontsize=12)

# Improve the date ticks
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

# Add a grid for easier reading
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout() 
plt.show()


# The resulting plot now presents a coherent visual narrative, revealing the evolution of guest sentiments towards 'location', 'cleanliness', and 'comfort' over the period 2009 to 2017. These sentiment scores (-1 to 1, with -1 being the most negative and 1 being the most positive sentiment) are plotted on the y-axis against date(year) on the x-axis. The sentiment scores are "weighted," implying some form of adjustment or scaling has been applied, possibly to account for the volume of comments or the significance of the sentiment expressed.
# 
# There are several interpretations that we can make from the plot: 
# 
# 1. **Location**: The blue line represents Location, which has the highest and most stable sentiment scores throughout the observed period. It starts just above 0.6 in 2009, rises sharply to peak near 1.0 in 2010, and then fluctuates with minor variability around 0.8 for the remainder of the timeline. The high and stable scores for Location suggest that guests consistently rate the location of Airbnb properties positively, and it seems to be a strong point for hosts.
# 
# 2. **Cleanliness**: The orange line represents Cleanliness, which exhibits the most volatility. The sentiment scores for Cleanliness start just above 0, indicating a neutral or mixed sentiment in 2009. There is a significant drop in late 2010, reaching into negative sentiment territory, which could indicate a period of widespread dissatisfaction with cleanliness among Airbnb properties. Following this dip, there's a dramatic improvement peaking in 2012 before another drop. Overall, from 2013 onwards, there is an upward trend with some fluctuations, suggesting a general improvement in guest satisfaction with cleanliness over time.
# 
# 3. **Comfort**: The green line represents Comfort, which shows moderate volatility. Starting around 0.4 in 2009, Comfort's sentiment score quickly rises and then experiences ups and downs throughout the years, albeit within a narrower range than Cleanliness. It remains above the score for Cleanliness for almost the entire period, suggesting that guests have found Airbnb properties to be relatively comfortable. There's an apparent dip around 2013, after which the scores recover and remain somewhat stable with slight improvements towards 2017.
# 
# In terms of the overall trend, all three aspects show improvement over the eight-year span, with Location consistently rated the highest. Cleanliness has the most room for improvement, given its volatility and the significant negative sentiment recorded at certain points. Comfort also shows room for enhancement but has been perceived better than Cleanliness throughout the period.
# 
# The plot is useful for understanding guest priorities and satisfaction levels over time. The spikes and troughs could correlate with specific events or changes in Airbnb's policies or market trends. For instance, a sudden drop in sentiment for Cleanliness could be due to a publicized issue with property standards, and the subsequent recovery could indicate effective remedial measures taken by the platform or its hosts.

# ## 2.2 Maps & Interpretations

# ### 2.2.1 Geoghraphical Distribution of Airbnb Listings in Boston

# In[26]:


listings['Coordinates'] = list(zip(listings.longitude, listings.latitude))
listings['Coordinates'] = listings['Coordinates'].apply(Point)
gdf = gpd.GeoDataFrame(listings, geometry='Coordinates')

# Read the Boston street segments shapefile
boston_map = gpd.read_file('Boston_Street_Segments.shp')

# Plot the map
fig, ax = plt.subplots(figsize=(15, 15))

# Plot the base map with light gray color and black edges
base = boston_map.plot(ax=ax, color='grey', edgecolor='white', gapcolor = 'white')

ax.set_xlabel('Longitude', fontsize=10)
ax.set_ylabel('Latitude', fontsize=10)
ax.set_title('Figure 8: Geographical Distribution of Listings', fontsize=20)

# Determine the bounds for zooming in
bounds = gdf.total_bounds
buffer = 0.01

ax.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
ax.set_ylim(bounds[1] - buffer, bounds[3] + buffer)

# Plot the Airbnb listings on top of the base map
gdf.plot(ax=ax, marker='o', color='red', markersize=5, label='Airbnb Listings', zorder=3)

# Add a legend
ax.legend()

# Show the plot
plt.show()


# The map displays a visualization of Airbnb listings in Boston, with red dots marking the location of each property. The clusters of red dots are noticeably denser in certain areas like Downtown, Back Bay, and along the Charles River, suggesting these are the most popular neighborhoods for Airbnb rentals, most likely due to their appeal to tourists and accessibility. The distribution of listings across the city provides a snapshot of the short-term rental market landscape, highlighting areas where guests are most likely to stay during their visit to Boston.

# ## 2.2.2 Distribution of Ratings Across Neighborhoods

# In[27]:


# Creating a new merged dataframe for geomapping the changes in sentiment scores across neighborhoods
ext_list_col = ['id', 'neighbourhood', 'review_scores_rating']
#ext_reviews_col = ['listing_id', 'date', 'compound']
listings_short = listings[ext_list_col]
#reviews_short = reviews[ext_reviews_col]
#mapping_df = listings_short.merge(reviews_short, left_on = 'id', right_on = 'listing_id', how = 'inner')
#mapping_df.drop(columns='id', inplace=True)
#mapping_df['date'] = pd.to_datetime(mapping_df['date'])

# Assign all downtown listings to 'Downtown' (just for this mapping)
listings_short['neighbourhood'] = listings_short['neighbourhood'].replace(['South End', 'Downtown Crossing', 'Theater District', 'Leather District', 'Financial District', 'Government Center'], 'Downtown')

# DF1: Averaging the sentiment scores for each listing
listings_short1 = listings_short.groupby(['id', 'neighbourhood'])['review_scores_rating'].mean().reset_index()

# DF2: Average the sentiment scores for each neighborhood for each year
#mapping_df['year'] = mapping_df['date'].dt.year
listings_short2 = listings_short1.groupby(['neighbourhood'])['review_scores_rating'].mean().reset_index()


# In[28]:


# Plotting the geomap
gdf_neighbor = gpd.read_file('Census2020_BG_Neighborhoods.shp')
gdf_neighbor = gdf_neighbor.merge(listings_short2, left_on='BlockGr202', right_on='neighbourhood')

# Plot the choropleth map
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gdf_neighbor.plot(column='review_scores_rating', ax=ax, legend=True, cmap='coolwarm', legend_kwds={'label': "Average Review Rating"},  
                  norm=Normalize(vmin=70, vmax=100))

# Add neighborhood borders
gdf_neighbor.boundary.plot(ax=ax, color='black', linewidth=0.2)

# Add neighborhood names as labels
for idx, row in gdf_neighbor.iterrows():
    plt.annotate(text=row['neighbourhood'], xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
                 ha='center', va='center', fontsize=7)

# Add titles and labels
ax.set_title('Figure 9: Distribution of Airbnb Rating by Neighborhood in Boston')
ax.set_axis_off()
plt.show()


# The provided plot shows the distribution of the average ratings across different neighborhoods in Boston. Note that since there is an overwhelmingly high proportion of positive ratings across all reviews, the color range is deliberately adjusted from [0, 100] to [70, 100] to make the color variations between neighborhoods more distinguished for visualization purposes. As discussed previously, this overwhelmingly positive rating can be explained by 'positivity bias' proposed by previous literature, where people have a tendency to leave positive reviews [2].
# 
# Despite the presence of positivity bias—wherein guests are generally predisposed to leave positive feedback—the slight variations in the ratings are still telling. The map shows that Charlestown, Jamaica Plain, Roslindale, and West Roxbury have the highest overall ratings in their review. While West End, Hyde Park, Mattapan, and Mission Hill have the lowest overall ratings. 
# 
# Charlestown’s historical significance and iconic landmarks may imbue a sense of charm and nostalgia in guests' experiences, thereby nudging ratings upward. Jamaica Plain, with its eclectic mix of culinary delights and cultural diversity, could engage guests in a way that compels them to share more enthusiastic reviews. Similarly, Roslindale and West Roxbury are perceived as serene, family-friendly locales offering a respite from the city’s hustle—this tranquility can translate into slightly more positive reviews.
# 
# In contrast, neighborhoods like West End, Hyde Park, Mattapan, and Mission Hill, though still positive, have registered lower relative ratings. These areas, possibly due to a denser urban environment, fewer tourist attractions, or a stronger local rather than tourist presence, may not elicit the same level of enthusiasm in reviews.
# 

# ## 2.2.3 Distribution of Price Per Guest Across Neighborhoods

# In[29]:


# Plotting the geomap
gdf_neighbor = gpd.read_file('Census2020_BG_Neighborhoods.shp')
gdf_neighbor = gdf_neighbor.merge(neigh_price_df, left_on='BlockGr202', right_on='neighbourhood')

# Plot the choropleth map
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
gdf_neighbor.plot(column='price_per_guest', ax=ax, legend=True, cmap='Reds', legend_kwds={'label': "Average Price per Guest per Night($/person/night)"},  )

# Add neighborhood borders
gdf_neighbor.boundary.plot(ax=ax, color='black', linewidth=0.2)

# Add neighborhood names as labels
for idx, row in gdf_neighbor.iterrows():
    plt.annotate(text=row['neighbourhood'], xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
                 ha='center', va='center', fontsize=7)

# Add titles and labels
ax.set_title('Distribution of Airbnb Price Per Guest Per Night by Neighborhood in Boston')
ax.set_axis_off()
plt.show()


# The map visualizes the distribution of Airbnb prices per guest per night by neighborhood in Boston. Darker shades indicate higher average prices, with neighborhoods such as West End, Back Bay, and Chinatown displaying the most intense colors, suggesting they are among the priciest areas for Airbnb stays. In contrast, neighborhoods like Hyde Park, Roslindale, and Mattapan are represented in lighter shades, indicating more affordable options for guests. This spatial price pattern might reflect the varying demand influenced by factors such as proximity to city attractions, access to transport, and neighborhood reputation. It also suggests a potential correlation between price and perceived value or desirability, which can be further explored in conjunction with sentiment analysis to understand how pricing influences guest satisfaction.

# # Project Three
# ## 3.1.1 Potential Data to Scrape

# A viable source of data for scraping includes the individual profiles of Airbnb hosts, which typically feature personal narratives about the hosts themselves. These narratives often detail the host's profession, hobbies, languages they can speak, and their educational background.
# 
# To incorporate this newly scraped data, information about each host will be aligned with the existing listing IDs within the dataset. This enhancement of the dataset will facilitate a more comprehensive analysis by introducing host-specific variables as potential predictors of guest satisfaction.
# 
# Incorporating this data allows for an assessment of how host attributes may affect guest reviews. For example, it could be discovered that guests have a preference for hosts with whom they share common interests or those who are multilingual. Additionally, the educational institution a host attended might serve as an indicator of the host's reliability and the degree of trust that guests are willing to extend.

# ## 3.1.2 Potential Challenges
# 
# Scraping each host's information from their individual profiles pose several major challenges:
# 
# Firstly, Airbnb's website is equipped with formidable anti-scraping measures to protect user data and maintain privacy. This includes sophisticated technologies like CAPTCHA systems and dynamic content that loads through JavaScript, making it incredibly hard to access the data I need. The variability and unstructured nature of the data presented another hurdle. Host profiles vary greatly in detail and format, posing a significant challenge to standardizing and cleaning the data for analysis.
# 
# Secondly, it involves scraping data from multiple webpages, specifically navigating through Airbnb's extensive listings to collect host information. This task requires a method to programmatically move from one page to the next, a skill I haven't fully mastered yet. Unlike extracting data from a single page, multi-page web scraping involves understanding the structure of the website's navigation and effectively automating the process to iterate over listings, which can vary widely in their URL patterns and page layouts.
# 
# Thirdly, the sheer volume and diversity of Airbnb's listings add another layer of complexity to my project. As someone who hasn't yet mastered scraping data from multiple webpages or navigating through different sections of a website programmatically, tackling a platform as vast and varied as Airbnb presents a significant challenge. Each listing's webpage has its unique URL and layout variations, especially in the "Host" section where personal information is provided. Learning to automate the process of moving between these pages, extracting relevant data without missing or duplicating information, requires advanced techniques in web scraping that I'm currently striving to master.
# 
# For the sake of completing the project in time, I will not be webscraping the data that I described above, but the practice dataset that describes the median household income by neighborhood in Boston. 

# ## 3.1.3 Scraping Data from a Website

# For this section, I will scrape a table from [https://statisticalatlas.com/neighborhood/Massachusetts/Boston/South-Boston/Household-Income] that lists the median household income by neighborhood in Boston. The median household income of a neighborhood may subtly color the expectations and consequently the ratings of Airbnb guests. Affluent areas often come with the anticipation of premium stays, and when these expectations are met, it reflects positively in the review ratings. However, should high-end listings in such neighborhoods fall short of these high standards, the discrepancy between expectation and reality can result in negative rating, despite the overall wealth of the area.
# 
# Conversely, neighborhoods with lower median incomes might set a different standard for guests, who could prize value for money and local authenticity. A satisfactory stay that provides safety, comfort, and a genuine local experience at a reasonable cost could yield positive reviews. Thus, the economic profile of a neighborhood doesn't just shape visitor expectations—it's a backdrop against which the guest's entire experience is judged, influencing their review ratings. 

# In[30]:


import requests
from bs4 import BeautifulSoup

url = "https://statisticalatlas.com/neighborhood/Massachusetts/Boston/South-Boston/Household-Income"

response = requests.get(url)

soup = BeautifulSoup(response.content, "html.parser")

content = soup.find("div", {"id":"figure/neighborhood-in-boston/median-household-income"})
rows = content.find_all('text', {"font-family":"sans-serif"})

neighborhoods = content.find_all('text', {
    'font-family': 'sans-serif',
    'font-size': '13',
    'font-style': 'normal',
    'font-weight': 'normal',
    'text-anchor': 'end',
    'text-decoration': 'none', 
    'x': '94'
})

incomes = content.find_all('text' , {'fill-opacity':"0.400",'fill':'#000'})

growth = content.find_all('text', {'fill-opacity': '0.500', 'text-anchor': 'end', 'x': '368.5', 'fill': '#000'})
    
neighborhoods_list = []
incomes_list = []
growth_list = [] 

for i in range(len(neighborhoods)):
    neighborhoods_list.append(neighborhoods[i].get_text())
    incomes_list.append(incomes[i].get_text())
    growth_list.append(growth[i].get_text())
    
neighborhoods_list.insert(17, 'South Boston')
incomes_list.insert(-1, '$37.3k')
growth_list.insert(-1, '-63.7%')

dic = {'neighborhood': neighborhoods_list, 
       'median household income': incomes_list,
      'annual growth (%)': growth_list}
income_by_neighborhood = pd.DataFrame(dic, index = None)

#convert values in growth list to float
income_by_neighborhood['annual growth (%)'] = income_by_neighborhood['annual growth (%)'].str.replace('+', '')
income_by_neighborhood['annual growth (%)'] = income_by_neighborhood['annual growth (%)'].str.replace('%', '')
income_by_neighborhood['annual growth (%)'] = income_by_neighborhood['annual growth (%)'].astype('float')

# convert neighborhood median household income to float
income_by_neighborhood['median household income'] = income_by_neighborhood['median household income'].str.replace('$', '')
income_by_neighborhood['median household income'] = income_by_neighborhood['median household income'].str.replace('k', '')
income_by_neighborhood['median household income'] = income_by_neighborhood['median household income'].astype(float)
income_by_neighborhood['median household income'] = income_by_neighborhood['median household income'] * 1000

# Match the names with my original dataset
income_by_neighborhood['neighborhood'] = income_by_neighborhood['neighborhood'].replace('Leather Dist', 'Leather District')
income_by_neighborhood['neighborhood'] = income_by_neighborhood['neighborhood'].replace('Fenway', 'Fenway/Kenmore')
income_by_neighborhood['neighborhood'] = income_by_neighborhood['neighborhood'].replace('Government Ctr', 'Government Center')
income_by_neighborhood['neighborhood'] = income_by_neighborhood['neighborhood'].replace('Allston', 'Allston-Brighton')
income_by_neighborhood['neighborhood'] = income_by_neighborhood['neighborhood'].replace('N Dorchester', 'Dorchester')
income_by_neighborhood['neighborhood'] = income_by_neighborhood['neighborhood'].replace('E Somerville', 'Somerville')
income_by_neighborhood['neighborhood'] = income_by_neighborhood['neighborhood'].replace('E Cambridge', 'Cambridge')
income_by_neighborhood['neighborhood'] = income_by_neighborhood['neighborhood'].replace('Dntn Crossing', 'Downtown Crossing')

#rename column
income_by_neighborhood = income_by_neighborhood.rename(columns = {'median household income':'median household income ($USD)'})


# The dataset concerning median household incomes by neighborhood was sourced from the Statistical Atlas website, specifically the page for South Boston's household income. The scraping process involved sending a GET request to the URL and parsing the HTML content using BeautifulSoup, a Python library that facilitates the extraction of data from HTML and XML documents.
# 
# My scraping focused on extracting three key pieces of information: the name of the neighborhood, its median household income, and the annual growth rate. This data was found within specific HTML 'text' elements, identified by their unique attributes, such as 'font-family', 'font-size', and 'x' position values.
# 
# Once the data was collected, I cleaned and processed it to ensure compatibility with analysis needs. This involved converting the growth percentage and median household income into float data types, accounting for symbols such as "$" and "k", and matching the neighborhood names with those used in the original dataset to ensure consistency.
# 
# The dataset was then structured into a pandas DataFrame. The final DataFrame consists of three columns—'neighborhood', 'median household income', and 'annual growth (%)'— which is later used to analyze the economic characteristics of each neighborhood in relation to Airbnb listings and guest satisfaction.

# ## 3.1.4 Visualizing the Scraped Dataset

# In[31]:


listings.loc[listings['neighbourhood'].isin(['Theater District', 'Financial District']), 'median household income'] = 168600.0
listings = listings.merge(income_by_neighborhood,
                          how='left',
                          left_on='neighbourhood',
                          right_on='neighborhood')

# change median household income ($USD) to med_house_income
listings.rename(columns={'median household income ($USD)': 'med_household_income'}, inplace=True)

income_neigh_df = listings[['neighbourhood', 'med_household_income']]

import matplotlib.pyplot as plt

income_neigh_df = listings[['neighbourhood', 'med_household_income']]

income_neigh_df = income_neigh_df[income_neigh_df['neighbourhood'].notna()]
col_to_drop = ['Theater District', 'Financial District', 'Brookline', 'Harvard Square']
income_neigh_df = income_neigh_df[~income_neigh_df['neighbourhood'].isin(col_to_drop)]

# Sorting the DataFrame by median household income
income_neigh_df.sort_values(by='med_household_income', ascending=False, inplace=True)

# Convert 'neighbourhood' column to strings
income_neigh_df['neighbourhood'] = income_neigh_df['neighbourhood'].astype(str)

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(income_neigh_df['neighbourhood'], income_neigh_df['med_household_income'], color='skyblue')
plt.xticks(rotation=90)
plt.xlabel('Neighbourhood')
plt.ylabel('Median Household Income ($USD)')
plt.title('Figure 5: Median Household Income by Neighbourhood')
plt.tight_layout()
plt.show()


# "Figure 5" depicts the median household income by neighborhood, an economic indicator that could potentially correlate with the perceived quality of a location—a topic extensively discussed in the previous LDA section. As guests frequently highlight 'location' in their reviews, the implication is that neighborhoods with higher median incomes may offer characteristics that guests associate with higher quality, such as safety, convenience, and amenities. 
# 

# In[32]:


listings_location_rating_by_neigh = listings.groupby('neighbourhood')['review_scores_location'].mean().reset_index()

listing_income_merged = listings_location_rating_by_neigh.merge(income_by_neighborhood, how = 'left', left_on = 'neighbourhood', right_on = 'neighborhood')

# Assign the median household income of 'Downtown' to 'Financial District' and 'Theater District'
downtown_income = income_by_neighborhood[income_by_neighborhood['neighborhood'] == 'Downtown']['median household income ($USD)'].iloc[0]
downtown_growth = income_by_neighborhood[income_by_neighborhood['neighborhood'] == 'Downtown']['annual growth (%)'].iloc[0]

listing_income_merged.loc[listing_income_merged['neighbourhood'].isin(['Theater District', 'Financial District']), 'median household income ($USD)'] = downtown_income
listing_income_merged.loc[listing_income_merged['neighbourhood'].isin(['Theater District', 'Financial District']), 'annual growth (%)'] = downtown_growth
listing_income_merged = listing_income_merged.dropna(subset=['neighborhood'])

plt.figure(figsize=(15, 10))
plt.xlabel('Median Household Income ($USD)')
plt.ylabel('Review Scores Location')
plt.title('Figure 6: Relationship between Median Household Income and Location Review Scores by Neighborhood')
plt.xticks(rotation=45)
plt.tight_layout() 

plt.scatter(listing_income_merged['median household income ($USD)'], listing_income_merged['review_scores_location'])

for index, row in listing_income_merged.iterrows():
    plt.annotate(row['neighborhood'], (row['median household income ($USD)'], row['review_scores_location']), fontsize = 10)
    
m, b = np.polyfit(listing_income_merged['median household income ($USD)'], listing_income_merged['review_scores_location'], 1)
plt.plot(listing_income_merged['median household income ($USD)'], m*listing_income_merged['median household income ($USD)'] + b, color='red')  # Add line of best fit


# Figure 6 displays a scatterplot correlating median household income with location review scores for various Boston neighborhoods, suggesting a potential link between the economic status of an area and guests' location satisfaction. The upward trend in the plot indicates that as median household income increases, so does the average score for location reviews. This could imply that guests perceive neighborhoods with higher incomes as better locations, possibly due to associated factors like safety, amenities, and overall appeal. However, it's important to note that while there's a visual trend, the scatterplot alone cannot confirm causality or the strength of this relationship. This initial observation will be examined in depth using OLS regression and machine learning techniques in subsequent sections of the research to determine the statistical significance and to control for other variables that might influence these scores.
# 

# ## 3.2.1 Adding a New Dataset

# For this part of the research, I added some new data from https://data.boston.gov/dataset/open-space, which shows the number of open space amenities in each neighborhood. The goal of this section is to investigate whether there having more open space amenities around the neighborhood improves guest satisfaction. 

# In[33]:


sites = pd.read_csv('Open_Space.csv')
neigh_sites_df = sites.groupby(['DISTRICT', 'TypeLong']).size().reset_index(name='Count')

downtown_neighborhoods = ['South End', 'Downtown Crossing', 'Theater District', 
                          'Leather District', 'Financial District', 'Government Center', 
                          'Chinatown', 'North End', 'West End', 'Back Bay']

# Function to categorize neighborhoods
def categorize_neighborhood(neigh):
    if neigh in downtown_neighborhoods:
        return 'Downtown'
    elif neigh == 'West Roxbury':
        return 'Roxbury'
    else:
        return neigh

listings['neigh_big'] = listings['neighbourhood'].apply(categorize_neighborhood)

neigh_sites_df['DISTRICT'] = neigh_sites_df['DISTRICT'].replace('Back Bay/Beacon Hill', 'Back Bay')
neigh_sites_df['DISTRICT'] = neigh_sites_df['DISTRICT'].replace('Central Boston', 'Downtown')
neigh_sites_df['DISTRICT'] = neigh_sites_df['DISTRICT'].replace('Fenway/Longwood', 'Fenway/Kenmore')


# In[34]:


# Pivot the dataframe
pivoted_df = neigh_sites_df.pivot(index='DISTRICT', columns='TypeLong', values='Count').reset_index()
# Fill missing values with 0
pivoted_df = pivoted_df.fillna(0)
pivoted_df = pivoted_df.drop('Other Open Land', axis = 1)

pivoted_df['neigh_large'] = pivoted_df['DISTRICT'].apply(categorize_neighborhood)

# Group by 'neigh_large' and sum the values for each group
combined_df = pivoted_df.groupby('neigh_large').sum().reset_index()

pivoted_df['neigh_large'] = pivoted_df['DISTRICT'].apply(categorize_neighborhood)

# Drop the 'DISTRICT' column
pivoted_df.drop(columns=['DISTRICT'], inplace=True)

# Group by 'neigh_large' and sum the values for each group
combined_df = pivoted_df.groupby('neigh_large').sum().reset_index()
# merge with original listings dataset
listings = pd.merge(listings, pivoted_df, left_on='neigh_big', right_on='neigh_large', how='inner')


# In[35]:


combined_df.set_index('neigh_large', inplace=True)
combined_df.plot(kind='bar', stacked=True, figsize=(10, 8))
plt.ylabel('Count')
plt.xlabel('Neighborhood')
plt.title('Figure 7: Distribution of Open Space Counts Across Neighborhoods')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# The bar plot shows how the number of open space amenities differ among neighborhoods. It shows the count of spaces such as parks, playgrounds, urban wilds, and others, which are indicators of a neighborhood's amenity offerings and recreational opportunities.
# 
# From this data, we observe that certain areas, like Dorchester and West Roxbury, boast a higher number of parks, playgrounds, and athletic fields, suggesting these neighborhoods may offer more outdoor and family-oriented activities. This could positively influence Airbnb guests' experiences, potentially reflected in higher sentiment scores in reviews related to these areas. Conversely, neighborhoods with fewer open spaces might provide a different kind of appeal, such as a more urban experience, which may or may not be as highly rated in terms of sentiment scores.
# 
# Conducting further analysis on the link between open spaces and guest satisfaction could be revealing. If sentiment scores from Airbnb reviews are higher for neighborhoods with more parks and recreational facilities, this could indicate a preference for such amenities among guests. This insight would be particularly useful for Airbnb hosts and policymakers to understand the value guests place on accessibility to green space, potentially guiding real estate and urban planning decisions to enhance visitor experiences in Boston.

# # 4. Final Project 

# ## 4.1 OLS Regression

# This investigation seeks to dissect the multifaceted nature of guest satisfaction on Airbnb by exploring the correlations between overall ratings and specific sub-ratings—check-in, communication, location, and value. These components, pivotal to the guest experience, are assumed to directly influence the overarching perception of a stay. From an economic standpoint, this approach echoes the theory of consumer choice, where guests' overall satisfaction is a composite good, influenced by these underlying "goods" or aspects of their stay. 
# 
# By employing regression analysis, we aim to quantify the marginal utility— the additional satisfaction derived from an incremental improvement in each aspect of the stay—provided by each sub-rating to the overall rating. This is akin to understanding how each specific service improvement can incrementally increase the perceived value of the stay, mirroring the concept of marginal rates of substitution in consumer theory. 
# 
# Furthermore, the analysis extends to examining how host-related factors such as responsiveness and verified identity impact communication ratings. Economic reasoning suggests that these factors can be seen as signals of quality in a market characterized by information asymmetry between hosts and guests. According to signaling theory, hosts who are superhosts or who respond quickly are signaling higher service quality, which can reduce information gaps and lead to higher satisfaction ratings.
# 
# In analyzing the impact of location, including proximity to amenities and the socioeconomic status of the neighborhood as indicated by median household income, this study applies the concept of hedonic pricing. Here, the location rating can be considered a function of various location attributes, with guests willing to pay a premium (or express higher satisfaction) for properties with desirable location features.
# 
# Lastly, the investigation into value for money through price per guest and property type analysis utilizes the economic principle of utility maximization under budget constraints. It explores how guests' perceived value (utility) of their stay is affected by the price, within the context of their budget constraints, and how different property types can maximize utility by offering varying combinations of attributes at different price points.
# 
# Through this economic lens, the analysis aims not only to illuminate the direct contributors to guest satisfaction but also to offer Airbnb hosts insights into optimizing their offerings. By understanding the economic underpinnings of guest preferences and satisfaction, hosts can make informed decisions to enhance the attractiveness of their listings, thereby improving their competitive edge in the market.

# ### 4.1.1 Preparing the Final Dataset for Regression Analyses

# In[36]:


# response_cols: review_scores_rating, review_scores_communication, review_scores_checkin, review_scores_location
# ,review_scores_value 

reg_df = listings[['review_scores_rating', 'review_scores_location', 'review_scores_communication', 'review_scores_checkin',
          'review_scores_value', 'amenities_count', 'price_per_guest', 'is_private_room', 'is_shared_room', 'med_household_income', 
          'Community Gardens', 'Malls, Squares & Plazas', 'Open Land', 'Parks, Playgrounds & Athletic Fields','Urban Wilds', 
          'is_downtown', 'host_since_months', 'host_response_rate', 'host_is_superhost', 'host_total_listings_count', 
          'host_has_profile_pic', 'host_identity_verified', '_within a day', '_within a few hours', 
                       '_within an hour', '_Bed & Breakfast', '_Boat', '_Condominium',
                       '_Dorm', '_Entire Floor', '_Guesthouse', '_House', '_Loft', 
                       '_Other', '_Townhouse', '_Villa']]
reg_df['const'] = 1

# Subratings
rating_cols = ['const', 'review_scores_location', 'review_scores_communication', 'review_scores_checkin','review_scores_value']
    
# Topic 1: Booking Process and Host Interaction - review_scores_communication & review_scores_checkin 
comm_checkin_cols = ['const', 'host_response_rate', 'host_is_superhost', 'host_total_listings_count', 'host_has_profile_pic', 
                    'host_identity_verified', 'host_since_months', '_within a day', '_within a few hours', '_within an hour']
host_cols = ['const', 'host_is_superhost', 'host_total_listings_count', 'host_has_profile_pic', 
                    'host_identity_verified', 'host_since_months']
response_cols = ['const', '_within a day', '_within a few hours', '_within an hour']

# Topic 2: Accommodation Quality & Location
for col in ['Community Gardens', 'Malls, Squares & Plazas', 'Open Land', 
            'Parks, Playgrounds & Athletic Fields', 'Urban Wilds']:
    reg_df[f'is_downtown*{col}'] = reg_df['is_downtown'] * reg_df[col]
    
quality_location_cols = ['const', 'Community Gardens', 'Malls, Squares & Plazas',
                         'Open Land', 'Parks, Playgrounds & Athletic Fields', 'Urban Wilds']
quality_location_cols_interaction = ['const', 'is_downtown', 'Community Gardens', 'Malls, Squares & Plazas', 'Open Land', 
                         'Parks, Playgrounds & Athletic Fields', 'Urban Wilds',
                         'is_downtown*Community Gardens', 'is_downtown*Malls, Squares & Plazas', 
                         'is_downtown*Open Land', 'is_downtown*Parks, Playgrounds & Athletic Fields', 
                         'is_downtown*Urban Wilds']

dt_cols = ['const', 'med_household_income_K']


# Topic 3: Amenities & Comfort
amenities_cols = ['const', 'amenities_count', 'is_private_room', 'is_shared_room']

# Topic 4: Value
value_cols = ['const', 'price_per_guest', 'is_downtown', '_Bed & Breakfast', 
              '_Boat', '_Condominium','_Dorm', '_Entire Floor', '_Guesthouse', 
              '_House', '_Loft', '_Other', '_Townhouse', '_Villa']
value_cols_downtown_price = ['const', 'price_per_guest', 'is_downtown']


# ### 4.1.2 Investigating the Correlations between Overall Rating and Sub-ratings on checkin_in, communication, location, and value

# In this section, we look at how overall guest satisfaction on Airbnb is influenced by their ratings on specific aspects of their stay: check-in, communication, location, and value. We're exploring whether there's a clear link between how guests rate these individual parts of their experience and their overall rating. Our starting point is the belief that if a guest is happy with certain aspects of their stay, they're likely to rate their overall experience more positively.
# 
# To test this idea, we use regression analysis with the overall rating as our main focus and the four sub-ratings as predictors. This method helps us understand the impact of each aspect on overall satisfaction and see which ones guests seem to care about the most based on the strength of these relationships in the data.
# 
# The results of this analysis will help us identify what matters most to guests. If we know which aspects of the stay have the biggest impact on overall satisfaction, hosts can focus on improving these areas. This part of our study aims to offer a clear direction on where hosts can make changes that are most likely to improve guest satisfaction, based on how strongly each sub-rating is connected to the overall rating. Through this approach, we'll get a clearer picture of guest priorities and what makes for a satisfying Airbnb experience.

# In[37]:


# Relationship between overall rating and subratings
reg1 = sm.OLS(reg_df['review_scores_rating'], reg_df[rating_cols], missing='drop').fit()

stargazer = Stargazer([reg1])
stargazer.title("Table 8: Regression Results: Overall Rating and Subratings")
display(HTML(stargazer.render_html()))


# From the regression results, we can extract several key information about how 'important' the four aspects contribute to the overall staying experience for Airbnb guests: 
# 
# 1. All four aspects contribute positively to the overall rating. This is expected because, for instance, if a guest is happy about the 'location' or 'checkin' and give a high rating on these aspects, they are also likely to give a higher overall score. 
# 2. The coefficient that corresponds to the 'value' of their stay is 5.0543, which is the highest among the four aspects. This suggests that for every unit increase in the rating on 'value', the expected overall rating that the guest is expected to give increases by 5.0576. 
# 3. The coefficient corresponding to the 'location' of the listing is 0.8949, which is the lowest among the four aspect. This might suggest that the value that guests place on the 'location' of the listing might be the lowest relative to the other three aspects. 
# 4. The R-squared and adjusted R-squared are 0.664, which suggests that the four subratings have a moderately strong correlation with the overall rating. 
# 
# In the next section, I will be digging further into each aspect to investigate the main determinants of each aspect. 

# ### 4.1.3 Regression Analysis on Key Aspects 

# #### 4.1.3.1 Topic 1: Booking Processs & Host Interaction

# In[38]:


# Topic 1: Booking Process and Host Interaction - review_scores_communication & review_scores_checkin 
from statsmodels.iolib.summary2 import summary_col
info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}", 
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

comm_checkin_cols = ['const', 'host_response_rate', 'host_is_superhost', 'host_total_listings_count', 'host_has_profile_pic', 
                    'host_identity_verified', 'host_since_months', '_within a day', '_within a few hours', '_within an hour']
host_cols = ['const', 'host_is_superhost', 'host_total_listings_count', 'host_has_profile_pic', 
                    'host_identity_verified', 'host_since_months']
response_cols = ['const', '_within a day', '_within a few hours', '_within an hour']

reg2 = sm.OLS(reg_df['review_scores_communication'], reg_df[host_cols], missing='drop').fit()
reg3 = sm.OLS(reg_df['review_scores_communication'], reg_df[response_cols], missing='drop').fit()

results_table = summary_col(results=[reg2, reg3],
                            float_format='%0.2f',
                            stars = True,
                            model_names=['Model 2', 'Model 3'],
                            info_dict=info_dict,
                            regressor_order=['const',
                                             'host_has_profile_pic',
                                             'host_identity_verified',
                                             'host_is_superhost',
                                             'host_since_months',
                                             'host_total_listings_count',
                                             '_within a day',
                                             '_within a few hours',
                                             '_within an hour'])

stargazer = Stargazer([reg2, reg3])
stargazer.title("Table 9: Regression Results: Factors Affecting Communication Rating")
display(HTML(stargazer.render_html()))


# In this section, I investigate how the hosts' experience and actions, and response speed affect guests' satisfaction on the 'communication' aspect of their experience. I used whether the host is a superhost, their years of being a host, and the number of listings they manage as proxies for their experience. In terms of their actions, I investigate whether having a profile picture and having their identity verified also contribute positively to the communication aspect. Lastly, I also investigate how the speed of them responding to guest inquiries affect this aspect. 
# 
# To improve the interpretability of my results, I create two separate models to understand how hosts' experience/actions (model 2) and their response speed affect communication score (model 3). There are several insights that I could extract from the two models: 
# 
# 1. All the variables included in the model, apart from the host having a profile picture, are significant in explaining the variations in communication rating. For hosts, having a profile picture does not significantly improve guests' rating on this aspect. 
# 2. Being a superhost and having their identity verified both contribute positively to the communication aspect. Since there is certain criteria of becoming a superhost, including the number of times they have hosted, this may suggest that over time Airbnb hosts learn that having a positive communication experience is important in determining guest satisfaction, and hence they pay more attention to this aspect. 
# 3. For response speed, it seems that hosts that respond to guest inquiries within an hour has a 0.12 higher score on communication rating compared to those that response after several days. An insignificant p-value for '_within a day' seems to suggest that if the host does not respond after a few hours after a guest posts a question, then it doesn't matter for the host to respond within a day or after several days. 
# 
# Finally the two models each have an R-squared of 0.09 and 0.03, which suggest that the host's response and their experience explain a small proportion of the changes in the communication ratings. These values are pretty low and imply that there are other aspects of the communication process that guests take into account. For instance, guests may place significant value in the instant booking feature, which provides immediate confirmation for their booking and reducesthe need for back-and-forth messages. 

# #### 4.1.3.2 Topic 2: Accomodation Quality & Location

# #### Investigating how different public spaces affect location rating

# In[39]:


# generate interaction terms
for col in ['Community Gardens', 'Malls, Squares & Plazas', 'Open Land', 
            'Parks, Playgrounds & Athletic Fields', 'Urban Wilds']:
    reg_df[f'is_downtown*{col}'] = reg_df['is_downtown'] * reg_df[col]

quality_location_cols = ['const', 'Community Gardens', 'Malls, Squares & Plazas',
                         'Open Land', 'Parks, Playgrounds & Athletic Fields', 'Urban Wilds']
quality_location_cols_interaction = ['const', 'is_downtown', 'Community Gardens', 'Malls, Squares & Plazas', 'Open Land', 
                         'Parks, Playgrounds & Athletic Fields', 'Urban Wilds',
                         'is_downtown*Community Gardens', 'is_downtown*Malls, Squares & Plazas', 
                         'is_downtown*Open Land', 'is_downtown*Parks, Playgrounds & Athletic Fields', 
                         'is_downtown*Urban Wilds']

reg4 = sm.OLS(reg_df['review_scores_location'], reg_df[quality_location_cols], missing='drop').fit()
reg5 = sm.OLS(reg_df['review_scores_location'], reg_df[quality_location_cols_interaction], missing='drop').fit()

stargazer = Stargazer([reg4, reg5])
stargazer.title("Table 10: Regression Results: How Public Spaces affect Location Rating")
stargazer.add_custom_notes([f'Model 1 AIC: {round(reg4.aic,2)}, BIC: {round(reg4.bic,2)}',
                            f'Model 2 AIC: {round(reg5.aic,2)}, BIC: {round(reg5.bic,)}'])
display(HTML(stargazer.render_html()))


# From the study, we learn how different nearby places can make an Airbnb more or less attractive to guests. Listings close to community gardens score a bit lower, which might surprise some. On the flip side, being near shopping areas like malls and plazas gives a boost to how guests rate the location. Interestingly, open spaces also contribute positively to location score, showing people do enjoy some nature around. However, being near parks or wild areas doesn’t seem to be as appealing, possibly because guests might worry about noise or safety. When we look at places downtown, the story changes a bit. The usual perks of being near shopping don't stand out as much, maybe because downtown areas already have lots of shops and activities.
# 
# Looking at R-squared, we see that the open spaces that we considered in model 4 explain around 20% of the variations in location score. When we added more details about being downtown, this understanding improved slightly to 21%. It's a small step up but shows that adding downtown details helps us get a slightly clearer picture.
# 
# Given the AIC and BIC scores for both models, model 5 shows a lower AIC score than model 4, suggesting it's a better fit for the data considering its complexity. the AIC reductionfrom 8148.28to 8132.57 indicates that model 5, with the added dummy that indicates whether the listing is located in downtown and the interactions terms of this dummy with other location features, improves our understanding of what affects Airbnb location ratings without overfitting. However, when looking at BIC, model 5 has a slightly higher BIC than model 4. This suggests that the additional complexity of model 5 may not be justifed to the same ectent when the stricter penalty of BIC is applied. Despite this, the lower AIC of model 5 still points towardsit being the preferable model, especially my goal here is to capture the nuanced effects of being downtown alongside other locational features on Airbnb location scores. 

# #### 4.1.3.3 Investigating how neighborhood income affect location rating for downtown listings

# In[40]:


# Investigating how living in different neighborhoods with different median household income affect the location rating
downtown_listings_df = reg_df[reg_df['is_downtown'] == 1]
downtown_listings_df[['review_scores_location','med_household_income']]

info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}", 
           'No. observations' : lambda x: f"{int(x.nobs):d}"}
reg_df['med_household_income_K'] = reg_df['med_household_income']/1000

dt_cols = ['const', 'med_household_income_K']

reg5 = sm.OLS(reg_df['review_scores_location'], reg_df[dt_cols], missing='drop').fit()

stargazer = Stargazer([reg5])
stargazer.title("Table 11: Regression Results: How Neighborhood Wealth Affects Location Rating")
display(HTML(stargazer.render_html()))


# In this section, I used median_household_income of the neighborhood as a proxy to represent the 'quality' of the neighborhoods as their wealth tend to correlate with the living standards in the area. Since median household income varies significantly in downtown neighborhoods and not in neighborhoods outside of downtown, I focused specifically on listings in the Boston downtown area specifically. I first subset the dataset so that it only includes listings in downtown. Then I create a new regression model with median household income (in thousands of US dollars) as the only variable. The regression result shows that if the median_household_income of the neighborhood increases by one thousand USD, the rating of the location is expected to increase by 0.08 (out of a scale of 10). This is a small change, which may imply that either median household income does not provide an accurate measure for the 'quality' of the neighborhood, or that there are other aspects that contribute more significantly to the guests' satisfaciton about the location.

# #### 4.1.3.4 Topic 3: Amenities & Comfort

# In[41]:


# the base group is entire house 
amenities_cols = ['const', 'amenities_count', 'is_private_room', 'is_shared_room']

reg6 = sm.OLS(reg_df['review_scores_rating'], reg_df[amenities_cols], missing='drop').fit()

stargazer = Stargazer([reg6])
stargazer.title("Table 12: Regression Results: How Amenities Count and Room Type Affect Overall Rating")
display(HTML(stargazer.render_html()))


# In this section, I investigate how the number of amenities that listing provides and the rooom type affects the overall rating. Amenities here refer to item such as TV, internet, free-parking and other conveniences featires offered to guest. Since the dataset does not contain any ratings that indicate the guests' 'comfort' level or satisfaction on amenities, I will just be investigating how the variables contribute to the their overall satisfaction. 
# 
# From the regression results, we observe that amenities_count has an coefficient of 0.44, and this value is significant. This suggests that adding an additional item/feature increases the overall rating by 0.44 (the overall rating score has a scale of 100), which is relatively small. 
# 
# Furthermore, the insignificance of 'is_private_room' and 'is_shared_room' suggests that guests usually do not mind sharing an apartment/house with the host or other guests. 

# #### 4.1.3.5 Topic 4: Value for Money

# In this section, I investigate variables that may potentially affect the perceived value that guests derived from their stay in return for the price they paid using two separate models. Model 7 investigates whether higher price leads to lower percention of value received and whether listings located within downtown lead to higher perceived value. Model 8 investigates which property type has the higher perceived value. 

# #### Effects of price per guest and being in downtown on the perceived value of the experience

# In[42]:


# Topic 4: Value
value_cols = ['const', 'price_per_guest', 'is_downtown', '_Bed & Breakfast', 
              '_Boat', '_Condominium','_Dorm', '_Entire Floor', '_Guesthouse', 
              '_House', '_Loft', '_Other', '_Townhouse', '_Villa']
value_cols_downtown_price = ['const', 'price_per_guest', 'is_downtown']

# Checking multicollinearity between is_downtown and price_per_guest
X = reg_df[['price_per_guest', 'is_downtown']]  


# Calculate VIF for each explanatory variable
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)

reg7 = sm.OLS(reg_df['review_scores_value'], reg_df[value_cols_downtown_price], missing='drop').fit()

stargazer = Stargazer([reg7])
stargazer.title("Table 13: Value Perception Disparities: Downtown vs. Non-downtown Listings")
display(HTML(stargazer.render_html()))


# Multicollinearity between price_per_guest and is_downtown is checked using Variance Inflation Factor (VIF) before building the regression since I expect downtown listings to be more expensive (per guest) compared to non-downtown listings. VIF is a measure used to detect the presence and severity of multicollinearity in a regression model, quantifying how much the variance of an estimated regression coefficient increases due to collinearity. A VIF value of 1 indicates no correlation between a given independent variable and any other independent variables in the model, while a VIF exceeding 5 or 10 suggests significant multicollinearity that may distort regression coefficients and weaken the statistical power of the model. By calculating VIF for each predictor, I can identify and address multicollinearity, improving model reliability and interpretation.
# 
# Both variables have VIF values less than 5, which suggest multicollinearity is not an issue here and we may safely interpret the regression coefficients.
# 
# Looking at the regression coefficients, the significance of 'price_per_guest' indicates that higher price indeed leads to lower perceived value. However, this effect is extremely small (less than 0.01) on the rating of value, which has a scale of 10. Second, listings that do not locate in downtown on average do not have a higher perceived value than those in downtown. 

# #### Which property type has the higher perceived value?

# In[43]:


# Base group is Apartment
value_cols_property = ['const', '_Bed & Breakfast', '_Boat', '_Condominium','_Dorm', '_Entire Floor', '_Guesthouse', 
              '_House', '_Loft', '_Other', '_Townhouse', '_Villa']

reg8 = sm.OLS(reg_df['review_scores_value'], reg_df[value_cols_property], missing='drop').fit()

stargazer = Stargazer([reg8])
stargazer.title('Table 14: Impact of Property Type on Guest Perception of Value')
display(HTML(stargazer.render_html()))


# Here, we are comparing the perceived value of different property types to that of apartments, which is the base group. We observe that listings that are condominium, house, and loft have a significantly higher perceived value than apartments. Listings that are condominium, house, and loft, on average receive 0.34, 0.18, and 0.42 higher rating on their perceived values than apartments, respectively.

# This analysis reveals the dynamics between Airbnb guests' overall satisfaction and their ratings on check-in, communication, location, and value, grounded in economic reasoning. 
# 
# Firstly, the positive impact of all four aspects on overall ratings confirms that improvements in specific service areas can enhance overall guest satisfaction. The most significant impact comes from value, suggesting that guests place a high emphasis on receiving good value for their money, aligning with the economic principle of consumer surplus.
# 
# The importance of host qualities like superhost status and verified identity in improving communication ratings highlights the role of trust and reliability, akin to signaling theory where hosts signal quality to reduce information asymmetry.
# 
# Response speed's positive effect on communication ratings emphasizes the value of time in service quality, indicating that quicker responses are seen as better service, potentially leading to higher satisfaction.
# 
# The study on location ratings through amenities and neighborhood socioeconomic status suggests a hedonic pricing model, where guests value certain location features differently, impacting their overall location satisfaction.
# 
# Furthermore, the analysis shows a nuanced relationship between price per guest and perceived value, illustrating the economic concept of diminishing returns where higher prices might not always lead to higher perceived value. The varied perceived value across property types also points to diverse guest preferences, reflecting different utility derived from different accommodation types.
# 
# In summary, the findings indicate that guest satisfaction on Airbnb is influenced by a combination of factors including value for money, trust and reliability of hosts, timely communication, and specific location features. These insights underscore the importance for hosts to focus on delivering value and maintaining high service standards to enhance guest satisfaction, informed by principles like consumer surplus, signaling theory, and utility maximization.

# ## 4.2 Machine Learning

# Section 4.1 OLS Regression focuses on how specific variables within those categories affect guest satisfaction on Airbnb. This section shifts attention to using machine learning techniques, including multivariate regression, regression trees, bagging, and random forests to understand how these variables influence the overall rating score of listings. It transitions from analyzing subratings to examining the overall rating score, aiming to identify the direct effects of amenities count, pricing, room type, neighborhood economic status, proximity to urban features, host characteristics, response speed, and property type on guest satisfaction.
# 
# In modeling the overall rating score of an Airbnb listing, incorporating variables like 'amenities_count', 'price_per_guest', 'is_private_room', and 'is_shared_room' captures the essential economic trade-offs guests make regarding comfort, privacy, and cost, directly impacting their satisfaction and perceived value. Variables indicating the listing's location, such as 'med_household_income' and 'is_downtown', alongside community features like 'Parks, Playgrounds & Athletic Fields', reflect the neighborhood's desirability and accessibility, influencing guest preferences and ratings.
# 
# Host-related attributes, including 'host_since_months', 'host_is_superhost', and 'host_response_rate', represent the host's experience, reliability, and commitment to service, which are crucial for guest trust and satisfaction. Additionally, response time categories and property types offer insights into the service quality and unique experiences provided, affecting guest perceptions and the overall rating.
# 
# Economically, these variables are significant as they align with the principles of supply and demand, utility maximization, and market segmentation, affecting guests' decision-making processes and the listings' competitive positioning. By including a broad set of variables, the model comprehensively accounts for the multifaceted factors that determine guest satisfaction and the economic dynamics of Airbnb listings.

# In[44]:


from sklearn import linear_model
reg_df = listings[['review_scores_rating', 'review_scores_location', 'review_scores_communication', 'review_scores_checkin',
          'review_scores_value', 'amenities_count', 'price_per_guest', 'is_private_room', 'is_shared_room', 'med_household_income', 
          'Community Gardens', 'Malls, Squares & Plazas', 'Open Land', 'Parks, Playgrounds & Athletic Fields','Urban Wilds', 
          'is_downtown', 'host_since_months', 'host_response_rate', 'host_is_superhost', 'host_total_listings_count', 
          'host_has_profile_pic', 'host_identity_verified', '_within a day', '_within a few hours', 
                       '_within an hour', '_Bed & Breakfast', '_Boat', '_Condominium',
                       '_Dorm', '_Entire Floor', '_Guesthouse', '_House', '_Loft', 
                       '_Other', '_Townhouse', '_Villa']]
reg_df['const'] = 1

# List of columns to impute
columns_to_impute = [
    'review_scores_rating', 
    'review_scores_location', 
    'review_scores_communication', 
    'review_scores_checkin', 
    'review_scores_value', 
    'med_household_income', 
    'host_response_rate'
]

# Performing mean imputation
for column in columns_to_impute:
    reg_df[column].fillna(reg_df[column].mean(), inplace=True)


# ### 4.2.1 Multivariate Linear Regression (with Regularization)

# By employing a multiple regression model, this analysis seeks to reveal patterns and relationships between these variables and the overall satisfaction rating. This approach builds on the previous insights, offering a broader perspective on what influences the guest experience on Airbnb. The results are intended to inform hosts about which factors most significantly impact their overall rating scores, guiding them in optimizing their listings to enhance guest satisfaction.

# In this section, I will include all variables that were discussed in the OLS section to understand their direct effects on the overall rating. Hence, the linear model is:
# 
# $$
# \text{overall_rating} = \beta_0 + \beta_1 \text{amenities_count} + \beta_2 \text{price_per_guest} + \\
# \beta_3 \text{is_private_room} + \beta_4 \text{is_shared_room} + \\
# \ldots + \beta_{30} \text{_Townhouse} + \beta_{31} \text{_Villa} + \epsilon
# $$
# 
# In prediction tasks with datasets containing many predictors, Ridge regression is a useful approach. It adds a regularization term to the standard least squares objective function, preventing overfitting by penalizing the magnitudes of coefficients. This regularization, controlled by the hyperparameter alpha, stabilizes coefficient estimates and handles multicollinearity, common in datasets with numerous features. Ridge regression strikes a balance between bias and variance, making it suitable for mitigating overfitting while maintaining the model's predictive accuracy.
# 
# $$
# \begin{align*}
# \text{Objective Function: } & ||Y - X\beta||^2_2 + \alpha||\beta||^2_2 \\
# \end{align*}
# $$
# $$
# \begin{align*}
# \text{where:} \\
# & ||Y - X\beta||^2_2 \text{ is the Residual Sum of Squares (RSS),} \\
# & \text{ } Y \text{ represents the observed overall ratings,} \\
# & \text{ } X \text{ is the matrix of predictor variables,} \\
# & \text{ } \beta \text{ is the vector of coefficients,} \\
# & \text{ } \alpha \text{ is the regularization strength,} \\
# & ||\beta||^2_2 \text{ denotes the squared L2 norm of } \beta \text{.}
# \end{align*}
# $$
# 

# In[45]:


# reg_df[['review_scores_rating', 'review_scores_location', 
#                 'review_scores_communication', 'review_scores_checkin',
#           'review_scores_value', 'amenities_count', 'price_per_guest', 
#                 'is_private_room', 'is_shared_room', 'med_household_income', 
#           'Community Gardens', 'Malls, Squares & Plazas', 'Open Land', 
#                 'Parks, Playgrounds & Athletic Fields','Urban Wilds', 
#           'is_downtown', 'host_since_months', 'host_response_rate', 
#                 'host_is_superhost', 'host_total_listings_count', 
#           'host_has_profile_pic', 'host_identity_verified', '_within a day', 
#                 '_within a few hours', '_within an hour', '_Bed & Breakfast', 
#                 '_Boat', '_Condominium', '_Dorm', '_Entire Floor', '_Guesthouse', '_House', '_Loft', 
#                        '_Other', '_Townhouse', '_Villa']]

X = reg_df[['amenities_count', 'price_per_guest', 'is_private_room', 'is_shared_room',
            'med_household_income', 'Community Gardens', 'Malls, Squares & Plazas', 'Open Land',
            'Parks, Playgrounds & Athletic Fields', 'Urban Wilds', 'is_downtown', 
            'host_since_months', 'host_response_rate', 'host_is_superhost', 
            'host_total_listings_count', 'host_has_profile_pic', 'host_identity_verified', 
            '_within a day', '_within a few hours', '_within an hour', '_Bed & Breakfast', 
            '_Boat', '_Condominium', '_Dorm', '_Entire Floor', '_Guesthouse', '_House', 
            '_Loft', '_Other', '_Townhouse', '_Villa']]

y = reg_df['review_scores_rating']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def plot_actual_vs_predicted(y_actual, y_predicted, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_predicted, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Figure 10: Actual vs. Predicted Values for MLR')
    plt.show()


# In[46]:


# Fit models on the training data
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

plot_actual_vs_predicted(y_test, y_pred_lr, 'MLR')
plt.show()


# This scatterplot of fitted vs. actual values evaluates the predictive performance of our multivariate regression model concerning the overall rating. Ideally, the data points would tightly cluster along the diagonal line, signaling precise predictions. However, the model demonstrates predictive inaccuracies at the upper end of the value spectrum, hinting at an overlooked non-linear relationship within the data. Moreover, the error variance, which fluctuates across different levels of predicted values, contradicts the linear regression model's assumption of constant error variance, or homoscedasticity. These complications imply that the model may require a reevaluation, possibly calling for a transformation of the target variable to better accommodate the data's characteristics. The observed dispersion of data points in the plot, coupled with a notably high Mean Squared Error (MSE) of 52.91 on a 100-point scale, clearly indicates that the multivariate linear model is not ideally suited for our data. 

# ### 4.2.2 Decision Tree

# As we've observed from the scatterplot, our current linear regression model exhibits signs of both non-linearity and heteroscedasticity, indicating that a more flexible approach may be warranted. Transitioning to the section on decision tree, we embrace a non-parametric methodology that can adeptly capture the complex, non-linear relationships and interactions between variables without the restrictive assumptions inherent in linear models. 
# 
# A Decision Tree partitions the data into subsets based on decision rules inferred from the predictor variables, which allows for a piecewise approximation of the underlying relationship. This approach not only mitigates the influence of outliers and the problem of non-constant variance but also provides a clear, interpretable structure for our model.
# 
# The objective function for the regression tree can be represented as: 
# 
# 
# $$
# \begin{align*}
# \text{Objective Function: } & \sum_{i=1}^{N} (y_i - f(x_{i1}, x_{i2}, \dots, x_{ip}))^2 
# \end{align*}
# $$
# 
# $$
# \begin{align*}
# \text{where:} \\
# & \text{ } y_i \text{ is the actual overall rating score of the i-th listing,} \\
# & \text{ } f(x_{i1}, x_{i2}, \dots, x_{ip}) \text{ is the prediction function of the regression tree for the i-th listing,} \\
# & \text{ } x_{i1}, x_{i2}, \dots, x_{ip} \text{ are the input variables for the i-th listing, including 'amenities_count', 'price_per_guest', etc.,} \\
# & \text{ } N \text{ is the total number of listings in the dataset.}
# \end{align*}
# $$
# 
# 
# 
# In this objective function, the goal is to minimize the sum of the squared differences between the actual and predicted rating scores across all listings. This minimization process effectively determines how the listings are split at each node of the tree, aiming to group listings such that within each group (or leaf), the variance of the rating scores is minimized. This results in a tree where each leaf node predicts the average rating score of the listings that fall into that leaf, with the splitting decisions made at each node designed to reduce the overall prediction error as measured by $J$.
# 
# The regression tree algorithm will iteratively select the best splits based on this objective function, choosing the variables and split points that result in the most significant decrease in $J$. This process continues until a stopping criterion is met, such as the predefined maximum depth of 3 of the tree that we defined.

# In[47]:


sqft_tree = DecisionTreeRegressor(max_depth=3)
sqft_tree.fit(X_train, y_train)
y_pred_tree = sqft_tree.predict(X_test)

# Scatterplot for decision tree
plot_actual_vs_predicted(y_test, y_pred_tree, 'Decision Tree')


# In[48]:


# Tree plot
tree_fig = plt.figure(figsize=(25,20))
tree_fig = tree.plot_tree(sqft_tree, feature_names=X.columns, filled=True)
plt.title("Figure 10: Decision Tree Visualization for Predicting Overall Rating", fontsize=24) 
plt.show()


# The regression tree results are helpful in providing a visual and straightforward way of understanding the factors that most influence overall rating and provide predictions of what the overall rating is expected to be under different scenarios.
# 
# A primary takeaway is the pivotal role of a host's superhost status; guests tend to rate their stay more positively when the host is a superhost. This is likely due to the trust and high-quality experience associated with Airbnb's superhost program. Host responsiveness also emerges as a critical factor. Listings where the host responds swiftly, particularly faster than 83% of the time, tend to have higher satisfaction scores, which underscores the importance of communication in the hospitality experience. However, it's not just about being fast—being a superhost suggests a level of care and service that guests clearly value. Additionally, the amenities count significantly sways guest satisfaction, with listings offering more than 8.5 amenities scoring lower, indicating that beyond a certain point, the number of amenities does not linearly increase satisfaction, possibly due to diminishing marginal utility.
# 
# The second layer of insight revolves around the economics of pricing and neighborhood effects. The regression tree suggests that guests who opt for more expensive listings—above $525 per guest—are among the most satisfied, indicating that certain guests are willing to pay a premium for an exceptional experience. This could be reflective of luxury market dynamics where higher prices align with elevated guest expectations. Furthermore, the model indicates a marginal preference for stays in neighborhoods with higher median incomes, hinting that guests may have a slight inclination towards areas perceived as more upscale or safer. Intriguingly, while one might expect cost to be a dampener on satisfaction, the data reveals that at the upper echelons of the market, the reverse is true—higher cost correlates with greater satisfaction, a testament to the power of perceived value over raw price figures. This insight is particularly valuable for hosts targeting the premium segment, emphasizing the need to align price with the luxury experience guests anticipate.

# ### 4.2.3 Bagging 

# Moving from the individual regression tree analysis, we now proceed to apply bagging methods. The transition aims to validate and potentially enhance the predictiveness of our model. Bagging will allow us to aggregate multiple models to address the overfitting seen in a single decision tree and provide a more generalized prediction. It's a method that can help in better understanding the data by creating numerous variations of the model and combining them to improve accuracy. This next step will help us ascertain the reliability of the factors identified by the regression tree in influencing Airbnb guest satisfaction. Through this approach, we hope to solidify the predictive strength of our model.

# In[49]:


regr_bagg = RandomForestRegressor(max_features=31, random_state=1)
regr_bagg.fit(X_train, y_train)
y_pred_bagg = regr_bagg.predict(X_test)

# Scatterplot for Bagging
plot_actual_vs_predicted(y_test, y_pred_bagg, 'Bagging (Random Forest with all features)')


# In this scatterplot of actual observed values and predicted values from our bagging model, the concentration of points along the diagonal line indicates the model's predictions are much more accurate than the multivariate regression model.

# ### 4.2.4 Random Forest

# Moving on from our initial findings with the regression tree, I will now conduct Random Forest to enhance my predictions. This approach combines many trees to improve the accuracy of my model. With Random Forest, I aim to confirm previous regression results and potentially discover new insights. It's a powerful tool that provides a multi-faceted view of the data, which can uncover complex interactions and dependencies that a single tree may overlook.

# In[50]:


#Random Forest: using 5 features
regr_RF = RandomForestRegressor(max_features=17, random_state=1)
regr_RF.fit(X_train, y_train)
y_pred_RF = regr_RF.predict(X_test)

plot_actual_vs_predicted(y_test, y_pred_RF, 'Random Forest (with 17 features)')


# In[51]:


# # Splitting the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Range of `max_features` to explore: from 1 up to the number of features in X
# max_features_range = range(1, X_train.shape[1] + 1)

# # Store the average scores for each value of `max_features`
# scores = []

# for max_features in max_features_range:
#     model = RandomForestRegressor(max_features=max_features, random_state=1)
#     score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
#     scores.append(np.mean(score))

# # Find the optimal `max_features` with the highest score (lowest MSE because scores are negative)
# optimal_max_features = max_features_range[np.argmax(scores)]

# print(f"Optimal number of features: {optimal_max_features}")


# The code determines the optimal `max_features` for a Random Forest model by iterating over a range from 1 to the total number of features, evaluating model performance through 5-fold cross-validation with Mean Squared Error (MSE) as the metric. Scores are negated to fit scikit-learn's maximization framework. The process identifies the `max_features` yielding the highest score, indicating the lowest MSE. An outcome of 17 as the optimal value means using 17 features per split minimizes prediction error and improves model accuracy.

# In[52]:


Importance = pd.DataFrame({'Importance':regr_RF.feature_importances_*100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
plt.title('Figure 11: Random Forests Importance Matrix')
plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# The importance matrix from the Random Forest model indicates that 'price per guest', 'number of amenities provided', and 'duration of the host's experience' are the three most significant variables for model prediction accuracy. This contrasts with the single regression tree's findings, where 'whether the host is a superhost' ranks as the most influential variable for predicting the overall rating, with 'number of amenities' and 'host response rate' following in importance. The Random Forest model also ranks these latter two variables highly, within the top five for variable importance. The divergence between the models' results underscores the differences in how they assess feature influence on the target variable. The ensemble method of Random Forest, which aggregates outcomes from multiple decision trees, provides a broader analysis of data and feature interactions, potentially offering a more reliable estimation of feature importance.
# 
# The differences and similarities in the results from the regression tree and the Random Forest importance matrix can be traced back to several factors. The structure of the models is a primary consideration; Random Forest's method of combining insights from many trees offers a layered understanding of how features interact to affect predictions. This approach diminishes the impact of anomalies, presenting a comprehensive view of feature significance. Conversely, a regression tree provides a straightforward perspective on the influence of individual features, which may not fully capture complex interactions. Additionally, the variation in feature importance between the models highlights how certain features may have varying degrees of influence based on the modeling approach. For example, the 'superhost' status might significantly affect guest perceptions in a linear decision-making model. However, in the multifaceted decision-making environment of Random Forest, which considers a wider dataset, 'price per guest' and 'amenities provided' appear as consistently critical factors across various Airbnb listings 

# ## 4.3 Comparison of Models

# In[53]:


# Evaluate and compare models
metrics = {
    'Model': ['Linear Regression', 'Decision Tree', 'Bagging', 'Random Forest'],
    'MSE': [
        mean_squared_error(y_test, y_pred_lr),
        mean_squared_error(y_test, y_pred_tree),
        mean_squared_error(y_test, y_pred_bagg),
        mean_squared_error(y_test, y_pred_RF)
    ],
    'R2 Score': [
        r2_score(y_test, y_pred_lr),
        r2_score(y_test, y_pred_tree),
        r2_score(y_test, y_pred_bagg),
        r2_score(y_test, y_pred_RF)
    ]
}

metrics_df = pd.DataFrame(metrics)

# Convert the DataFrame to an HTML table with specific styling
html_table = metrics_df.to_html(classes='table table-striped table-hover', border=0)
# Display the HTML table in the Jupyter notebook
display(HTML(html_table))


# The comparison of machine learning models reveals a progressive improvement in performance from Linear Regression through to Random Forest, as evidenced by Mean Squared Error (MSE) and R2 Score metrics. Linear Regression, with the highest MSE and lowest R2 Score, suggests a limited ability to capture complex data relationships, likely due to its assumption of linearity between predictors and the target variable. The Decision Tree model, showing a marginal enhancement in both metrics, indicates a better grasp of non-linear relationships, though its vulnerability to overfitting may constrain its predictive accuracy. The introduction of ensemble methods, starting with Bagging, marks a significant leap in model performance. By leveraging multiple learners to reduce variance and improve prediction robustness, Bagging demonstrates a substantial decrease in MSE and increase in R2 Score, underscoring the efficacy of ensemble strategies in handling complex data patterns. The Random Forest model, an extension of Bagging with additional randomness in selecting features for splitting nodes, achieves the lowest MSE and highest R2 Score among the evaluated models. This highlights its superior capability in generalizing the learned patterns to new data, thereby providing the most accurate and reliable predictions.

# # 5. Conlcusion 

# The investigation into what drives guest satisfaction in Boston's Airbnb market has provided a detailed look at how specific factors—amenities, host communication, location, pricing, and property type—play pivotal roles. Through the application of economic theories and principles, alongside statistical and machine learning analyses, this study has revealed the intricate balance between supply and demand dynamics, consumer behavior, and market segmentation within the Airbnb ecosystem.
# 
# First, amenities were found to significantly influence guest satisfaction, highlighting the principle of diminishing marginal utility. Guests value a certain level of convenience and comfort provided by amenities up to a point, after which additional amenities do not contribute as significantly to satisfaction. This suggests that hosts should focus on providing essential, high-quality amenities that align with guest expectations rather than increasing the sheer quantity of amenities.
# 
# Host-guest communication emerged as another critical factor, aligning with the signaling theory. Hosts who are responsive and engage positively with guests signal trustworthiness and reliability, thereby reducing information asymmetry and increasing guest satisfaction. The superhost status acts as a credible signal of a high-quality hosting experience, further enhancing guest trust and satisfaction.
# 
# The study also underscored the importance of location, applying the hedonic pricing model to understand its impact on satisfaction. Guests are willing to pay a premium for properties located in desirable areas, indicating that location attributes are significantly valued. This preference for location reflects the economic concept of positional goods, where the value is derived from the property's relative position or desirability within a certain context or neighborhood.
# 
# Pricing strategy plays a complex role, demonstrating the relationship between perceived value and cost. The analysis suggests that pricing must be carefully considered to reflect the value provided. Economic principles of price elasticity and consumer surplus are at play, with guests evaluating the trade-off between price and the quality of their stay, indicating a nuanced relationship between pricing, expectations, and satisfaction.
# 
# Lastly, the influence of property type on guest satisfaction highlights market segmentation and consumer preferences within the Airbnb market. Different guest segments have distinct preferences and willingness to pay, which can be catered to by offering various types of properties. Understanding these segments allows hosts to tailor their offerings more effectively, maximizing utility for both hosts and guests.
# 
# In conclusion, this research has delineated the multifaceted nature of guest satisfaction in Boston's Airbnb market through an economic lens, offering actionable insights for hosts. By focusing on quality amenities, effective communication, strategic pricing, and understanding the economic underpinnings of guest preferences, hosts can enhance guest satisfaction and improve their competitive standing in the market. This study not only contributes to the academic understanding of the short-term rental market but also provides practical guidelines for hosts aiming to optimize their offerings in the ever-evolving Airbnb ecosystem. 
# 
# An area for further exploration is the price elasticity of demand within the Airbnb market. A focused study on how price fluctuations impact booking volume could yield critical insights into optimal pricing strategies for hosts. Investigating price sensitivity not only informs individual pricing decisions but also provides broader economic understanding of consumer behavior in the sharing economy. Such an analysis could reveal the thresholds at which guests are willing to trade off between cost and the perceived value of an Airbnb experience, guiding more nuanced and dynamic pricing models.
# 

# 
# # References
# [1] Airbnb. “Boston Airbnb Open Data.” Kaggle, November 17, 2019. https://www.kaggle.com/datasets/airbnb/boston?select=reviews.csv. 
# 
# [2] Craciun, Georgiana, Wenqi Zhou, and Zhe Shan. “Discrete Emotions Effects on Electronic Word-of-Mouth Helpfulness: The Moderating Role of Reviewer Gender and Contextual Emotional Tone.” Decision Support Systems 130 (March 2020): 113226. https://doi.org/10.1016/j.dss.2019.113226. 
# 
# [3] Dogru, Tarik, Lydia Hanks, Makarand Mody, Courtney Suess, and Ercan Sirakaya-Turk. “The Effects of Airbnb on Hotel Performance: Evidence from Cities beyond the United States.” Tourism Management 79 (August 2020): 104090. https://doi.org/10.1016/j.tourman.2020.104090. 
# 
# [4] Gibbs, Chris, Daniel Guttentag, Ulrike Gretzel, Jym Morton, and Alasdair Goodwill. “Pricing in the Sharing Economy: A Hedonic Pricing Model Applied to Airbnb Listings.” Journal of Travel &amp; Tourism Marketing 35, no. 1 (April 10, 2017): 46–56. https://doi.org/10.1080/10548408.2017.1308292. 
# 
# [5] Guttentag, Daniel, Stephen Smith, Luke Potwarka, and Mark Havitz. “Why Tourists Choose Airbnb: A Motivation-Based Segmentation Study.” Journal of Travel Research 57, no. 3 (April 27, 2017): 342–59. https://doi.org/10.1177/0047287517696980. 
# 
# [6] Law, Rob, Irene Cheng Chan, and Liang Wang. “A Comprehensive Review of Mobile Technology Use in Hospitality and Tourism.” Journal of Hospitality Marketing &amp; Management 27, no. 6 (January 25, 2018): 626–48. https://doi.org/10.1080/19368623.2018.1423251. 
# 
# [7] Byers, John, Davide Proserpio, and Georgios Zervas. “The Rise of the Sharing Economy: Estimating the Impact of Airbnb on the Hotel Industry.” SSRN Electronic Journal, 2013. https://doi.org/10.2139/ssrn.2366898. 
# 
