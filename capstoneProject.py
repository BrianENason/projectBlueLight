
import datetime

print(datetime.datetime.now(), ": Project Loading.")
print('')
print(datetime.datetime.now(), ': Loading Libraries...')
print('\t...This could take a few minutes')
print('')

print(datetime.datetime.now(), ': Loading Library [os]...')
import os
print(datetime.datetime.now(), ': [os] library loaded!')
print(datetime.datetime.now(), ': Loading Library [sys]...')
import sys
print(datetime.datetime.now(), ': [sys] library loaded!')
print(datetime.datetime.now(), ': Loading Library [piedot]...')
import pydot
print(datetime.datetime.now(), ': [piedot] library loaded!')
print(datetime.datetime.now(), ': Loading Library [pandas]...(this could take a couple minutes)')
import pandas as pd
print(datetime.datetime.now(), ': [pandas] library loaded!')
print(datetime.datetime.now(), ': Loading Library [numpy]...')
import numpy as np
print(datetime.datetime.now(), ': [numpy] library loaded!')
print(datetime.datetime.now(), ': Loading Library [PySimpleGui]...')
import PySimpleGUI as sg
print(datetime.datetime.now(), ': [PySimpleGui] library loaded!')
print(datetime.datetime.now(), ': Loading Libraries from [Matplotlib]...')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
print(datetime.datetime.now(), ': [Matplotlib] libraries loaded!')
print(datetime.datetime.now(), ': Loading Libraries from [sklearn]...')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
print(datetime.datetime.now(), ': [sklearn] libraries loaded!')


# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression
# from matplotlib.scale import LinearScale
# from numpy.polynomial import Chebyshev

# Create target Directory if don't exist
dirName = 'charts'
if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")

print('')
print(datetime.datetime.now(), ': Libraries Loaded!')
print('')
print(datetime.datetime.now(), ': Importing .csv file and prepping the data...')

# Main Meat is from here...


# def read_file(url):
#     url = url + "?raw=true"
#     df = pd.read_csv(url)
#     return df

# url = "https://github.com/BrianENason/Capstone-Project/blob/main/kc_house_data.csv"

# df = read_file(url)

# Import the file into a pandas dataframe
df = pd.read_csv('kc_house_data.csv')

# SET VARIABLES BEFORE RUNNING!
showGraphs = True    # Variable to turn on/off graph making
printGraphs = True   # Variable to turn on/off graph save to .png 
dfAnalysis = False     # Variable to turn on/off Data Frame analysis
debugModeOn = False    # Variable to turn on/off debug
firstModel = False    #Variable to turn on/off first model
largeTreePng = False  # Variable to turn on/off large tree diagram creation
smallTreePng = False  # Variable to turn on/off small tree diagram creation

# import sys
# print(sys.modules.keys())
# !pip freeze > requirements.txt

'''
from google.colab import drive
drive.mount('/content/drive')
'''

"""Once the data is imported and placed in a data frame, we will run analysis of the data."""

# Discover the row/column count of the dataset
# df.shape

if dfAnalysis:
  print(df.shape)

# Display information about the columns including if they have any "null" values and their data type
# df.info()

if dfAnalysis:
  df.info()

# Another way to verify if any "null" values exist
# df.isnull().sum()

if dfAnalysis:
  print(df.isnull().sum())

# Find out if there are any duplicate data rows
# df.duplicated()

if dfAnalysis:
  print(df.duplicated())

# Show the statistics of each column
# df.describe()

if dfAnalysis:
  print(df.describe())

print(datetime.datetime.now(), ': Calculating Average Home Sale Price')
avg2015price = round((df['price'].sum() / df['price'].count()), 2)
print(datetime.datetime.now(), ': Average Home Sale Price in 2015 is $', avg2015price)



"""Looking at the information above gives a better idea of the dataframe. We can see the mean, min and max values in each column as well as other statistics. Noticeable data from above includes:
1. The minimum sale price starts at 75,000 and goes up to 7.7 million, but most sales fall around 540,000
1. There exists some homes without bedrooms and/or bathrooms
1. Waterfront is likely a T/F as the values are only 1 and 0
1. View has a min of 0 and a max of 4.
1. Half of all homes sold were built in the last 45 years 

Using this as our base, we can start cleaning.<br> 
First we have to remove any null data entries.<br> 
Second, we will remove any of the data we don't want and/or don't need in the model.<br> 
Finally, we can then start to run graphs on the data to figure out the important information.
"""

'''
# Drop any rows without data - 
#     NOTE: In the initial dataset it is discovered that there are no Null values, so this is commented out.
df.dropna()
'''

# Remove extra data columns

df.pop('id')
df.pop('date')
df.pop('lat')
df.pop('long')

# Function to convert the tags of the y-axis to more readable form
def format_number(data_value, index):
	if data_value >= 1000000:
		formatter = '{:1.1f}M'.format(data_value*0.000001)
	else:
		formatter = '{:1.0f}K'.format(data_value*0.001)
	return formatter

print(datetime.datetime.now(), ': Data Prepped and Ready!')
print('')
print(datetime.datetime.now(), 'Creating Graphs and Charts From Data')

if showGraphs:    
    
    print(datetime.datetime.now(), ': Creating Scatter Plot 1')
    
    # Assign values from the dataframe to the variables
    x = df['grade']
    y = df['price']
    step_value = 500000

    # Set up size
    fig, ax = plt.subplots(figsize=(14, 7))

    # Create scatter plot
    ax.scatter(x, y)

    # Add a fit line
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b)

    # Format the graph
    plt.xlim(0, 14)
    plt.xlabel("Grade")
    plt.xticks(range(1,14))
    plt.ylim(0, 8000000)
    plt.ylabel("Price (of sale)")
    plt.yticks(np.arange(0, max(y)+step_value, step_value))
    ax.yaxis.set_major_formatter(formatter = FuncFormatter(format_number))
    plt.title('Effects of grade on sale price')

    plt.tight_layout()

    # Add grid line from y-label across the graph
    plt.grid(axis='y')

    # Save the graph to a .png file
    if printGraphs:
        plt.savefig('charts/gradeToPriceComparisonWithFitLineScatterPlot.png')
    print(datetime.datetime.now(), ': Scatter Plot 1 Created!!')

    # View the graph
    # plt.show()

"""Looking at the above graph, we can see that there is a definite correlation between Grade and Price.<br> There are a couple noticeable outliers that we will need to remove before training the model like:
1. Any grade 11 above $4 million 
1. Any grade 2 or below.
"""

# Scatter plot to locate any correlation between grade and year built.
if showGraphs:

    print(datetime.datetime.now(), ': Creating Scatter Plot 2')

    # Set up the axis and size of the chart
    fig, dx = plt.subplots(figsize = (14, 7))
  
    x = df['yr_built']  
    y = df['grade']
  
    plt.scatter(x, y, color='purple')  

    plt.xlabel('Year Built')
    plt.ylabel("Grade")

    plt.xticks(np.arange(min(x) - 1, max(x) + 1, 1), rotation=90)
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.yticks(np.arange(0, max(y) + 1, 1))
    plt.ylim(0, max(y) + 1)

    plt.title('Comparing Year Built to Grade')

    plt.tight_layout()

    # Add lines from the y-axis data across the graph
    plt.grid(axis='y')

    # Create a fit line
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b)

    # Save the graph to a .png file
    if printGraphs == True:
        plt.savefig('charts/gradeToYearBuiltScatter.png')
    print(datetime.datetime.now(), ': Scatter Plot 2 Created!!')
    # View the chart
    # plt.show()

"""The Scatter plot above shows us that there is a density correlation between the grade of the house and the year it was built. The fit line demonstrates this correlation. As the year built goes up, the general grade goes up as well. This makes sense when you consider construction methods, age, etc. all effect the perceived grade of the home."""

# Barchart to see the correlation between the 'view' column and the 'price' column to find any correlation
# NOTE: 'view' data from the Dataset has 0-4 "Views" meaning how many sides of the house have views.

if showGraphs:
    
    print(datetime.datetime.now(), ': Creating Bar Chart 1... (this could take up to 45 seconds)')

    # Assign values from the dataframe to the variables
    x = df['view']
    y = df['price']
    step_value = 500000

    # Set up size
    fig, bx = plt.subplots(figsize=(14, 7))

    # Create Bar Graph
    bx.bar(x, y, color='green')

    # Format the graph
    plt.xlabel("Sides of house with a view")
    plt.ylim(0, 8000000)
    plt.ylabel("Price (of sale)")
    plt.yticks(np.arange(0, max(y)+step_value, step_value))
    bx.yaxis.set_major_formatter(formatter = FuncFormatter(format_number))
    plt.title('Sale Price compared to View')

    plt.tight_layout()

    # Save the graph to a .png file
    if printGraphs == True:
        plt.savefig('charts/homesGroupedBySidesBarChart.png')
    print(datetime.datetime.now(), ': Barchart 1 Created!!')
    # View the chart
    # plt.show()

"""Looking at the above graph, we can see that there really isn't a correlation between how many sides have a view and how much the price of the house was with the exception of:<br>
1. 2, 3, and 4 have better sales than 0
1. 0 has higher sale number than 1<br>

We can consider the pie chart below to see how many of each view-type there actually is before drawing any actual conclusions.
"""

# Pie Chart to see how many houses have how many sides with views
if showGraphs:

    print(datetime.datetime.now(), ': Creating Pie Chart 1')

    # Set up chart size
    plt.subplots(figsize = (14, 7))
    explode = (0, 0.5, 0.5, 0.5, 0.5)

    # Labels to be used in the pie chart (String name correlates to index in "sides" array)
    labels = ['Zero', 'One', 'Two', 'Three', 'Four']

    # Count how many of each of the 5 possibilities of sides there are in the dataset
    sides = df['view'].value_counts()

    # Create Pie Chart of the data
    plt.pie(sides, 
            explode=explode, 
            labels=labels, 
            autopct='%2.1f%%', 
            startangle = -120,
            wedgeprops={"linewidth": 1, "edgecolor": "white"})
    plt.axis('equal')

    # Add title to chart
    plt.title("How many view-sides the houses have as compared to number of sales")

    # Save the graph to a .png file
    if printGraphs == True:
        plt.savefig('charts/homesGroupedBySidePieChart.png')
    print(datetime.datetime.now(), ': Pie Chart 1 Created!!')
    # View the chart
    # plt.show()

"""As we can see in the Pie Chart above, more than 90% of the house sales were of homes with zero view. This would account for the inconclusive data from the bar chart above. in analysis we can see:
1. There is 20X more home sales with 0 view-sides than with one
1. There is almost 2X the number of home sales with two view sides vs one
1. There are equal number of 3 and 4 view-side home sales <br>

Kings County washington is a dense urban setting, so it is logical that most of the home sales are considered as having 0 view-sides.
"""

# Scatter plot to identify any outliers in the data comparing living size to sale price
if showGraphs:

    print(datetime.datetime.now(), ': Creating Scatter Plot 3')

    # Assign values from the dataframe to the variables
    x = df['sqft_living']
    y = df['price']
    step_value = 500000

    # Set up size
    fig, cx = plt.subplots(figsize=(14, 7))

    # Create Bar Graph
    cx.scatter(x, y, color='orange')

    # Format the graph
    plt.xlabel("Living Size (in Sq Feet)")
    # plt.xlim(0, 14000)
    plt.xticks(np.arange(0, max(x) + 250, 250), rotation=45)
    plt.ylim(0, 8000000)
    plt.ylabel("Price (of sale)")
    plt.yticks(np.arange(0, max(y)+step_value, step_value))
    cx.yaxis.set_major_formatter(formatter = FuncFormatter(format_number))
    plt.title('How Living Space size correlates to Sale Price of House')

    plt.tight_layout()

    # Save the graph to a .png file
    if printGraphs == True:
        plt.savefig('charts/livingSizeCompToSalePriceScatter.png')
    print(datetime.datetime.now(), ': Scatter plot 3 Created!!')
    # View the chart
    # plt.show()

"""The Scatter plot above shows the relationship between the living space (in square feet) and the sale price of the house. We can draw 3 conclusions from the above graph:
1. There is a definite correlation between living space and sale price
1. both the upper limits and lower limits increase linearly, but at different slopes indicating the range of sale price widens as the square footage goes up.
1. The data has some extraneous data points that we can clean up (the 2 points above 12000 square feet)
"""

## Scatter plot to identify any outliers in the data comparing living size to sale price
#if showGraphs:

#    print(datetime.datetime.now(), ': Creating Scatter plot 4')

#    # Assign values from the dataframe to the variables
#    x = df['sqft_lot']
#    y = df['price']
#    step_value = 500000

#    # Set up size
#    fig, cx = plt.subplots(figsize=(14, 7))

#    # Create Scatter Plot
#    cx.scatter(x, y, color='orange')

#    # Format the graph
#    plt.xlabel("Lot Size (in Sq Feet)")
#    # plt.xlim(0, 14000)
#    plt.xticks(np.arange(0, max(x) + 100000, 100000), rotation=90)
#    plt.ylim(0, 8000000)
#    plt.ylabel("Price (of sale)")
#    plt.yticks(np.arange(0, max(y)+step_value, step_value))
#    cx.yaxis.set_major_formatter(formatter = FuncFormatter(format_number))
#    plt.title('How Lot size correlates to Sale Price of House')

#    plt.tight_layout()

#    # Save the graph to a .png file
#    if printGraphs == True:
#        plt.savefig('charts/lotSizeCompToSalePriceScatter.png')
#    print(datetime.datetime.now(), 'Scatter plot 4 Created!!')
#    # View the chart
#    # plt.show()

# Scatter plot to locate any correlation between year built and sale price.
if showGraphs:

    print(datetime.datetime.now(), ': Creating Scatter Plot 5')

    # Set up the axis and size of the chart
    fig, dx = plt.subplots(figsize = (14, 7))
    dx.yaxis.set_major_formatter(formatter = FuncFormatter(format_number))

    # Transfer the data frame information into x and y variables for ease of graphing
    x = df['yr_built']
    y = df['price']
    step_value = 500000

    # plt.subplots(figsize = (18, 9))
    plt.scatter(x, y, color='purple')

    # Add labels to the chart
    plt.xlim(1899, 2020)
    plt.xlabel('Year the house was built')
    plt.xticks(np.arange(1900, max(x) + 5, 5))
    plt.ylim(0, 8000000)
    plt.ylabel("Price (of sale)")
    plt.yticks(np.arange(0, max(y)+step_value, step_value))
    plt.title('How much the house sold for compared to the Year it was built')

    plt.tight_layout()

    # Add lines from the y-axis data across the graph
    plt.grid(axis='y')

    # Create a fit line
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b)
  
    # Save the graph to a .png file
    if printGraphs == True:
        plt.savefig('charts/yrBuiltToSalesScatter.png')
    print(datetime.datetime.now(), ': Scatter plot 5 Created!!')
    # View the chart
    # plt.show()

"""The scatter plot above compares the year a house was built to how much it sold for. Some interesting points of interest are:
1. Looking at the blue "Best Fit" line, we can see that there is relatively little change between how much a house sells for as compared to the year it was built.
1. There are some outlier data points we can clean up to better train the model, most notably the 3 above $6,000,000
"""

# Creating Histogram of number of sales per year built

if showGraphs:

    print(datetime.datetime.now(), ': Creating Histogram 1')

    # Set up the size of the graph
    plt.subplots(figsize = (14, 7))

    # Pull Zipcode data, sort it, and convert it to a "String" type
    yearToString = (df['yr_built'].sort_values()).astype('string')

    # Set bin to length of the array (70 in this case so far)
    binCount = len(yearToString.value_counts())

    # Set up the graph's data
    plt.hist(
    yearToString,
    align='mid',  
    bins=binCount,
    facecolor='black', 
    rwidth=0.75)

    # Set up X and Y Axis increments and labels
    plt.xlabel('Year')
    plt.xticks(rotation=45)
    plt.xlim(yearToString.min(), yearToString.max())

    plt.ylabel('Number of sales compared to Year Built')
    # plt.yticks(np.arange(0, 700 + 25, 25))

    # Give the histogram a title
    plt.title("Frequency of sales per year")

    plt.tight_layout()

    # Add lines from the y-axis data across the graph
    plt.grid(axis='y')

    # Save the graph to a .png file
    if printGraphs == True:
        plt.savefig('charts/yearSalesHistogram.png')
    print(datetime.datetime.now(), ': Histogram 1 Created!!')
    # View the graph
    # plt.show()

"""In the histogram above, we can see the number of home sales that occurred for each division of year the house was built.<br>
1. There is a noticeably slight incline as the year built reaches towards today
1. The oldest houses on record (those built in 1900) have almost 100 sales.
1. The most sales are homes built 2 years prior to 2015

We will use this data to train the model on all home sales, and then on home sales for houses built in the last 50 years.
"""

# Creating Histogram of number of sales per zipcode
if showGraphs:

    print(datetime.datetime.now(), ': Creating Histogram 2')

    # Set up the size of the graph
    plt.subplots(figsize = (14, 7))

    # Pull Zipcode data, sort it, and convert it to a "String" type
    zipcodeToString = (df['zipcode'].sort_values()).astype('string')

    # Set bin to length of the array (70 in this case so far)
    binCount = len(zipcodeToString.value_counts())

    # Set up the graph's data
    plt.hist(
    zipcodeToString,
    align='mid',  
    bins=binCount,
    facecolor='red', 
    rwidth=0.75)

    # Set up X and Y Axis increments and labels
    plt.xlabel('Zipcode')
    plt.xticks(rotation=45)
    plt.xlim(zipcodeToString.min(), zipcodeToString.max())

    plt.ylabel('Number of sales in Zipcode')
    plt.yticks(np.arange(0, 700 + 25, 25))

    # Give the histogram a title
    plt.title("Frequency of sales per zipcode")

    plt.tight_layout()

    # Add lines from the y-axis data across the graph
    plt.grid(axis='y')

    # Save the graph to a .png file
    if printGraphs == True:
        plt.savefig('charts/zipSalesHistogram.png')
    print(datetime.datetime.now(), ': Histogram 2 Created!!')
    # View the graph
    # plt.show()

"""The Histogram above is to deduce what are the zipcodes with the most sales and what zipcodes have no sales. We can see that there are 70 zipcodes in consideration in Kings County, Washington.
1. The most-selling zipcode is 98103
1. The least-selling zipcode is 98039
"""

# Piechart to break down number of homesales per $100,000 increments

if showGraphs:

    print(datetime.datetime.now(), ': Creating Pie Chart 2')

    plt.subplots(figsize = (14, 7))

    # Round Sales Data to Hundred Thousands
    rndHTh = round(df['price'], -5) # "Round to Hundred-Thousands"

    # Convert anything above 1.5 mil
    rndHTh = np.where(rndHTh>1500000, 1500000, rndHTh)

    # Pull out unique values from the data
    sizes = np.unique(rndHTh, return_counts=True)

    # Set label titles
    labelArray=['$100,000', '$200,000', '$300,000', 
                '$400,000', '$500,000', '$600,000', 
                '$700,000', '$800,000', '$900,000', 
                '$1,000,000', '$1,100,000', '$1,200,000', 
                '$1,300,000', '$1,400,000', '$1.5 million +']
    explode = (0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

    # Create pie-plot
    plt.pie(sizes[1], 
            labels=labelArray,
            explode=explode,
            autopct='%2.1f%%', 
            startangle=180,
            wedgeprops={"linewidth": 1, "edgecolor": "white"})
    plt.axis('equal')
    plt.title("Number of homesales per $100K")

    # Save the graph to a .png file
    if printGraphs == True:
        plt.savefig('charts/homesalesPer100KPieChart.png')
    print(datetime.datetime.now(), ': Pie Chart 2 Created!!')
    # plt.show()

"""The Pie Chart divides the sales data into 100,000 increments. From this, we can deduce:
1. There are few home sales in the 0 - 100,000 dollar range
1. The largest range is the 400,000 dollar increment
1. Over half of all home sales are between 200,000 dollars and 599,999 dollars
1. Only 2.6 percent of the sales are homes above 1.5 million dollars

We can now remove outliers identified in the data stats and the graphs from above to generate a more accurate model later on
"""

print(datetime.datetime.now(), ': All Graphs have been Created')
print('')
print(datetime.datetime.now(), ': Cleaning Data Further for Model...')

# Remove outliers
df = df.drop(df.index[df['bedrooms'] > 6])
df = df.drop(df.index[df['bathrooms'] < 1])
df = df.drop(df.index[df['yr_built'] < 1965])

"""Above, we removed any houses with more than 6 bedrooms, less than 1 bathroom, and focused the data on homes built within the last 50 years.<br><br>
Below, we are going to remove some of the outlier data seen in the graphs to give the Model a better chance at accuracy
"""

df = df.drop(df.index[df['price'] > 6000000])
df = df.drop(df.index[df['sqft_living'] > 10000])
df = df.drop(df.index[df['grade'] < 3])

# Show how many rows/columns are left in the dataset with display being (row, column)
# print('Shape of dataset:', df.shape)

if dfAnalysis:
  print('Shape of dataset:', df.shape)

# Create numerical data for any non-numerical entries
df = pd.get_dummies(df)

# For testing/debug purposes:
#   make sure the dataframe has been updated
# df.head(5)

# We want to show price prediction, so we pull out the "price" column and give it its own variable
prices = np.array(df['price'])

# Remove the prediction from the data
df = df.drop('price', axis = 1)

# Save the category names
df_list = list(df.columns)

# Convert data to a numpy array
df = np.array(df)

# This is to ensure the category names have been saved
# df_list

print(datetime.datetime.now(), ': Data is Clean!')
print('')
print(datetime.datetime.now(), ': Setting up Model Parameters...')


if debugModeOn:
  print(df_list)

"""
From here to the next text break, you will find one iteration of the model that will produce (ultimately) the following stats:

Mean Absolute Error: 79437.67 dollars.
Accuracy: 86.18 %.

This is achieved using random_state = 42, test_size = 0.25, and 1000 decision trees."""

# Split the data into training and testing sets. 
#   NOTE: random_state is for reproducibility

testSizePercent =0.25  # 0.25 is the best so far
randomSeed =427        # 427 is the best so far
numTrees =1000         # 1000 is the best so far
maxDepth =None           # None is the best so far (as in, don't  include max_depth)

# NOTE: This setup with 42 as random state and test_size of 0.25 works with the further code
train_df, test_df, train_prices, test_prices = train_test_split(df, prices, test_size = testSizePercent, random_state = randomSeed)

'''
# For testing/debug purposes:
#   This will show how many (row, columns) are in the divided sets
print('Training Dataframe Shape:', train_df.shape)
print('Training Prices Shape:', train_prices.shape)
print('Testing Dataframe Shape:', test_df.shape)
print('Testing Prices Shape:', test_prices.shape)
'''

if firstModel:
  print(datetime.datetime.now(), ': Creating an Initial Model...')
  # Instantiate model with 1000 decision trees. 
  #   NOTE: random_state is for reproducibility

  # NOTE: This setup with 427 as random state and 1000 trees outputs: 
  #   Mean Absolute Error: 80191.55 dollars and Accuracy: 85.99 %.
  rf = RandomForestRegressor(n_estimators = numTrees, random_state = randomSeed, max_depth=maxDepth)

  # Train the model on training data
  rf.fit(train_df, train_prices);

  # Use the forest's predict method on the test data
  predictions = rf.predict(test_df)

  # Calculate the absolute errors
  errors = abs(predictions - test_prices)

  # Print out the mean absolute error (M.A.E.)
  print('Mean Absolute Error:', round(np.mean(errors), 2), 'dollars.')

  # Find the mean absolute percentage error
  meanError = 100 * (errors / test_prices)

  # Calculate and display accuracy
  accuracy = 100 - np.mean(meanError)
  print('Accuracy:', round(accuracy, 2), '%.')

# For testing/debug purposes:
#   Display what the Initial Model thinks is most important determining factors

if dfAnalysis and firstModel:
  model_ranks=pd.Series(rf.feature_importances_, index=df_list, name="Importance")
  ax=model_ranks.plot(kind='barh')

print(datetime.datetime.now(), ': Creating Random Forest Regression Model...(this could take up to a minute or more)')

# New random forest with only the two most important variables
# NOTE: This setup with 42 as random state and 1000 trees outputs: 
#   Mean Absolute Error: 79437.67 dollars and Accuracy: 86.18 %.
rf_most_important = RandomForestRegressor(n_estimators= numTrees, random_state=randomSeed, max_depth=maxDepth)

# Extract the most important features from the dataframe
important_indices = [df_list.index('zipcode'),
                     df_list.index('yr_built'),
                     df_list.index('bedrooms'),
                     df_list.index('bathrooms'),
                     df_list.index('view'),
                     df_list.index('waterfront'),
                     df_list.index('sqft_living'),
                     df_list.index('sqft_lot'),
                     df_list.index('sqft_above'),
                     df_list.index('sqft_basement'),
                     df_list.index('grade')]
                     
train_important = train_df[:, important_indices]
test_important = test_df[:, important_indices]

# List of headers for analysis graph
df_new_list = ('zipcode', 'yr_built', 'bedrooms', 'bathrooms', 'view', 'waterfront',
                   'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'grade')

# Train the random forest
rf_most_important.fit(train_important, train_prices)

# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_prices)

print(datetime.datetime.now(), ': Model is Complete. Testing Accuracy of Model')

# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'dollars.')
mape = np.mean(100 * (errors / test_prices))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

"""This is the end of the "Good" base model running. Commented out for further debugging purposes to see if there is a model combination with better results to be found"""

# For testing/debug purposes:
#   Show new model's most important features that affect price

if dfAnalysis:
  model_ranks=pd.Series(rf_most_important.feature_importances_, index=df_new_list, name="Importance")
  ax=model_ranks.plot(kind='barh')  

#This cell is for creating a large visual of the decision tree
if largeTreePng:
  #Create a visual of the Random Forest Decision Tree
  # Pull out one tree from the forest
  tree = rf.estimators_[5]

  # Export the image to a dot file
  export_graphviz(tree, out_file = 'housePriceTree.dot', feature_names = df_list, rounded = True, precision = 1)

  # Use dot file to create a graph and Write graph to a png file
  (graph, ) = pydot.graph_from_dot_file('housePriceTree.dot')
  graph.write_png('housePriceTree.png')

#This cell is for creating a small visual of the decision tree
if smallTreePng:
  # Create a smaller tree for visual of the Random Forest Decision Tree limited to a depth of 3
  rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
  rf_small.fit(train_df, train_prices)

  # Extract the small tree
  tree_small = rf_small.estimators_[5]

  # Save the tree as a png image
  # export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = df_list, rounded = True, precision = 1)
  (graph, ) = pydot.graph_from_dot_file('small_tree.dot')
  graph.write_png('small_tree.png');

# For Testing/Debug purposes:
#   Sending in random values to make sure the algorithm spits out something significant

if dfAnalysis:
  singleValues = [[8, 2400,98074, 2011, 3, 1, 8090, 3, 5, 1200, 800]]
  single_prediction = rf_most_important.predict(singleValues)
  single_prediction

print(datetime.datetime.now(), ': Testing Single Input to Model...')

# Sending in actual data from one random line to see the model's output.
# This should output ~510000
newSingleValues = [[98074, 1987, 3, 2, 0, 0, 1680, 8080, 1680, 0, 8]]
new_single_prediction = rf_most_important.predict(newSingleValues)
print(new_single_prediction)

# Function for converting predicted cost to today's numbers using inflation average for Kings County

# Where to find the inflation of homeprice median for calculation:
# https://www.redfin.com/county/118/WA/King-County/housing-market

def inflationKC(basePrice):
  priceAvg2015 = avg2015price  # Median Home sale Price in 2015 calculated from data
  priceAvg2021 = 805000.00  # Median Home sale Price in 2021 from website in comment above
  inflation = float(priceAvg2015) / float(priceAvg2021)  
  newPrice = (basePrice * (inflation + 1))
  return newPrice

# Sends in the numerical value from the output array created by the model (The answer to the question: "What is the sale price"). 
# Output is the adjusted numerical value as is, not in an array!
print("Adjusted for Inflation: $",  round(inflationKC(new_single_prediction[0]), 2))

print(datetime.datetime.now(), ': Single model input testing complete')

'''
# This code cell is for exporting of the model in one of three ways. NOTE: The first is 750MB, the second and third are around 150mb

import joblib
import os

joblib.dump(rf_most_important, "./random_forest.joblib")

joblib.dump(rf_most_important, "RF_uncompressed.joblib", compress=0) 
print(f"Uncompressed Random Forest: {np.round(os.path.getsize('RF_uncompressed.joblib') / 1024 / 1024, 2) } MB")

joblib.dump(rf_most_important, "RF_compressed.joblib", compress=3)  # compression is ON!
print(f"Compressed Random Forest: {np.round(os.path.getsize('RF_compressed.joblib') / 1024 / 1024, 2) } MB")

joblib.dump(rf_most_important, "RF_Uber_compressed.joblib", compress=6)  # compression is ON!
print(f"Uber Compressed Random Forest: {np.round(os.path.getsize('RF_Uber_compressed.joblib') / 1024 / 1024, 2) } MB")
'''

# !pip freeze > requirements.txt

# ...to Here

def waterFrontDecision(onWater):
  isOnWater = 0;
  if onWater == True:
    isOnWater = 1
  return isOnWater    

def convertToInt(posInt):
    newPosInt = 0
    try:
        newPosInt = int(posInt)
    except:
        newPosInt = -1
    return newPosInt

print('')
print(datetime.datetime.now(), ': Loading GUI...')

def make_window():
    sg.theme('Kayak')   
        
    zipcodes = [98001, 98002, 98003, 98004, 98005, 98006, 98007, 98008, 98010, 98011, 98014, 98019, 98022, 98023, 98024, 98027, 98028, 98029, 98030, 98031, 98032, 98033, 98034, 98038, 98039, 98040, 98042, 98045, 98052, 98053, 98055, 98056, 98058, 98059, 98065, 98070, 98072, 98074, 98075, 98077, 98092, 98102, 98103, 98105, 98106, 98107, 98108, 98109, 98112, 98115, 98116, 98117, 98118, 98119, 98122, 98125, 98126, 98133, 98136, 98144, 98146, 98148, 98155, 98166, 98168, 98177, 98178, 98188, 98198, 98199] # [91919, 92929, 93939, 94949, 95959, 96969]
    bedrooms = [1, 2, 3, 4, 5, 6, 7, 8]
    bathrooms = [1, 2, 3, 4, 5, 6, 7, 8]
    views = [0, 1, 2, 3, 4]
    grade = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    year = [] 
    for i in range(1900,( datetime.datetime.now().year + 1)):
        year.append(i)    

    main_layout =  [
	    [sg.Text('Set the following values and then click "Submit" when finished')],
        [sg.Text('Name This Collection'), sg.Input(default_text='Collection Name',size=(20,1), key='-NAMECOLLECTION-')],
	    [sg.Text('Zip Code'), sg.Combo(values=(zipcodes), default_value=zipcodes[0], readonly=False, k='-ZIPCODE-')],
        [sg.Text('Year Built'), sg.Combo(values=(year), default_value=year[0], readonly=False, k='-YEARBUILT-')],	    
	    [sg.Text('Bedrooms'), sg.Combo(values=(bedrooms), default_value=bedrooms[0], readonly=False, k='-BEDROOMS-')],
	    [sg.Text('Bathrooms'), sg.Combo(values=(bathrooms), default_value=bathrooms[0], readonly=False, k='-BATHROOMS-')],
	    [sg.Text('Sides with View'), sg.Combo(values=(views), default_value=views[0], readonly=False, k='-VIEWS-')],
	    [sg.Checkbox('Check if on Waterfront', default=False, k='-WATERFRONT-')],
        [sg.Text('Inside Square footage'), sg.Input(default_text='200',size=(6,1), key='-SQFTLIVING-')],
        [sg.Text('Lot Square footage'), sg.Input(default_text='200',size=(6,1), key='-SQFTLOT-')],
        [sg.Text('Above Ground Inside Square footage'), sg.Input(default_text='200',size=(6,1), key='-SQFTABOVE-')],
        [sg.Text('Basement Square footage'), sg.Input(default_text='200',size=(6,1), key='-SQFTBASEMENT-')],
        [sg.Text('Grade'), sg.Combo(values=(grade), default_value=grade[0], readonly=False, k='-GRADE-')],        
	    [sg.Button('Submit'), sg.Button('Exit')]] 

    logging_layout = [
        [sg.Text("History of your Collections")],
        [sg.Multiline(size=(60,15), font='Courier 8', expand_x=True, expand_y=True, write_only=True, reroute_stdout=True, reroute_stderr=True, echo_stdout_stderr=True, autoscroll=True, auto_refresh=True)],
        [sg.Button('Exit', key='Exit2')]]
    
    graphing_layout = [
        [sg.Text("Select the Graph you wish to view")],
        [sg.Image(key='-GRAPH-', size=(16, 8))],
        [sg.Text(key='-DESCGRAPH-')],
        [sg.Button('Grade to Price', key='-SPLOT1-'), 
            sg.Button('Homesale per 100K', key='-PIE1-'), 
            sg.Button('Sales per Build Year', key='-HIST1-'),
            sg.Button('Grade to Year', key='-SPLOT2-'),            
            sg.Button('Living size to Price', key='-SPLOT3-'),
            sg.Button('Year to Price', key='-SPLOT4-'),
            sg.Button('Homes with View', key='-PIE2-'),
            sg.Button('Sales per Zip', key='-HIST2-'),                        
            sg.Button('Clear Charts', key='-CLR-')],
        [sg.Button('Exit', key='Exit3')]]  # sg.Button('Lot to Price', key='-SPLOT5-'),

    about_layout = [
        [sg.Text('Price/Cost Estimator for Kings County WA')],
        [sg.Text('Version 1.2.1')],        
        [sg.HSeparator()],
        [sg.Text('Brian Nason')],
        [sg.Text('Student Number: 001003011')],
        [sg.Text('email: bnason1@wgu.edu')],
        [sg.Text('')],
        [sg.Button('Exit', key='Exit4')]]

    
    layout = [[sg.Text('House Price/Cost Estimator', size=(38, 1), justification='center', font=("Helvetica", 16), relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=False)]]
    layout +=[[sg.TabGroup([[  sg.Tab('Main Page', main_layout),
                               sg.Tab('Graphs', graphing_layout),
                               sg.Tab('About', about_layout),                               
                               sg.Tab('Log', logging_layout)]], key='-TAB GROUP-', expand_x=True, expand_y=True),

               ]]
    layout[-1].append(sg.Sizegrip())
    # window = sg.Window('Capstone Final Project', layout, grab_anywhere=False, resizable=True, margins=(0,0), location=(0, 0), use_custom_titlebar=True, finalize=True, keep_on_top=True)
    window = sg.Window('Capstone Final Project', layout, no_titlebar=False, disable_close=True, resizable=True, margins=(20, 20), location=(0, 0), finalize=True, keep_on_top=True)    
    window.set_min_size(window.size)
    
    window.DisableClose=True
    return window

def main():
    window = make_window()    
    print('')
    print(datetime.datetime.now(), ': GUI Loaded')
    collectionNames = []
    pngCollection = ['charts/gradeToPriceComparisonWithFitLineScatterPlot.png',
                    'charts/gradeToYearBuiltScatter.png', 'charts/homesGroupedBySidesBarChart.png',
                    'charts/homesGroupedBySidePieChart.png', 'charts/livingSizeCompToSalePriceScatter.png',
                    'charts/lotSizeCompToSalePriceScatter.png', 'charts/yrBuiltToSalesScatter.png', 
                    'charts/yearSalesHistogram.png', 'charts/zipSalesHistogram.png', 'charts/homesalesPer100KPieChart.png']
    
    #resultName2 = 'Not Found'
    #resultPrice2 = 'Not Found'

    # This is an Event Loop 
    while True:
        event, values = window.read(timeout=100)

        if event in (None, 'Exit', 'Exit2', 'Exit3', 'Exit4'):
            for png in pngCollection:
                try:
                    os.remove(png)
                    print(datetime.datetime.now(),': Deling file:', png)
                except:
                    print(datetime.datetime.now(), png, ' does not exist')
            print(datetime.datetime.now(), ": Clicked Exit!")
            break

        elif event == '-SPLOT1-':
            print(datetime.datetime.now(), 'Grade to Price graph selected by user (-SPLOT1-)')
            window['-GRAPH-'].update('charts/gradeToPriceComparisonWithFitLineScatterPlot.png')
            window['-DESCGRAPH-'].update('Looking at the above graph, we can see that there is a definite correlation between Grade and Price.\nThere are a couple noticeable outliers that we will need to remove before training the model like:\n1. Any grade 11 above $4 million \n2. Any grade 2 or below.')

        elif event == '-SPLOT2-':
            print(datetime.datetime.now(), 'Grade to Year graph selected by user (-SPLOT2-)')
            window['-GRAPH-'].update('charts/gradeToYearBuiltScatter.png')
            window['-DESCGRAPH-'].update("The Scatter plot above shows us that there is a density correlation between the grade of the house and the year it was built. The fit line demonstrates this correlation. \nAs the year built goes up, the general grade goes up as well. This makes sense when you consider construction methods, age, etc. all effect the perceived grade of the home.")

        elif event == '-SPLOT3-':
            print(datetime.datetime.now(), 'Living size to Price graph selected by user (-SPLOT3-)')
            window['-GRAPH-'].update('charts/livingSizeCompToSalePriceScatter.png')
            window['-DESCGRAPH-'].update("The Scatter plot above shows the relationship between the living space (in square feet) and the sale price of the house. We can draw 3 conclusions from the above graph:\n1. There is a definite correlation between living space and sale price.\n2. both the upper limits and lower limits increase linearly, but at different slopes indicating the range of sale price widens as the square footage goes up.\n3. The data has some extraneous data points that we can clean up (the 2 points above 12000 square feet)")

        elif event == '-SPLOT4-':
            print(datetime.datetime.now(), 'Year to Sale graph selected by user (-SPLOT4-)')
            window['-GRAPH-'].update('charts/yrBuiltToSalesScatter.png')
            window['-DESCGRAPH-'].update("The scatter plot above compares the year a house was built to how much it sold for. Some interesting points of interest are:\n1. Looking at the blue \"Best Fit\" line, we can see that there is relatively little change between how much a house sells for as compared to the year it was built.\n2. There are some outlier data points we can clean up to better train the model, most notably the 3 above $6,000,000")

        #elif event == '-SPLOT5-':
        #    window['-GRAPH-'].update('charts/lotSizeCompToSalePriceScatter.png')
        #    window['-DESCGRAPH-'].update("Quick Description of this Scatter plot")

        elif event == '-PIE1-':            
            print(datetime.datetime.now(), 'Home sales to 100K Pie Chart selected by user (-PIE1-)')
            window['-GRAPH-'].update('charts/homesalesPer100KPieChart.png')
            window['-DESCGRAPH-'].update("The Pie Chart divides the sales data into 100,000 increments. From this, we can deduce:\n1. There are few home sales in the 0 - 100,000 dollar range\n2. The largest range is the 400,000 dollar increment\n3. Over half of all home sales are between 200,000 dollars and 599,999 dollars\n4. Only 2.6 percent of the sales are homes above 1.5 million dollars\nWe can now remove outliers identified in the data stats and the graphs from above to generate a more accurate model later on")

        elif event == '-PIE2-':            
            print(datetime.datetime.now(), 'Home sales vs. Num View Sides Pie Chart selected by user (-PIE2-)')
            window['-GRAPH-'].update('charts/homesGroupedBySidePieChart.png')
            window['-DESCGRAPH-'].update("As we can see in the Pie Chart above, more than 90% of the house sales were of homes with zero view. This would account for the inconclusive data from the bar chart above. in analysis we can see:\n1. There is 20X more home sales with 0 view-sides than with one\n2. There is almost 2X the number of home sales with two view sides vs one\n3. There are equal number of 3 and 4 view-side home sales\nKings County washington is a dense urban setting, so it is logical that most of the home sales are considered as having 0 view-sides.")

        elif event == '-HIST1-':
            print(datetime.datetime.now(), 'Sales per Year Histogram selected by user (-HIST1-)')
            window['-GRAPH-'].update('charts/yearSalesHistogram.png')
            window['-DESCGRAPH-'].update("In the histogram above, we can see the number of home sales that occurred for each division of year the house was built:\n1. There is a noticeably slight incline as the year built reaches towards today\n2. The oldest houses on record (those built in 1900) have almost 100 sales.\n3. The most sales are homes built 2 years prior to 2015\nWe will use this data to train the model on all home sales, and then on home sales for houses built in the last 50 years.")

        elif event == '-HIST2-':
            print(datetime.datetime.now(), 'Sales per Zipcode Histogram selected by user (-HIST2-)')
            window['-GRAPH-'].update('charts/zipSalesHistogram.png')
            window['-DESCGRAPH-'].update("The Histogram above is to deduce what are the zipcodes with the most sales and what zipcodes have no sales. We can see that there are 70 zipcodes in consideration in Kings County, Washington.\n1. The most-selling zipcode is 98103\n2. The least-selling zipcode is 98039")

        elif event == '-CLR-':
            print(datetime.datetime.now(), 'Clear Chart Screen selected by user (-CLR-)')
            window['-GRAPH-'].update()
            window['-DESCGRAPH-'].update('Select a button below to view the chart')

        elif event == 'About':
            print(datetime.datetime.now(), ": Clicked on About!")
            sg.popup('Price/Cost Estimator for Kings County WA',
                     'Version 1.2.1',
                     'Brian Nason',
                     'Student Number: 001003011',
                     'email: bnason1@wgu.edu', keep_on_top=True)

        elif event == 'Submit':
            print(datetime.datetime.now(), ': User Clicked Submit')

            values['-WATERFRONT-'] = waterFrontDecision(values['-WATERFRONT-'])            
            values['-SQFTLIVING-'] = convertToInt(values['-SQFTLIVING-'])
            values['-SQFTLOT-'] = convertToInt(values['-SQFTLOT-'])
            values['-SQFTABOVE-'] = convertToInt(values['-SQFTABOVE-'])
            values['-SQFTBASEMENT-'] = convertToInt(values['-SQFTBASEMENT-'])   
            
            rawInput = []
            modelInput = []
            formatInput = []
            resultName = values['-NAMECOLLECTION-']
            
            for key in values:
                rawInput.append(values[key])            
            for i in range(1, 12):
                modelInput.append(rawInput[i])
            
            negList = list(filter(lambda x: (x < 0), modelInput))

            if (resultName in collectionNames):
                print(datetime.datetime.now(), ": Name Element Already Exists!!")
                sg.popup('The name you chose:' , resultName,'Has Already been used.', 'Please change it to continue!',keep_on_top=True)
            elif (len(negList) > 0):
                print(datetime.datetime.now(), ": Invalid Data entered!!")
                sg.popup('Invalid Data - Check your input fields',keep_on_top=True)
            else:
                collectionNames.append(resultName)
                print(datetime.datetime.now(), ': Collection ', resultName , 'Has the following data:',modelInput)            
                collectionNames.append(resultName)
                formatInput.append(modelInput)
                # print(formatInput)
                new_single_prediction = rf_most_important.predict(formatInput)
                print(new_single_prediction)
                resultPriceBase = new_single_prediction[0];
                resultPriceInfl = round(inflationKC(resultPriceBase), 2)

                # Where we get the model involved
                print(datetime.datetime.now(), ': Result for ', resultName, 'is: ', resultPriceBase, 'With inflation, it is', resultPriceInfl )
                sg.popup('According to your parameters in', resultName, 'Your house price in 2015 is:', '$' + str(resultPriceBase), 'Adjusted for inflation, your 2022 price is:',  '$' + str(resultPriceInfl),  keep_on_top=True)


    window.close()        
    exit(0)



if __name__ == '__main__':        
    sys.exit(main())



# https://mybinder.org/v2/gh/BrianENason/Capstone-Project/HEAD