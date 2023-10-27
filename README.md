## Introduction

🎃😱 Happy Halloween!!! 🎃😱

This workshop introduces the fundamentals of data analysis using Python. We will delve into fetching data from remote sources, reading data from CSV files, preprocessing and cleaning data, understanding basic descriptive statistics, checking for outliers, employing visualization techniques, and providing an overview of forecasting.

Our primary goal is to strengthen your proficiency in Python for data exploration and analysis. The techniques we'll cover are universal and relevant to any field or dataset. 
 
**Why Python?**

- Python is user-friendly and versatile
- Rich set of tools & libraries for data ecosystem (Pandas, Numpy, scikit-learn, Matplotlib)
- Strong community support for problem-solving.

**Learning Outcomes**

- Understand and work with common data formats: JSON, CSV, etc.
- Learn basic data preprepocessing with Python.
- Fundamental descriptive statistics.
- Discover tools for cleaning, exploring, and visualizing data.
- Basic predictive analysis.

Please feel free to ask questions or work in teams. 

You can reach the CSC at csc@barnard.edu or reach out to Marko Krkeljas at mkrkelja@barnard.edu. 


## Setting Up

For this workshop, we'll be using [Google Colab Notebooks](https://colab.research.google.com/). 

**What is a Google Colab Notebook?**

  - Google Colab lets you write and execute Python code in your browser.
  - No setup required; just a Google account.

**Creating a Google Colab Notebook**

  1. [Follow this link.](https://colab.research.google.com/)
  2. Click on the "+ New notebook" button on the bottom-left side.

**Other Ways to Use Python**

- **Local Installation**: Python can be installed on your computer. [Follow the instructions here](https://www.python.org/downloads/).  
- **Anaconda distribution**: Comes packaged with many data science libraries, including the popular notebook format that we're using. [Download here.](https://www.anaconda.com/download)
 

> &#x26a0;&#xfe0f; **For this workshop, we highly recommend using Google Colab, as we can't troubleshoot installation issues.**

## Workshop

### 1. Intro to Python (Data Types, Functions, Loops)

**Integers, floats, and strings**

```python
x = 31 # Integer
y = 13.13 # Float
z = "Happy Halloween" # String
print(x, y, z)
```

**Arithmetic**

```python
print(y - 6.47)
```


**Lists**: collections of data types (ints, floats, strings, etc.). Denoted by square brackets:

```python
my_list = ["Raven", "Crow", "Magpie", 4, 9]
my_list[0] # Indexing
```

**Dictionaries**: Key-Value pairs. Denoted by curly braces:

```python
treats = {
    "candy_corn": "Surprisingly tasty.",
    "twix": "A classic.",
    "almond_joy": "Underrated gem."
}

treats["almond_joy"]
```


**List of dictionaries**: A common way to represent data when working with computer programs.

```python
novels = [{'title': 'Grimscribe: His Lives and Works', 'author': 'Thomas Ligotti'}, {'title': 'The Haunting of Hill House', 'author': 'Shirley Jackson'}, {'title': 'The Cipher', 'author': 'Kathe Koja'}, {'title': 'The Turn of the Screw', 'author': 'Henry James'}]
```

**Loops**: Enable us to iterate over sequences.

```python
for novel in novels:
    print(novel["author"] + " wrote " + novel["title"])
```

**Functions**: blocks of reusable code that perform a task. They are denoted by the `def` keyword:

```python
def trick_or_treat(current_day):
    if current_day == 31:
        return "Treat 🎃"
    else:
        return "Wait for Halloween! 👻"

print(trick_or_treat(27))
print(trick_or_treat(31))
```


### 2. Remote Data Fetching

Computer programs communicate with servers via `HTTP` requests to get or send data. This data is often shared in formats like JSON. 

**JSON**: JavaScript Object Notation

- Lightweight data-interchange format.
- Human-readable and easy for machines to parse.

```json
[
	{
	  "name": "Victor Van Dort",
	  "dob": "1843-10-13",
	},
	{
	  "name": "Emily Doe",
	  "dob": "1847-10-31",
	}
]
```

**Dataset**

- [Water Consumption in the City of New York](https://data.cityofnewyork.us/Environment/Water-Consumption-in-the-City-of-New-York/ia2d-e54m)
- [API Endpoint](https://data.cityofnewyork.us/resource/ia2d-e54m.json)

**Fetching the data**: To fetch the data, we make a `GET` request.

```python
import requests
import pandas as pd

# API Endpoint
url = "https://data.cityofnewyork.us/resource/ia2d-e54m.json"

# Fetch the data
response = requests.get(url)

# Convert the JSON data to a pandas DataFrame
if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)
else:
    print("Failed to retrieve data.")
```

Alternatively, to load a CSV from your computer::

```python
path_to_file = "/path/to/your/file.csv"
df = pd.read_csv(path_to_file)
print(df.head())
```

**Getting the columns**

```python
columns = df.columns
print(columns)
```

**Visualize the data**

```python
df.plot(x='year', y=['nyc_consumption_million_gallons_per_day', 'per_capita_gallons_per_person_per_day'])
```

**Discussion**: Why doesn't this work?

### 3. Inspecting & Preprocessing Data

**Explore the Dataset**

- Check dimensions with the `shape` attribute.
- Check the first few rows with `head()`. 
- Check the last few rows with `tail()`.

```python
print(df.shape)
print(df.head())
print(df.tail())
```

**Data Types & Structure**

- Use `dtypes` to check column data types.
- Convert types as necessary using `astype()`.

```python
print(df.dtypes)
```

We expected numbers, but got the `object` type, which usually means strings or mixed values. Try converting to integers: 

```python
df = df.astype(int)
```

**Discussion**: Why does this throw an error? Try opening the data in Excel.

1. [Link to data page.](https://data.cityofnewyork.us/Environment/Water-Consumption-in-the-City-of-New-York/ia2d-e54m)
2. Click `Export` on the top right.
3. Click `CSV` for Excel.
4. Open locally with Excel (or any spreadsheet software). 

**Solution:** convert to `float` first, then `int`.

```python
df = df.astype(float).astype(int)
print(df.dtypes)
```

**Handling Missing Values**

- Detect with `isnull()` and `sum()`.
- Remove using `dropna()`.
- Impute (fill) using `fillna()`.

`isnull` returns a DataFrame where missing data is `True` and existing data is `False`. `sum()` then counts the `True` values per column:

```python
df.isnull().sum()
```

There are no missing values, but if there were, we could fix them like this:

**fillna**: Replace missing values with a specific value or method. For example, using forward-filling (replace by the most recent non-missing value):

```python
df = df.fillna(method='ffill')
df = df.fillna(0) # Alternatively, replace missing values with zeros.
```

**dropna**: Simply drop the rows or columns that contain any missing values:

```python
df = df.dropna()
df = df.dropna(axis=1) # Drop columns by providing an axis paramters. 
```

### 4. Descriptive Statistics

Use `describe` to get statistics such as count, mean, standard deviation, and quartiles:

```python
descriptive_stats = df.describe().astype(int)
descriptive_stats
```

**Querying the data**

**Year with the highest population**:

```python
largest_population_year = df.sort_values(by='new_york_city_population', ascending=False)['year'].iloc[0]
```

**Spooky Digression**

Let's query a different dataset: [**Accused Witches Data Set**](https://www.kaggle.com/datasets/rtatman/salem-witchcraft-dataset)

From the Kaggle link:

> The Accused Witches Data Set contains information about those who were formally accused of witchcraft during the Salem episode. This means that there exists evidence of some form of direct legal involvement, such as a complaint made before civil officials, an arrest warrant, an examination, or court record.

**Fetch the Data**:

```python
salem_url = "https://raw.githubusercontent.com/barnardcsc/Fall-23-Data-Analysis-With-Python-Halloween-Edition/main/accused-witches-data-set.json"
salem_response = requests.get(salem_url)
salem_data = salem_response.json()
salem_df = pd.DataFrame(salem_data)
```

**Accusations by month**:

```python
salem_df['Month of Accusation'].value_counts()
```

**Accusations by city/residence**:

```python
salem_df['Residence'].value_counts()
```

**Quickly, a map!**

```python
import folium

locations = {
    "Andover": [42.6583, -71.1368],
    "Salem Town": [42.5195, -70.8967],
    "Salem Village": [42.5552, -70.8406],
    "Gloucester": [42.6159, -70.6627],
    "Reading": [42.5257, -71.0953],
    "Topsfield": [42.6370, -70.9495],
    "Haverhill": [42.7762, -71.0773],
    "Rowley": [42.7165, -70.8787],
    "Lynn": [42.4668, -70.9495],
    "Beverly": [42.5584, -70.8800],
    "Ipswich": [42.6792, -70.8417],
    "Woburn": [42.4793, -71.1523],
    "Boxford": [42.6652, -70.9787],
    "Billerica": [42.5584, -71.2689],
    "Charlestown": [42.3782, -71.0602],
    "Piscataqua, Maine": [43.0802, -70.7694],
    "Boston": [42.3601, -71.0589],
    "Wells, Maine": [43.3222, -70.5805],
    "Manchester": [42.5770, -70.7676],
    "Malden": [42.4251, -71.0662],
    "Salisbury": [42.8418, -70.8606],
    "Amesbury": [42.8584, -70.9300],
    "Marblehead": [42.5001, -70.8578],
    "Chelmsford": [42.5998, -71.3673]
}

m = folium.Map(location=[42.3601, -71.0589], zoom_start=9)

# Plot each person's residence on the map
for person in salem_data:
    city = person['Residence']
    if city in locations:
        lat, lon = locations[city]
        folium.Marker([lat, lon], tooltip=person['Accused Witch']).add_to(m)

# Display the map
m
```

**Further exploration of this dataset is left to the reader!**

**Correlation Matrix**:

```python
correlation = df['new_york_city_population'].corr(df['nyc_consumption_million_gallons_per_day'])
print(f"The correlation between population and water consumption is: {correlation:.2f}")
```

**Discussion**: How would you interpret the correlation matrix? 


**Handling Outliers**

Check for outliers with IQR (interquartile range). This example looks at `nyc_consumption_million_gallons_per_day`. 

```python
# Calculate Q1, Q3, and IQR
Q1 = df['nyc_consumption_million_gallons_per_day'].quantile(0.25)
Q3 = df['nyc_consumption_million_gallons_per_day'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)     
```

The IQR represents the middle 50% of the data (the "box" in a box-and-whisker plot). Outliers are defined as points that fall outside a range of 1.5 times the IQR, a commonly used threshold.

```python
lower_boundary = Q1 - 1.5 * IQR
upper_boundary = Q3 + 1.5 * IQR
```

The code below:

1. Checks values in the column `'nyc_consumption_million_gallons_per_day'`.
2. Values below `lower_boundary` or above `upper_boundary` are considered outliers.
3. The `|` means OR, so we're looking for values that are either too low OR too high.

```python
outliers = df[(df['nyc_consumption_million_gallons_per_day'] < lower_boundary) | 
              (df['nyc_consumption_million_gallons_per_day'] > upper_boundary)]       
              
print(outliers)
```

This can be confusing to look at, try splitting it out:

```python
df['nyc_consumption_million_gallons_per_day'] < lower_boundary
```

### 5. Data Visualization

**Import `matplotlib`**: 

```python
import matplotlib.pyplot as plt
```

**Line Plot**

```python
df.plot(x='year', y=['nyc_consumption_million_gallons_per_day', 'per_capita_gallons_per_person_per_day'])

plt.xlabel('Year')
plt.ylabel('Gallons')
plt.title('NYC Water Consumption Over Time')

plt.show()
```

**[Box Plot](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.boxplot.html)**

```python
df.boxplot(column=['nyc_consumption_million_gallons_per_day', 'per_capita_gallons_per_person_per_day'])
```

**Fixing this plot**:

1. Add a title
2. Adjusting the figure size
3. Rotating the x-axis labels
4. Adjust padding around subplots with `tight_layout`


```python
df.boxplot(column=['nyc_consumption_million_gallons_per_day', 'per_capita_gallons_per_person_per_day'])

plt.title('Distribution of NYC Water Consumption Metrics') # Adding a title
plt.figure(figsize=(10,6)) # Adjust the figure size
plt.xticks(rotation=45) # Rotating the x-axis labels for better readability
plt.tight_layout()  # Adjusts the layout to prevent overlap
plt.show()
```

**Altair**: [A Python visualization library, similar to R's ggplot](https://altair-viz.github.io/index.html)

> &#x26a0;&#xfe0f; Note: If you face issues using Altair, install it by running `pip install altair vega_datasets` inside a code-block.


```python
import altair as alt

alt.Chart(df).mark_line().encode(
    x='year:O',
    y='nyc_consumption_million_gallons_per_day:Q'
).properties(title='NYC Water Consumption Over Time')
```

**With a couple of modifications**: 

```python
alt.Chart(df).mark_line().encode(
    x=alt.X("year:O", title="Year"),
    y=alt.Y(
        "nyc_consumption_million_gallons_per_day:Q", title="Million Gallons per Day"
    ),
).properties(
    title={
        "text": "NYC Water Consumption Over Time",
        "subtitle": "Based on available data",
        "align": "left",
        "anchor": "start",
    }
)
```

**[Lastly, Seaborn](https://seaborn.pydata.org/examples/index.html)**


```python
import seaborn as sns
sns.distplot(df['nyc_consumption_million_gallons_per_day'], bins=10, color='red')
plt.title("Distribution of NYC Water Consumption")
plt.show()
```

### Making Predictions

To forecast future values, we'll first build a model using past data. We'll explore a basic example using Python's `statsmodels` and discuss other methods. This is a starting point to understand predictive modeling.

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.ols(formula="nyc_consumption_million_gallons_per_day ~ year", data=df).fit()
```

**`model = smf.ols(...)`**: 

- `ols` stands for "ordinary least squares"; the model finds the line that best fits the data by minimizing the distance between the predicted line and the actual data.

**`formula='nyc_consumption_million_gallons_per_day ~ year'`**: 

- The `~` separates the dependent variable (what we want to predict) from the independent variable (what we use to make predictions). 
- We're predicting `nyc_consumption_million_gallons_per_day` based on the `year`.

**`.fit()`**: 

- Finds the line that best fits by performing the OLS calculations to minimize the difference between predicted and actual values.
  
**Add predictions to the dataframe**:

```python
predictions = model.predict(df['year'])
df['predicted'] = model.predict(df['year'])
```

Compute the difference between actual and predicted values (residuals), then add to the dataframe:

```python
df['residuals'] = df['nyc_consumption_million_gallons_per_day'] - df['predicted']
```

**Visualize the actual and predicted water consumption**:

```python
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['nyc_consumption_million_gallons_per_day'], 'o', label='Actual')
plt.plot(df['year'], df['predicted'], 'r--', label='Predicted')
plt.title('Water Consumption: Actual vs. Predicted')
plt.xlabel('Year')
plt.ylabel('Million Gallons per Day')
plt.legend()
plt.show()
```

**Predict water consumption for 2023 to 2025**:

```python
new_data = pd.DataFrame({'year': [2023, 2024, 2025]})
new_data['predicted_consumption'] = model.predict(new_data)
```

**Visualize the actual and forecasted water consumption**:

```python
plt.figure(figsize=(12, 6))

# Original data
plt.plot(df['year'], df['nyc_consumption_million_gallons_per_day'], 'o-', label='Actual')

# Forecasted data
plt.plot(new_data['year'], new_data['predicted_consumption'], 'r^--', label='Predicted')

plt.title('Water Consumption: Actual vs. Forecasted')
plt.xlabel('Year')
plt.ylabel('Million Gallons per Day')
plt.legend()
plt.grid(True)
plt.show()
```

**For advanced forecasting**:

**scikit-learn**: Python libraries for data analysis, modeling, and machine learning.

**Time Series Models**: Such as ARIMA or Facebook's Prophet, tailored for time-based data.

**Machine Learning Models**: Like Random Forests or Gradient Boosting, useful when adding more data features.
