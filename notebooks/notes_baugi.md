# Notes and Suggestions 

## EDA
- Categorical data - simple countplots
- Numeric data - histplots to determine distribution, boxplots to detect outliers
- Adding some pairplots and other visualizations?
- Correlation Heatmap
- Something else?

## Handling Missing Values
- For encoding categorical variables is SimpleImputer enough?
- Impute then encode or the other way around?
- For numeric data IterativeImputer - should be fine, the size of data is small

## Encoding
- SimpleEncoder for variables that have only 2 unique values
- OneHotEncoder for the rest

## Scaling
- MinMaxEncoder for all data
- Something else?

## Splitiing the Data

## Feature Selection
- Correlation feature selecion
- Mutual Information feature selection
