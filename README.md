# Starbucks Marketing Project 

## Project Overview

The aim of this project is to use data provided by Starbucks in order to understand how each marketing campagin performed on different consumer demographics and use that information to predict how a campagins may perform in the future. The bussiness case for this is that having better knowledge of which users will be more influenced by which campaigns will ultimatly help improve efficentcies in Starbucks marketing strategy. It will also potentially allow Starbucks to model ahead how well marketing will perform so that they can adjust these stratergies when needed.

A blog has been created on this project of judging marketing spend on medium here: 
https://medium.com/@wrommelrath/predicting-marketing-performance-with-machine-learning-c8472bc7807?sk=467a23f098a6eb8bad7a52441c2a4c8c


## Performing the Analysis

The layout of this project will follow the priciples of the CRISP-DM framework starting with the buissness understanding as outlined above and then moving through data understanding, data prepartaion, modelling and finally evaluation. 

There are a number of different ways the project could have been undertaken but after the intital data understanding stage it was decided to split the project into two core parts:

- The first involved trying to understand the main demographic groups that use Starbucks on a regular basis. This will help the understanding of which users are more easily influenced by the marketing campagins and therefore help the pediction modelling later on. It was decided to do this using the K-means unsupervised clustering method as it is a quick and easily applied unsupervised learning approach.

- The second stage would is to build a model on the back of these demographics that can predict how user spend would change under the influence of the offers Starbucks has live. This stage will involve using regression models using xgboost in order to try to predict daily user spend. This model was choosen as it high accuracy and low computation time but it has many Hyper Parameters to tune. In order to speed up the tuning of these parameters I decided to use xgboosts native API for testing as it's quicker and more flexible than the SKlearn equivalent. This model will be judged by looking at the Root Mean Squared Error (RMSE), this performance metric is useful when errors accross the whole dataset need to be kept to a minimum. This is useful as I will be aiming to predict every day properly no matter if it is a slight outlier or not.

A third round of modeling was initially planned to predict the likelihood of a user making a transaction on any given day. However whilst performing the cleaning & formatting phase of the process it became apparent that this behaviour would be very difficult from given data. In order to potentially perfrom this analysis in future a longer time period would be needed without any marketing running to estabish a baseline rate of normal transactions for users in each demogaphic and judge user behaviour.

I will split the CRISP-DM process into the following notebooks in this project:
1. Starbucks Customer Data (Data Understanding).ipynb
2. Starbucks Customer Behaviour (Data Cleaning, Formatting & Processing).ipynb
3. Starbucks Demographics (Data Modelling - Part 1).ipynb
4. Starbucks Customer Spend (Data Modelling - Part 2).ipynb
5. Starbucks Customer Insights (Data Evaluation).ipynb

## Packages needed for this analysis
Some of the pythong packages that may need to be installed are as follows. They can be ran in a notebook or in the terminal:
- ! pip install xgboost
- ! pip install seaborn
- ! pip install joblib


## Data Sources

The following data sources have been provided by Starbucks to undertake this analysis:
- portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
- profile.json - demographic data for each customer
- transcript.json - records for transactions, offers received, offers viewed, and offers completed

All of the datasources above will be explored in more depth in the first notebook on Data Understanding.

## Results

The results above show a lot of promise and many more insights could gained from them already. I was able to successfully predict the spend with some success although this could be improved upon. I was also able to judge marketing performance on each demographic using a baseline created through the prediction modelling. 

To solve the problem of judging marketing performance, I processed the Starbucks membership data making some assumptions in order to create a useful modelling input. Next, I modelled the different demographics that use the Starbucks membership in order to group together users with similar behaviours using the K-means clustering. Finally, I created a regression model for each demographic using the XGBoost method to predict spend and quantify a baseline that could be use to determine marketing performance.

However, there are also many improvements that could be made to the results if more time and data were given to the project. In this short section I will go through a few of the improvements that could be made:

- One of the biggest changes could be an additional model that calculates the chance of an actual transaction taking place and being influenced. This was originally in my scope of work for this project as currently it's not a true predict as currently I'm relying on the fact that I know what transactions will be influenced to predict spend. This third model would tie everything together but I found that in order to create a decent model we would potentially need much more training data than just one month of sales.


- A second improvement would be simply to get a much larger training set and adding variables around the actual products that users are buying. This would potentially add another layer to the predictions and would probably result in better results from the modeling.


- In a similar way I could have used more of the data behind the campaigns such as what format they are web, social etc. This could have improved the model as it would weight similar campaigns in a similar way.


- Finally, as mentioned in the first section a lot more analysis can be done with these models. I could loop through zeroing each campaign before performing a prediction, this would show which campaigns are most effective on each demographic. Or I could look further into when each campaign should be deployed as some demographics probably respond better to offers on different days (this is already shown by a varying baseline on the graphs above.

Using the two models and potentially a third (probability of transactions) could help Startbucks or any business improve marketing efficiency and profit. This shows how powerful ML can be for gaining key insights on the success of marketing campaigns.

