# Starbucks Marketing Project

The aim of this project is to use data provided by Starbucks in order to understand how each marketing campagin performed on different consumer demographics and use that information to predict how a campagins may perform in the future. The bussiness case for this is that having better knowledge of which users will be more influenced by which campaigns will ultimatly help improve efficentcies in Starbucks marketing strategy. It will also potentially allow Starbucks to model ahead how well marketing will perform so that they can adjust these stratergies when needed.

## Background



## Performing the Analysis

The layout of this project will follow the priciples of the CRISP-DM framework starting with the buissness understanding as outlined above and then moving through data understanding, data prepartaion, modelling and finally evaluation. 

There are a number of different ways the project could have been undertaken but after the intital data understanding stage it was decided to split the project into two core parts:

- The first involved trying to understand the main demographic groups that use Starbucks on a regular basis. This will help the understanding of which users are more easily influenced by the marketing campagins and therefore help the pediction modelling later on. It was decided to do this using the K-means unsupervised clustering method as it is a quick and easily applied unsupervised learning approach.

- The second stage would is to build a model on the back of these demographics that can predict how user spend would change under the influence of the offers Starbucks has live. This stage will involve using regression models using xgboost in order to try to predict daily user spend. This model was choosen as it high accuracy and low computation time but it has many Hyper Parameters to tune. In order to speed up the tuning of these parameters I decided to use xgboosts native API for testing as it's quicker and more flexible than the SKlearn equivalent. This model will be judged by looking at the Root Mean Squared Error (RMSE), this performance metric is useful when 

A thrid round of modeling was initially planned to predict the likelihood of a user making a transaction on any given day. However whilst performing the cleaning & formatting phase of the process it became apparent that this behaviour would be very difficult from given data. In order to potentially perfrom this analysis in future a longer time period would be needed without any marketing running to estabish a baseline rate of normal transactions for users in each demogaphic and judge user behaviour.


## Data Sources:
The following data sources have been provided by Starbucks to undertake this analysis:
- portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
- profile.json - demographic data for each customer
- transcript.json - records for transactions, offers received, offers viewed, and offers completed


