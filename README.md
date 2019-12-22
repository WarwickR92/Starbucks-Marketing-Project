# Starbucks Marketing Project

The aim of this project is to use data collected from past marketing campagins ran by Starbucks in order to understand how they have influneced user behaviour and use that information to predict how a campagins may perform in the future. The bussiness case for this is that having better knowledge of which users will be more influenced by which campaigns will ultimatly help improve efficentcies in starbucks marketing. It will also potentially allow Starbucks to model ahead how well marketing will perform allowing them to adjust stratergies increaing profits and improving user experience.

# Performing the Analysis

The layout of this project will follow the priciples of the CRISP-DM framework starting with the buissness understanding as outlined above and then moving through data understanding, data prepartaion, modelling and finally evaluation. 

There are a number of different ways the project could have been undertaken but after the intital data understanding stage tt was decided to split the project into two core parts:

- The first involved trying to understand the main demographic groups that use Starbucks on a regular basis. This will help the understanding of which users are more easily influenced by the marketing campagins and therefore help the pediction modelling later on. It was decided to do this using the K-means unsupervised clustering method.

- The second stage would be to build a model on the back of these demographics that could potentially predict how user spend would change under the influence of certain offers. This stage will involve using regression models using xgboost in order to try to predict daily user spend. 


### Data Sources:
The following data sources have been provided by Starbucks to undertake this analysis:
- portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
- profile.json - demographic data for each customer
- transcript.json - records for transactions, offers received, offers viewed, and offers completed
