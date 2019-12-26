"""
---
Contains all of the useful functions needed to clean and process the Starbucks data ready for modeling
---
|_
|_ 
|_ 

---
"""
# import general functions
import pandas as pd
import numpy as np
import json

# modules for the modeling
import joblib
from sklearn.preprocessing import StandardScaler

def clean_transcript_data(data):
    """
    this process cleans the values column and formats the transcript data
    """
    # creates a column for the type of interaction   
    data['interaction_value'] = [list(x.keys())[0] for x in data['value']]
    
    # creates a column related to the value amount or id    
    data['id'] = [list(x.values())[0] for x in data['value']]
    
    # drops the value column
    data = data.drop(columns=['value'])
    
    # cleans the interaction type column so offer id is consistent
    data['interaction_value'] = [x.replace('offer id','offer_id') for x in data['interaction_value']]
    
    # split out interaction_type
    temp_df = pd.get_dummies(data['interaction_value'])

    # combine the dataframes
    data = pd.concat([temp_df, data], axis=1, sort=True)
    
    # split out event
    temp_df = pd.get_dummies(data['event'])

    # combine the dataframes
    data = pd.concat([temp_df, data], axis=1, sort=True)

    # drop the original columns
    data = data.drop(columns=['interaction_value','event'])    
    
    return data # returns the clean transcript data


def clean_profile_data(data):
    """
    this process clean age, income and became_member_on columns in the profile data
    """
    # rename the column 'id' to person
    data.columns = ['age','member joined','gender','person' ,'income']
    
    # replace 118 in the age column with a zero indicating no age 
    # it might be worth looking at this a seperate group of users later on
    data['age'] = data['age'].replace(118,0)

    # update the became_member_on column to a datetime format
    data['member joined'] = pd.to_datetime(data['member joined'], format='%Y%m%d')
    
    # replace the NaN's in the income
    data['income'] = data['income'].fillna(0)
    
    # replace M, F, O and None types to get the 4 groups of customers
    data['gender'] = data['gender'].replace('M','male')
    data['gender'] = data['gender'].replace('F','female')
    data['gender'] = data['gender'].replace('O','other')
    data['gender'] = data['gender'].fillna('unknown gender')

    # split the column into seperate columns
    temp_df = pd.get_dummies(data['gender'])

    # combine the dataframes
    data = pd.concat([temp_df, data], axis=1, sort=True)

    # drop the original column
    data = data.drop(columns=['gender'])

    return data


def clean_portfolio_data(data):
    """
    this process has been created to clean columns in the profile data
    """
    # splits the channels column into seperate columns
    # creates temporary dataframes and lists  
    temp_df = pd.DataFrame(columns=['web', 'email', 'mobile','social'])
    temp_list = []

    # loop through the rows and attach the values to a dic   
    for index, row in data.iterrows():
        for value in row['channels']:
             temp_list.append({'index': index, 'value':value})

    # change the list into dataframe
    temp_df = temp_df.append(temp_list, ignore_index=False, sort=True)
    temp_df = temp_df.groupby('index')['value'].value_counts()
    temp_df = temp_df.unstack(level=-1).fillna(0)
    
    # combine the dataframes
    data = pd.concat([temp_df, data], axis=1, sort=True)
    
    # split the column into seperate columns
    temp_df = pd.get_dummies(data['offer_type'])

    # combine the dataframes
    data = pd.concat([temp_df, data], axis=1, sort=True)

    # drop the original columns
    data = data.drop(columns=['offer_type','channels'])
    
    return data


def transactions(data):
    """
    returns all the transactions from the transcript dataframe
    """
    transactions_df = data[data['transaction'] == 1]
    transactions_df = transactions_df[['person','time','id']]
    transactions_df.columns = ['person','transaction_time','spend']
    
    return transactions_df


def offers(transcript_data, portfolio_data):
    """
    returns all of the offers that were received/viewed/completed combined with portfolio data
    """
    # keep only the recived offers
    received_offer = transcript_data[transcript_data['offer received'] == 1]
    received_offer = received_offer[['offer received','person', 'time', 'id']]
    received_offer.columns = ['offer received','person', 'time_received', 'id_offer']    
    
    # keep only the viewed offers
    veiwed_offer = transcript_data[transcript_data['offer viewed'] == 1]
    veiwed_offer = veiwed_offer[['offer viewed','person', 'time', 'id']]
    veiwed_offer.columns = ['offer viewed','person', 'time_viewed', 'id_offer']
    
    # keep all the offers completed data as informational campaigns don't have a completed flag
    completed_offer = transcript_data
    completed_offer = completed_offer[['offer completed','person', 'time', 'id']]
    completed_offer.columns = ['offer completed','person', 'time_completed', 'id_offer']
    
    # merge the offers data into one dataframe based on id and person
    merged_veiws = received_offer.merge(veiwed_offer, on=['person','id_offer']) 
    merged_completed = merged_veiws.merge(completed_offer, on=['person','id_offer']) 
    
    # drop anywhere the offer was recived after being viewed 
    # (not useful as it suggests it was a different offer)
    merged_completed = merged_completed[merged_completed['time_viewed'] > 
                                        merged_completed['time_received']]
    
    # merges all of the offer data with info in the portfolio data
    portfolio_data = portfolio_data.rename(columns = {'id':'id_offer'})
    offers = merged_completed.merge(portfolio_data, on=['id_offer'])
    
    # change duration time to hours
    offers['duration'] = offers['duration']*24
    
    return offers


def influenced_bogo(transcript_data, portfolio_data):
    """
    this function has been created to keep only BOGO offers that influenced a purchase
    """
    # gets all of the offers that were received/viewed/completed formatted together
    offer_data = offers(transcript_data, portfolio_data)
    
    # select only the bogo offers and have been completed
    bogo_offers = offer_data[(offer_data['bogo'] == 1) & 
                             (offer_data['offer completed'] == 1)]
    
    # removes any that were completed prior to being viewed
    bogo_offers = bogo_offers[bogo_offers['time_completed'] >= 
                              bogo_offers['time_viewed']]
    
    # removes offers that were completed outside of the offer timeframe (indicating it was a second offer)
    bogo_offers = bogo_offers[(bogo_offers['duration'] >= (bogo_offers['time_completed'] - 
                                                           bogo_offers['time_received']))
                             ]

    # creates the transaction data
    transactions_data = transactions(transcript_data)
    
    # merge the offers and transactions
    transactions_bogo = transactions_data.merge(bogo_offers, on=['person'])
    
    # filter the tansactions keeping ones that occured at same time as the offer was complete
    transactions_bogo = transactions_bogo[transactions_bogo['transaction_time'] == 
                                          transactions_bogo['time_completed']]
    
    # remove any repeat transactions
    transactions_bogo = transactions_bogo.drop_duplicates(subset=['person','transaction_time','spend'], keep="first")
    
    return transactions_bogo


def influenced_discount(transcript_data, portfolio_data):
    """
    this function has been created to keep only discount offers that influenced a purchase
    """
    # gets all of the offers that were received/viewed/completed formatted together
    offer_data = offers(transcript_data, portfolio_data)
    
    # select only the discuont offers and have been completed
    discount_offers = offer_data[(offer_data['discount'] == 1) & 
                                 (offer_data['offer completed'] == 1)]
    
    # removes any that were completed prior to being viewed
    discount_offers = discount_offers[discount_offers['time_completed'] >= 
                                      discount_offers['time_viewed']]
    
    # removes offers that were completed outside of the timeframe (indicating it was a second offer)
    discount_offers = discount_offers[discount_offers['duration'] >= (discount_offers['time_completed'] - 
                                                                      discount_offers['time_received'])]

    # creates the transaction data
    transactions_data = transactions(transcript_data)
    
    # merge the offers and transactions
    transactions_discount = transactions_data.merge(discount_offers, on=['person'])
    
    # filter the tansactions keeping the ones after the offer was viewed but before it was completed
    transactions_discount = transactions_discount[(transactions_discount['transaction_time'] >= transactions_discount['time_viewed']) &
                                                 (transactions_discount['transaction_time'] <= transactions_discount['time_completed'])]
    
    # remove any repeat transactions
    transactions_discount = transactions_discount.drop_duplicates(subset=['person','transaction_time','spend'], keep="first")
    
    return transactions_discount


def influenced_informational(transcript_data, portfolio_data):
    """
    this function has been created to keep only informational offers that influenced a purchase
    """
    # gets all of the offers that were received/viewed/completed formatted together
    offer_data = offers(transcript_data, portfolio_data)
    
    # select only the informational offers
    info_offers = offer_data[(offer_data['informational'] == 1)]

    # creates the transaction data
    transactions_data = transactions(transcript_data)
    
    # merge the offers and transactions
    transactions_info = transactions_data.merge(info_offers, on=['person'])
    
    # filter the tansactions keeping the ones after the offer was viewed
    transactions_info = transactions_info[(transactions_info['transaction_time'] >= transactions_info['time_viewed'])]
    
    # removes transactions that happened outside of duration timeframe of the offer
    transactions_info = transactions_info[transactions_info['duration'] >= (transactions_info['transaction_time'] - 
                                                                            transactions_info['time_viewed'])]
    
    # remove any repeat transactions
    transactions_info = transactions_info.drop_duplicates(subset=['person','transaction_time','spend'], keep="first")
    
    return transactions_info


def norm_transactions(clean_trans_df, clean_port_df):
    """
    produces all the transactions that weren't influenced by offers
    """
    # creates the transaction data
    transactions_data = transactions(clean_trans_df)
    
    # all offer affected transactions
    inf_discount = influenced_discount(clean_trans_df, clean_port_df)
    inf_bogo = influenced_bogo(clean_trans_df, clean_port_df)
    inf_informational = influenced_informational(clean_trans_df, clean_port_df)
    
    # combine all the influenced transcations    
    inf_trans = inf_informational.append(inf_discount.append(inf_bogo))
    
    # drop to have the same columns as all transactions    
    inf_trans = inf_trans[['person', 'transaction_time', 'spend']]
    
    # remove offer related transactions
    norm_trans = pd.concat([transactions_data, inf_trans]).drop_duplicates(keep=False)
    
    return norm_trans


def user_transactions(profile, transactions):
    """
    this creates useful information of individual users transactions
    """
    # list of consumers in the transaction data
    consumers = transactions.groupby('person').sum().index

    # calculate the total transaction values for a consumer
    consumer_spend = transactions.groupby('person')['spend'].sum().values

    # calculate the number of transactions per consumer
    consumer_trans = transactions.groupby('person')['spend'].count().values

    # create a dataframe with spend info per consumer
    consumer_data = pd.DataFrame(consumer_trans, index=consumers, columns=['total transactions'])

    # add the total transaction column
    consumer_data['total spend'] = consumer_spend 
    
    # average spend per transaction    
    consumer_data['spend per trans'] = consumer_data['total spend']/consumer_data['total transactions']
    
    # average spend per day
    consumer_data['spend per day'] = consumer_data['total spend']/30
    
    # combine profile and transaction data
    consumer_profile = profile.merge(consumer_data, on=['person']).fillna(0)
    
    # I will take the last date the final day data has been collected
    final_date = consumer_profile['member joined'].max()
    
    # membership length in weeks
    consumer_profile['membership length'] = [round((final_date - x).days / 7,0) for x in consumer_profile['member joined']]

    return consumer_profile


def spend_per_day(clean_trans_df, clean_port_df):
    """
    this creates the spend per day by person which will be used for the regression analysis
    """
    # all offer affected transactions
    inf_discount = influenced_discount(clean_trans_df, clean_port_df)
    inf_bogo = influenced_bogo(clean_trans_df, clean_port_df)
    inf_informational = influenced_informational(clean_trans_df, clean_port_df)

    # combine all the influenced transcations    
    inf_trans = inf_informational.append(inf_discount.append(inf_bogo))

    # keep only the columns needed
    inf_trans = inf_trans[['person', 'transaction_time', 'spend', 'id_offer']]

    # creates dummies for each type of offer that was avalible
    inf_off = pd.get_dummies(inf_trans['id_offer'])

    # concates the offers with the transactions
    inf_trans = pd.concat([inf_trans, inf_off], axis=1).drop(columns=['id_offer'])

    # changes the transaction time to a day
    inf_trans['transaction_time'] = np.ceil(inf_trans['transaction_time']/24)

    # groupby the person and transaction_time 
    influenced = inf_trans.groupby(['person','transaction_time']).sum()
    
    # unstack and restack in index to fill days with zeros   
    influenced = influenced.unstack().fillna(0).stack()
    
    # create the same file for all other transactions to get spend   
    trans_up = transactions(clean_trans_df)

    # changes the transaction time to a day
    trans_up['transaction_time'] = np.ceil(trans_up['transaction_time']/24)

    # group all of the transaction
    trans_up = trans_up.groupby(['person','transaction_time']).sum()
    
    # fill any empty days with zeros 
    trans_up = trans_up.unstack().fillna(0).stack()
    
    # megre the files to have spend by day and if they were influenced by any offers
    spend_per_day = trans_up.merge(influenced, right_index=True, left_index=True) 
    
    return spend_per_day

def predict_demographic(profile_data, demographic_model='kmeans_demographic_model.pkl'):
    """
    this can be used to predict the demographics of group of consumers
    """
    # Reads the volume model 
    final_kmeans = joblib.load(demographic_model)
    
    # remove unwanted columns
    profile_data_input = profile_data.drop(columns=['member joined',
                                              'person', 
                                              'total transactions', 
                                              'total spend'])
    
    # process the profile data
    scaler = StandardScaler()
    scaler.fit(profile_data_input)
    input_demo_data = scaler.transform(profile_data_input)
    
    # predict the demographics   
    predictions = final_kmeans.predict(input_demo_data)
    
    # add the pedictions for the clustering onto the original dataset
    updated_dataframe = profile_data
    updated_dataframe['demographic'] = predictions
    
    return updated_dataframe

# create a function to use each of the above models to predict spend
def predict_spend(input_data, model_demographic):
    """
    this function predicts the spend of users based on the model made for there demographic
    """    
    # load in the model needed to predict spend on the analysis
    demo_model = joblib.load(f"xgboost_price_model_{model_demographic}.pkl")
    
    # keep only the demographic data related to the model to be used in this section
    input_data = input_data[input_data['demographic'] == model_demographic]

    # keep only the columns needed
    input_data = input_data.drop(columns=['transaction_time','person',
                                          'spend','demographic'])
    
    # calculate the prediction based on the input date
    prediction = demo_model.predict(input_data)
    
    # attach the prediction to the original filtered df
    input_data['prediction'] = prediction
    
    return input_data

def create_dummy_days(data):
    """
    this module creates dummies depending on the day of the week
    (this is useful as users behaviour will be different on a weekday vs weekend)
    """
    # add dummies for the days of the week 
    day_of_week = pd.DataFrame(list(data['transaction_time']))
    for n in [1,2,3,4,5,6,7]:
        day_of_week = day_of_week.replace((n+7), n).replace((n+7*2), n).replace((n+7*3), n).replace((n+7*4), n)   
    day_of_week = pd.DataFrame([str(x) for x in day_of_week.iloc[:,0]])
    input_data_test = pd.concat([data, pd.get_dummies(day_of_week)], axis=1, join='inner')
    
    return input_data_test