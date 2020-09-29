import numpy as np
import pandas as pd

# we should put the vast majority of the data cleaning stuff into a single (big arse) function
# horrible programmming practise, but fuck it, this is Python ...

def clean_data_frame(raw_df, cols_with_cat_nans):
    # get rid of the "£" symbol
    raw_df['AuthLimAmt'].replace('£','',regex=True, inplace=True)
    for column in cols_with_cat_nans:
        raw_df[column].fillna('Unknown', inplace=True)
    raw_df['MerchantCategory'].replace('Unknown',13,regex=True, inplace=True)
    raw_df['MerchantCategory'] = raw_df['MerchantCategory'].astype(int)
    raw_df["AcctID"] = raw_df["AcctID"].astype('category')
    object_dtype_columns = list(raw_df.select_dtypes(include=['object']).columns)
    object_dtype_columns.remove('TransactionDatePosted')
    for f in object_dtype_columns:
        raw_df[f] = raw_df[f].astype('category')
    raw_df['start'] =  pd.to_datetime(raw_df['TransactionDatePosted'])
    # set index from column 'start', so we can select by datetime values
    # raw_df = raw_df.set_index('start')
    return raw_df

# the default start & end values here are from the whole dataframe 
def get_missing_dates(df, field='start', start='2018-12-01', end='2020-02-29'):
    return pd.date_range(start = start, end = end ).difference(df[field])

def getDateSize(_date, _df, field='start'):
    whatever_it_is =  _df[_df[field] == _date]
    return whatever_it_is.size

def getDateData(_date, _df, field='start'):
    whatever_it_is =  _df[_df[field] == _date]
    return whatever_it_is

# this code is, how do i say this?, fucking horrible ...
def latestFieldValue(_df, _field='AccountRunningBalance'):
    thing=_df.tail(1).to_dict()[_field]
    return [v for k, v in thing.items()][0]

def createAggregatedRow(_df, _static_fields):
    # get any row, doesn't matter which ...
    # i'm going to spell this out coz referencing ...
    acct_id = latestFieldValue(_df, _field='AcctID')
    agegroup = latestFieldValue(_df, _field='agegroup')
    auth_lim = latestFieldValue(_df, _field='AuthLim')
    auth_lim_amt = latestFieldValue(_df, _field='AuthLimAmt')
    date = latestFieldValue(_df, 'start')
    
    copied_vals = { 'AcctID': acct_id,
                   'start': date,
                    'agegroup': agegroup,
                    'AuthLim': auth_lim,
                    'AuthLimAmt': auth_lim_amt,
                  }
    # get the last, latest 'AccountRunningBalance'
    running_amount = latestFieldValue(_df)
    copied_vals['AccountRunningBalance'] = running_amount
    # now we need the sum of the transactions for this date
    aggregate_transactions = _df['TransactionAmt'].sum()
    copied_vals['TransactionAmt'] = aggregate_transactions
    return copied_vals

def aggreateDataframe(_df, _date_range, _static_fields):
    list_of_shonaries=[]
    for _date in _date_range:
        # what about size == 1?
        if getDateSize(_date, _df) > 0:
            #print(f"  data for {_date}")
            data = getDateData(_date, _df)
            aggregated_shonary=createAggregatedRow(data, _static_fields)
            list_of_shonaries.append(aggregated_shonary)
        else:
            # now we need to start filling in nthe missing data ...
            previous_dict = list_of_shonaries[-1]
            #print(f"Missing data: previous row is {previous_dict}")
            #running_amt = previous_dict['AccountRunningBalance']
            new_shonary = createMissingDate(_date, previous_dict)
            #print(f"New data: {new_shonary}")
            list_of_shonaries.append(new_shonary)
    
    return list_of_shonaries

# damned ugly ...

def createMissingDate(_date, _previous_row_shonary):
    return {
        'AcctID': _previous_row_shonary['AcctID'],
        'start': _date,
        'agegroup': _previous_row_shonary['agegroup'],
        'AuthLim': _previous_row_shonary['AuthLim'],
        'AuthLimAmt': _previous_row_shonary['AuthLimAmt'],
        'AccountRunningBalance': _previous_row_shonary['AccountRunningBalance'],
        'TransactionAmt': 0.0,
    }

def fillInTheBlanks(df, _account, field='AcctID'):
    
    _df=df[df[field] == _account]
    
    # calculate any missing transactions
    missing_dates = get_missing_dates(_df, start=_df.start.min(), end=_df.start.max())
    
    # get the first date in the dataframe 
    _date = _df.iloc[0].to_dict()['start']
    
    # these are the 'static' categories
    static_fields = ['AcctID', 'agegroup', 'AuthLim', 'AuthLimAmt']
    
    # the entire date range for this account
    r = pd.date_range(start=_df.start.min(), end=_df.start.max()) 
    
    # this creates a list of dictionaries
    shonary_list = aggreateDataframe(_df, r, static_fields)
    
    #return _df[_df[field] == _account]
    
    return pd.DataFrame(shonary_list)

