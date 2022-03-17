from env import get_db_url
import pandas as pd
import os


def get_zillow_data(use_cache=True):
    """pull from SQL unless zillow.csv exists"""
    filename = "zillow.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading from csv...")
        return pd.read_csv(filename)

    print("reading from sql...")
    url = get_db_url("zillow")
    query = """
    SELECT *
    FROM properties_2017
    """
    df = pd.read_sql(query, url)

    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df


def prep_zillow(use_cache=True):
    """pull from full zillow.csv unless prepped_zillow.csv exists"""
    filename = "prepped_zillow.csv"
    if os.path.isfile(filename) and use_cache:
        print("Prepped csv exist, pulling data...")
        return pd.read_csv(filename)

    print("preparing data from get_zillow_data()")
    df = get_zillow_data()
    # select single family residential (value == 261)
    df = df[df.propertylandusetypeid == 261]
    # keep the columns I want: bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, and fips
    columns = [
        "bedroomcnt",
        "bathroomcnt",
        "calculatedfinishedsquarefeet",
        "taxvaluedollarcnt",
        "yearbuilt",
        "taxamount",
        "fips",
    ]
    df = df[columns]
    # drop the nulls, its a small subset of data as
    df = df.dropna()
    print("Saving prepped data to csv in local directory...")
    df.to_csv(filename, index=False)
    return df
