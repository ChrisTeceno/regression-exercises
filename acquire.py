from env import get_db_url
import pandas as pd
import os


def get_titanic_data(use_cache=True):
    """pull from SQL unless titanic.csv exists"""
    filename = "titanic.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading from csv...")
        return pd.read_csv(filename)

    print("reading from sql...")
    url = get_db_url("titanic_db")
    query = """
    SELECT *
    FROM passengers
    """
    df = pd.read_sql(query, url)

    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df


def get_iris_data(use_cache=True):
    """pull from SQL unless iris.csv exists"""
    filename = "iris.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading from csv...")
        return pd.read_csv(filename)

    print("reading from sql...")
    url = get_db_url("iris_db")
    query = """
    SELECT *
    FROM measurements
    JOIN species USING(species_id)
    """
    df = pd.read_sql(query, url)

    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df


def get_telco_data(use_cache=True):
    """pull from SQL unless telco.csv exists"""
    filename = "telco.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading from csv...")
        return pd.read_csv(filename)

    print("reading from sql...")
    url = get_db_url("telco_churn")
    query = """
    SELECT *
    FROM customers
    JOIN contract_types USING(contract_type_id)
    JOIN internet_service_types USING(internet_service_type_id)
    JOIN payment_types USING(payment_type_id)
    """
    df = pd.read_sql(query, url)

    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df


def get_attendance_data(use_cache=True):
    """pull from SQL unless attendance.csv exists"""
    filename = "attendance.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading from csv...")
        return pd.read_csv(filename)

    print("reading from sql...")
    url = get_db_url("tidy_data")
    query = """
    SELECT *
    FROM attendance
    """
    df = pd.read_sql(query, url)

    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df


def get_coffee_levels_data(use_cache=True):
    """pull from SQL unless coffee_levels
    .csv exists"""
    filename = "coffee_levels.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading from csv...")
        return pd.read_csv(filename)

    print("reading from sql...")
    url = get_db_url("tidy_data")
    query = """
    SELECT *
    FROM coffee_levels

    """
    df = pd.read_sql(query, url)

    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df


def get_cake_recipes_data(use_cache=True):
    """pull from SQL unless cake_recipes.csv exists"""
    filename = "cake_recipes.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading from csv...")
        return pd.read_csv(filename)

    print("reading from sql...")
    url = get_db_url("tidy_data")
    query = """
    SELECT *
    FROM cake_recipes
    """
    df = pd.read_sql(query, url)

    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df
