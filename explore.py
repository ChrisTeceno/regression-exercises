import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")
from env import get_db_url
import pandas as pd
import os


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


def prep_telco(df=get_telco_data()):
    # make autopay column where payment type id 1,2 is 0(no autopay)
    # payment type id and 3,4 is 1(autopay)
    df["autopay"] = (df.payment_type_id > 2).astype(int)
    # drop repetitive columns
    df = df.drop(
        columns=[
            "payment_type_id",
            "internet_service_type_id",
            "contract_type_id",
            "customer_id",
        ]
    )
    # remove all spaces from strings
    df["total_charges"] = df["total_charges"].str.strip()
    # remove all empty values
    df = df[df.total_charges != ""]
    # convert to float
    df.total_charges = df.total_charges.astype(float)
    # replace yes and female with 1, and no's with 0
    df = df.replace(
        ["No", "Yes", "Female", "Male", "No internet service", "No phone service"],
        [0, 1, 1, 0, 0, 0],
    )
    # replace gender with is_female to correlate with above change
    df.rename(columns={"gender": "is_female"}, inplace=True)
    # make dummies and add them
    df = pd.concat(
        [
            df,
            (
                pd.get_dummies(
                    df[["contract_type", "internet_service_type", "payment_type"]],
                    dummy_na=False,
                    drop_first=False,  # i did not drop first to make it more human readable
                )
            ),
        ],
        axis=1,
    )
    # drop the columns that have been converted to dummies
    df = df.drop(columns=["contract_type", "internet_service_type", "payment_type"])
    return df


# function to split, will copy to explore.py when done
def split_continuous(df, y_value):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, validate = train_test_split(train, test_size=0.3, random_state=42)
    # into x and y
    x_train = train.drop(columns=[y_value])
    y_train = train[y_value]
    x_validate = validate.drop(columns=[y_value])
    y_validate = validate[y_value]
    x_test = test.drop(columns=[y_value])
    y_test = test[y_value]
    return (
        x_train,
        y_train,
        x_validate,
        y_validate,
        x_test,
        y_test,
        train,
        test,
        validate,
    )


# take in df and list of columns to make pair plot
def plot_variable_pairs(df, columns):
    # pair plot with reg line
    sns.pairplot(
        df[columns], corner=True, kind="reg", plot_kws={"line_kws": {"color": "blue"}}
    )
    plt.show()


def months_to_years(df):
    # add tenure_years columns and set it to nearest whole year, converted to int
    df["tenure_years"] = (df["tenure"] / 12).round().astype(int)
    return df


def plot_categorical_and_continuous_vars(df, columns):
    sns.jointplot(kind="reg", data=df[columns], x=columns[0], y=columns[1])
    sns.jointplot(kind="reg", data=df[columns], x=columns[0], y=columns[2])
    sns.jointplot(kind="reg", data=df[columns], x=columns[0], y=columns[3])


def get_mall_data(use_cache=True):
    """pull from SQL unless mall_customers.csv exists"""
    filename = "mall_customers.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading from csv...")
        return pd.read_csv(filename)

    print("reading from sql...")
    url = get_db_url("mall_customers")
    query = """
    SELECT *
    FROM customers
    """
    df = pd.read_sql(query, url)

    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df
