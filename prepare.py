import acquire
import pandas as pd
from sklearn.model_selection import train_test_split


def prep_iris(df=acquire.get_iris_data()):
    df.drop(columns=["species_id", "measurement_id"], inplace=True)
    df.rename(columns={"species_name": "species"}, inplace=True)
    df = pd.concat(
        [df, (pd.get_dummies(df[["species"]], dummy_na=False, drop_first=True))], axis=1
    )
    return df


def prep_titanic(df=acquire.get_titanic_data()):
    # drop class, embarked and alone becuause they are repetive or covered by other vars
    df = df.drop(columns=["passenger_id", "class", "embarked", "alone"])
    # fill age nulls with mean age
    df.age = df.age.fillna(df.age.mean())
    # keep columns with less than 10 nulls
    df = df.loc[:, df.isna().sum() < 10]
    # make dummies and add them
    df = pd.concat(
        [
            df,
            (
                pd.get_dummies(
                    df[["sex", "embark_town"]], dummy_na=False, drop_first=True
                )
            ),
        ],
        axis=1,
    )
    # drop original columns that are encoded
    df = df.drop(columns=["sex", "embark_town"])
    return df


def prep_telco(df=acquire.get_telco_data()):
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


def split_data(df, y_value):
    # split the data set
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[y_value]
    )
    train, validate = train_test_split(
        train, test_size=0.3, random_state=42, stratify=train[y_value]
    )
    # into x and y
    x_train = train.drop(columns=[y_value])
    y_train = train[y_value]
    x_validate = validate.drop(columns=[y_value])
    y_validate = validate[y_value]
    x_test = test.drop(columns=[y_value])
    y_test = test[y_value]
    return x_train, y_train, x_validate, y_validate, x_test, y_test
