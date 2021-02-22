import pandas as pd
import cudf
from sklearn.preprocessing import LabelEncoder


def encoder_train_test(train, test, method="LabelEncoder", prints=True):
    '''Encode object columns in Train and Test, leaving NAs to be imputed later.
    Uses Label Encoder from Sklearn or Dummy Encoding from Pandas.
    Train and Test need to be Pandas dataframes.
    Returns the train and test dataframes with encoded column + a list of 
    the columns what have been encoded.'''
    # Convert to CPU dataframe
    if isinstance(train, cudf.core.dataframe.DataFrame):
        train = train.to_pandas()
        test = test.to_pandas()

    # Select all columns with type == "object"
    train_cols = [col for col in train.columns if train[col].dtype == 'object']
    test_cols = [col for col in test.columns if test[col].dtype == 'object']
    string_cols = list(set(train_cols + test_cols))
    if prints:
        print("Train: {} columns to encode.".format(len(train_cols)), "\n"
              "Test: {} columns to encode.".format(len(test_cols)), "\n")

    # --- Label Encoder ---
    if method == "LabelEncoder":

        for df, cols in zip([train, test], [train_cols, test_cols]):
            for col in cols:
                encoder = LabelEncoder()

                # select all values to encode but NAs (we'll impute these later)
                fit_by = pd.Series([i for i in df[col].unique()
                                    if type(i) == str])
                encoder.fit(fit_by)
                # encode the column, leaving NAs untouched
                df[col] = df[col].apply(lambda x: encoder.transform([x])[0]
                                        if type(x) == str else x)

    # --- Dummy Encoder ---
    if method == "Dummy":

        # Create Dummy Variables
        encoded_train = pd.get_dummies(train[train_cols])
        encoded_test = pd.get_dummies(test[test_cols])

        # Strip columns of leading/trailing spaces
        encoded_train.columns = encoded_train.columns.str.strip()
        encoded_test.columns = encoded_test.columns.str.strip()
        # Erase train or test columns that might be found in the other df
        # this can and it happens: there are categories in train data
        # that aren't found in the test data
        # nevertheless, they need to be found in both (or not)
        in_train_not_test = list(
            set(encoded_train.columns) - set(encoded_test.columns))
        in_test_not_train = list(
            set(encoded_test.columns) - set(encoded_train.columns))

        # Drop old columns and replace with encoded ones
        train.drop(columns=train_cols, axis=1, inplace=True)
        test.drop(columns=test_cols, axis=1, inplace=True)

        train = pd.concat([train, encoded_train], axis=1)
        test = pd.concat([test, encoded_test], axis=1)

        if in_train_not_test:
            train.drop(columns=in_train_not_test, axis=1, inplace=True)

        if in_test_not_train:
            test.drop(columns=in_test_not_train, axis=1, inplace=True)

        # Get all categ columns again (updated with latest changes)
        if in_train_not_test:
            new_train_cols = list(
                set(encoded_train.columns) - set(in_train_not_test))
        else:
            new_train_cols = list(encoded_train.columns)
        if in_test_not_train:
            new_test_cols = list(
                set(encoded_test.columns) - set(in_test_not_train))
        else:
            new_test_cols = list(encoded_test.columns)

        string_cols = list(set(new_train_cols + new_test_cols))

    # Convert back to GPU dataframe
    train = cudf.DataFrame.from_pandas(train)
    test = cudf.DataFrame.from_pandas(test)

    print("Encoding finished.")
    # Categ columns are needed to be properly shifted to int32 later
    return train, test, string_cols


def inspect_missing_data(df, treshold=0.5):
    '''Insect missing patterns from data to assess next steps.'''

    missing_data = df.isna().sum().reset_index().sort_values(by=0, ascending=False)
    no_missing = missing_data[missing_data[0] != 0].shape[0]
    total_cols = df.shape[1]
    total_rows = df.shape[0]

    missing_data.columns = ["name", "missing appearences"]
    missing_data["%missing from total"] = missing_data[
        missing_data["missing appearences"] != 0]["missing appearences"] / total_rows

    too_much_miss = missing_data[
        missing_data["%missing from total"] > treshold].shape[0]
    to_drop = missing_data[
        missing_data["%missing from total"] > treshold]["name"].to_list()

    print("There are {}/{} columns with missing data.".format(
        no_missing, total_cols))
    print("There are {}/{} columns with more than {}% missing data (these columns will be dropped)".format(
        too_much_miss, no_missing, treshold * 100))

    return missing_data, to_drop


def drop_missing_train_test(train, test, treshold=0.5):
    print("--- {} ---".format("Train"))
    _, to_drop_train = inspect_missing_data(train, treshold)
    print("--- {} ---".format("Test"))
    _, to_drop_test = inspect_missing_data(test, treshold)

    diff_test = set(to_drop_test) - set(to_drop_train)
    print("! {} has more than {}% missingness in test, but not in train; nevetheless, we'll drop it in both.".format(
        diff_test, treshold * 100))

    # Drop columns with more than threshold% missingness
    train.drop(labels=to_drop_test, axis=1, inplace=True)
    test.drop(labels=to_drop_test, axis=1, inplace=True)

    return train, test


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=10):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).reset_index().sort_values(0, ascending=False).reset_index(drop=True)
    au_corr.rename({0: "corr_abs"}, axis=1, inplace=True)
    au_corr
    return au_corr[0:n]
