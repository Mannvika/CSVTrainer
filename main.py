import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

encoded_columns = set()
normalized_columns = set()

def select_column(df, dtype):
    if dtype == 'all':
        cols = df.columns
    else:
        cols = df.select_dtypes(include=dtype).columns
    if len(cols) == 0:
        print('No columns to perform this operation')
        return -1

    col_map = {}
    for i, col in enumerate(cols):
        col_map[i] = col

    selection = '-1'
    while not selection.isnumeric() or int(selection) > len(cols) or int(selection) < 0:
        print('Select a column to perform an operation on: ')

        for i, col in enumerate(cols):
            print(f"{i}. {col}")

        selection = input()

    target_col = df[col_map[int(selection)]]
    print('\nSelected Column:')
    print(target_col)
    return target_col


def normalize(selected_col, df):

    if selected_col.name in normalized_columns:
        print('Column already normalized.')
        return

    print('Overview of pre-normalized Data: ')
    print(df[selected_col.name])

    options = ['StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'Cancel']
    scalers = [StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler]

    option = print_options(options)
    if option == len(options) - 1:
        return
    else:
        scaler = scalers[option]()

    df[selected_col.name] = scaler.fit_transform(df[[selected_col.name]])
    normalized_columns.add(selected_col.name)


def impute_missing_data(selected_col, df):
    if df[df[selected_col.name].isna()].empty:
        print('No missing data for selected column.')
        return

    print('Overview of Missing Data: ')

    print(df[df[selected_col.name].isna()])
    options = ['Most Frequent', 'Delete Rows', 'Cancel']
    option = print_options(options)

    if option == len(options) - 1:
        return
    if option == len(options) - 2:
        df.dropna(subset=[selected_col.name], inplace=True)
        return
    if option == 0:
        imputer = SimpleImputer(strategy='most_frequent')

    df[selected_col.name] = imputer.fit_transform(df[[selected_col.name]])
    print(df[selected_col.name])


def encode(selected_col, df):
    if selected_col.name not in encoded_columns:
        encoder = LabelEncoder()
        df[selected_col.name] = encoder.fit_transform(selected_col)
        encoded_columns.add(selected_col.name)
    else:
        print('Column already encoded.')


def drop_column(selected_col, df):
    df.drop(columns=[selected_col.name], inplace=True)


def print_options(options):
    selection = '-1'
    while not selection.isnumeric() or int(selection) > len(options) or int(selection) < 0:
        print('Please select an operation to perform: ')
        for i, option in enumerate(options):
            print(f"{i}. {option}")

        selection = input()
    return int(selection)


def print_heatmap(df):
    # Extract numerical columns
    numerical = df.select_dtypes(include=['number'])

    # Calculate the correlation matrix
    corr = numerical.corr()

    plt.figure(figsize=(10, 5))

    # Plot the heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()


def main():
    # Display all features
    pd.set_option('display.max_columns', None)

    print('Welcome to the CSV Trainer!\n')
    print('Enter the file path of the CSV you wish to train:')

    # Load tabular data based on filepath
    filepath = input()
    df = pd.read_csv(filepath)

    # Display first few data
    print('\nYour Data: ')
    print(df.head(), '\n')

    # Preprocessing loop:
    preprocessing_options = ['Impute Data (Categorical)', 'Encode Data (Categorical)', 'View Heatmap (Numerical)', 'Normalize (Numerical)',
               'Drop Feature (Both)', 'View Data', 'Complete Feature Engineering', 'Leave']

    preprocessing_functions = [impute_missing_data, encode, print_heatmap, normalize, drop_column]
    while True:
        option = print_options(preprocessing_options)
        if option == len(preprocessing_options) - 1:
            exit()
        if option == len(preprocessing_options) - 2:
            break
        if option == len(preprocessing_options) - 3:
            print(df)
        if option == 0 or option == 1:
            selected_col = select_column(df, ['object'])
            preprocessing_functions[option](selected_col, df)
        if option == 2:
            preprocessing_functions[option](df)
        if option == 3:
            selected_col = select_column(df, 'number')
            preprocessing_functions[option](selected_col, df)
        if option == 4:
            selected_col = select_column(df, 'all')
            preprocessing_functions[option](selected_col, df)

    # First -> Pick Target Variable
    target_col = select_column(df, 'all')

    if target_col.dtype == object:
        # GIVE CLASSIFICATION MODELS
        pass
    else:
        model_options = []
        pass


    # Model Training Loop:
    while True:
        # Second -> Pick Model
        # Third -> Pick Hyperparameters
        # Fourth -> Display Accuracy Metrics with Chosen Hyper parameters
        # Fifth -> Loop and tune hyperparameters until satisfied
        # Sixth -> Save Model
        pass


if __name__ == '__main__':
    main()