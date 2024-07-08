import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


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


def get_input_value(param_name, param_type):
    while True:
        user_input = input(f"Enter value for {param_name} ({param_type}): ")
        try:
            if param_type == 'bool':
                if user_input.lower() in ['true', 'false']:
                    return user_input.lower() == 'true'
                else:
                    raise ValueError("Input must be 'true' or 'false'")
            elif param_type == 'int':
                return int(user_input)
            elif param_type == 'float':
                return float(user_input)
            else:
                print(f"Unsupported parameter type: {param_type}")
                return None
        except ValueError as e:
            print(f"Invalid input: {e}")


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

    target_col = select_column(df, 'all')
    X = df[target_col.name]
    df.drop(columns=[target_col.name], inplace=True)

    test_size = '-1'
    while True:
        print('What test size do you want? (Enter a number between 0 and 0.9)')
        test_size = input()
        try:
            test_size_float = float(test_size)
            if 0 <= test_size_float <= 0.9:
                break
            else:
                print("The number must be between 0 and 0.9.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    test_size = float(test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, df, test_size=test_size)

    model = None
    if target_col.dtype == object:
        # GIVE CLASSIFICATION MODELS
        pass
    else:
        while True:
            model_options = ['Linear Regressor', 'RandomForestRegressor']
            models = [LinearRegression, RandomForestRegressor]
            print('Which model would you like to use?')
            option = print_options(model_options)
            model = models[option]()
            params = model.get_params()

            for param in params:
                param_value = params[param]
                if isinstance(param_value, bool):
                    param_type = 'bool'
                elif param_value is None or isinstance(param_value, int):
                    param_type = 'int'
                elif isinstance(param_value, float):
                    param_type = 'float'
                else:
                    print(f"Unsupported parameter type for {param}: {type(param_value)}")
                    continue
                params[param] = get_input_value(param, param_type)

            model.set_params(**params)
            while True:
                model.fit(X_train, y_train)
                model_predictions = model.predict(X_test)
                metric = mean_squared_error(y_test, model_predictions)
                metric2 = r2_score(y_test, model_predictions)
                print("Mean Squared Error: ", metric)
                print("R2 Score: ", metric2)
                break

            options = ['Yes', 'No']
            print("Try again with different hyperparameters?")
            option = print_options(options)
            quit()


if __name__ == '__main__':
    main()