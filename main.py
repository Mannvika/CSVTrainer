import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

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

    df[selected_col.name] = imputer.fit_transform(df[[selected_col.name]]).ravel()
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


def load_data(filepath):
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    elif filepath.endswith('.json'):
        df = pd.read_json(filepath)
    elif filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    elif filepath.endswith('.h5'):
        df = pd.read_hdf(filepath)
    elif filepath.endswith('.feather'):
        df = pd.read_feather(filepath)
    else:
        raise ValueError("Unsupported file format")
    return df


def create_and_train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def main():
    # Display all features
    pd.set_option('display.max_columns', None)

    print('Welcome to the CSV Trainer!\n')
    print('Enter the file path of the CSV you wish to train:')

    # Load tabular data based on filepath
    filepath = input()
    df = load_data(filepath)

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
            if len(df.select_dtypes(include=['object', 'category']).columns) > 0:
                print('Warning: You still have categorical features.')
                print('You can either drop all categorical features or go back and encode/drop them: ')
                finish_options = ['Drop all', 'Continue preprocessing']
                option = print_options(finish_options)
                if option == 0:
                    print('Dropping all categorical features.')
                    df.drop(columns=df.select_dtypes(include=['object', 'category']).columns, inplace=True)
                    break
                elif option == 1:
                    print('Please continue preprocessing')
                    continue
            else:
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
    y = df[target_col.name]
    df.drop(columns=[target_col.name], inplace=True)

    while True:
        test_size = '-1'
        while True:
            print('What test size do you want? (Enter a number between 0 and 0.3)')
            test_size = input()
            try:
                test_size_float = float(test_size)
                if 0 <= test_size_float <= 0.3:
                    break
                else:
                    print("The number must be between 0 and 0.3.")
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

        test_size = float(test_size)
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size)

        model_options = ['RandomForestRegressor', 'RandomForestClassifier']
        models = [RandomForestRegressor, RandomForestClassifier]
        print('Which model would you like to use?')
        option = print_options(model_options)
        min_n_estimators = int(input('Please enter the minimum number of estimators: '))
        max_n_estimators = int(input('Please enter the maximum number of estimators: '))
        min_max_depth = int(input('Please enter the minimum max_depth to test: '))
        max_max_depth = int(input('Please enter the maximum max_depth to test: '))

        filepath = input('Name the folder you want to save the graphs in.')
        os.makedirs(filepath, exist_ok=True)

        n_estimators_list = range(min_n_estimators, max_n_estimators + 1)
        best_max_depth = 0
        best_n_estimators = 0
        max_accuracy = 0
        for d in range(min_max_depth, max_max_depth+1):
            accuracies = []
            for n in n_estimators_list:
                model = models[option](n_estimators=n, max_depth=d)
                acc = create_and_train_model(model, X_train, y_train, X_test, y_test)
                accuracies.append(acc)
                if acc > max_accuracy:
                    max_accuracy = acc
                    best_max_depth = d
                    best_n_estimators = n
                print(f"n_estimators: {n}, max_depth: {d}, accuracy: {acc}")

            plt.scatter(n_estimators_list, accuracies, color='blue')
            plt.plot(n_estimators_list, accuracies, color='blue', linestyle='-', marker='o')
            plt.xlabel('Number of Estimators')
            plt.ylabel('Accuracy')
            plt.title(f'Random Forest Accuracy vs Number of Estimators with Max_Depth: {d}')
            plt.savefig(os.path.join(filepath, f'accuracy_vs_estimators_depth_{d}.png'))
            plt.close()
        print(f"Best number of estimators: {best_n_estimators}, max_depth: {best_max_depth}, accuracy: {max_accuracy}")
        print("Try again with different model/hyperparameters?")
        options = ['Yes', 'No']
        option = print_options(options)
        if option == 0:
            continue
        elif option == 1:
            print('Thank you for using Tabular Trainer!')
            quit()


if __name__ == '__main__':
    main()
