import pandas as pd
from sklearn.preprocessing import LabelEncoder

encoded_columns = set()


def select_column(df):
    cols = df.columns
    col_map = {}
    for i, col in enumerate(cols):
        col_map[i] = col

    selection = '-1'
    while not selection.isnumeric() or int(selection) > len(cols) or int(selection) < 0:
        print('Select a column to perform an operation on: ')

        for i, col in enumerate(cols):
            print(f"{i}. {col} ({df[col].dtype})")

        selection = input()

    target_col = df[col_map[int(selection)]]
    print('\nSelected Column:')
    print(target_col)
    return target_col


def normalize(selected_col, df):
    pass


def impute_missing_data(selected_col, df):
    print('Brief Overview of Missing Data: ')
    print(selected_col.isna())


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



def main():
    pd.set_option('display.max_columns', None)
    print('Welcome to the CSV Trainer!\n')

    print('Enter the file path of the CSV you wish to train:')
    filepath = input()
    df = pd.read_csv(filepath)

    print('\nYour Data: ')
    print(df.head(), '\n')

    while True:
        selected_col = select_column(df)

        if selected_col.dtype == object:
            categorical_options = ['Impute Missing Data', 'Encode', 'Drop Column', 'Cancel']
            categorical_functions = [impute_missing_data, encode, drop_column]
            option = print_options(categorical_options)
            if option == 0:
                continue
            else:
                categorical_functions[option](selected_col, df)
        else:
            numerical_options = ['Normalize', 'Drop Column', 'Cancel']
            numerical_functions = [normalize, drop_column]


if __name__ == '__main__':
    main()