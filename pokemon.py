def train_test_splitter(DataFrame, column):
    df_train = DataFrame.loc[df[column] != 1]
    df_test = DataFrame.loc[df[column] == 1]
    df_train = df_train.drop(column, axis=1)
    df_test = df_test.drop(column, axis=1)
    return (df_train, df_test)


def label_delineator(df_train, df_test, label):
    train_data = df_train.drop(label, axis=1).values
    train_labels = df_train[label].values
    test_data = df_test.drop(label, axis=1).values
    test_labels = df_test[label].values
    return (train_data, train_labels, test_data, test_labels)


def data_normalizer(train_data, test_data):
    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    return (train_data, test_data)


train_data, test_data = data_normalizer(train_data, test_data)


def predictor(test_data, test_labels, index):
    prediction = model.predict(test_data)
    if np.argmax(prediction[index]) == test_labels[index]:
        print(f'This was correctly predicted to be a \"{test_labels[index]}\"!')
    else:
        print(
            f'This was incorrectly predicted to be a \"{np.argmax(prediction[index])}\". It was actually a \"{test_labels[index]}\".')
        return (prediction)
