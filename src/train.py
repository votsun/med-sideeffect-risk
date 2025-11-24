def split_data(df, label_col="label"):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test
