import numpy as np

def custom_kfold(df, k=5, random_state=21):
    np.random.seed(random_state)
    indices = df.index.values.copy()
    np.random.shuffle(indices)

    chunks = np.array_split(indices, k)

    folds = []

    for i in range(k):
        test_idx = chunks[i]

        train_chunks = [chunks[j] for j in range(k) if j != i]
        train_idx = np.concatenate(train_chunks)

        folds.append((train_idx, test_idx))

    return folds

def custom_group_kfold(df, group_field, k=5, random_state=21):
    np.random.seed(random_state)
    unique_groups = np.array(df[group_field].unique())
    np.random.shuffle(unique_groups)

    group_chunks = np.array_split(unique_groups, k)
    
    folds = []
    for i in range(k):
        test_group = group_chunks[i]

        test_mask = df[group_field].isin(test_group)

        test_idx = df[test_mask].index.values
        train_idx = df[~test_mask].index.values

        folds.append((train_idx, test_idx))

    return folds

def custom_stratified_kfold(df, stratify_field, k=5, random_state=21):
    np.random.seed(random_state)

    folds_test_chunks = [[] for _ in range(k)]

    for label, group_df in df.groupby(stratify_field):
        indices = group_df.index.values.copy()
        np.random.shuffle(indices)

        class_chunks = np.array_split(indices, k)

        for i in range(k):
            folds_test_chunks[i].extend(class_chunks[i])

    folds = []
    for i in range(k):
        test_idx = np.array(folds_test_chunks[i])

        train_chunks = [folds_test_chunks[j] for j in range(k) if j != i]
        train_idx = np.concatenate(train_chunks)

        folds.append((train_idx, test_idx))

    return folds

def custom_timeseries_split(df, date_field, k=5):
    df_sorted = df.sort_values(by=date_field)
    df_sorted[date_field] = pd.to_datetime(df_sorted[date_field])
    indices = df_sorted.index.values.copy()

    chunks = np.array_split(indices, k + 1)

    folds = []
    train_idx = chunks[0]

    for i in range(1, k + 1):
        test_idx = chunks[i]
        folds.append((train_idx, test_idx))

        train_idx = np.concatenate([train_idx, test_idx])

    return folds

