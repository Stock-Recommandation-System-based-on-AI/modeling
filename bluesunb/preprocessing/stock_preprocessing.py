import numpy as np
import pandas as pd


def _fill_zero(df, epsilon=1e-6):
    if any(df.isnull().sum()):
        df.dropna(inplace=True)
    return df.replace(0, epsilon)


def _moving_avg(df):
    df[df.columns[1:]] = df[df.columns[1:]].rolling(10).mean()
    df.dropna(how='any', axis=0, inplace=True)
    return _fill_zero(df)


def _add_datatime_tag(df, name_tag=False):
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    if name_tag:
        df['datename'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df = df.sort_values(by=['date'])
    return df


def _pct_change(df, c_min, c_max):
    df = df.pct_change()
    df.dropna(axis=0, inplace=True)
    df = df.clip(c_min, c_max, axis=1)
    return df


def _normalize_pct(df, groups, n_pct=20):
    pct = n_pct / 100 if n_pct > 1 else n_pct
    times = sorted(df.index.values)
    last_20pct = sorted(df.index.values)[-int(pct * len(times))]

    def normalize(cols):
        if isinstance(cols, str):
            normalize([cols])
        else:
            min_return = min(df[df.index < last_20pct][cols].min(axis=0))
            max_return = max(df[(df.index < last_20pct)][cols].max(axis=0))

            for c in cols:
                df[c] = (df[c] - min_return) / (max_return - min_return)

    for group in groups:
        normalize(group)

    return df


def _split_train_test(df, valid_pct=10, test_pct=10):
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    if test_pct > 1:
        test_pct /= 100
    if valid_pct > 1:
        valid_pct /= 100

    n_valid = int(len(df) * valid_pct)
    n_test = int(len(df) * test_pct)
    n_train = int(len(df) - n_valid - n_test)

    df_train = df.iloc[:n_train]
    df_valid = df.iloc[n_train:n_train + n_valid]
    df_test = df.iloc[-n_test:]

    train_data = df_train.values
    valid_data = df_valid.values
    test_data = df_test.values
    return train_data, valid_data, test_data


def _split_label(df, label_idx, seq_len, pred_len):
    X_data, y_data = [], []
    for i in range(seq_len, len(df) - pred_len + 1):
        X_data.append(df[i - seq_len: i])
        y_data.append(df[i:i + pred_len, label_idx].flatten())
    return np.array(X_data), np.array(y_data)


def load_data_with_preprocessing(path, options):
    df = pd.read_csv(path)
    df.drop(['code', 'name', 'section', 'n_stock'], axis=1, inplace=True)

    df = _fill_zero(df)
    df = _moving_avg(df)
    df = _add_datatime_tag(df, name_tag=False)

    df.drop('date', axis=1, inplace=True)
    # using_cols = ['open', 'high', 'low', 'close', 'vol']
    using_cols = options.get('using_cols', df.columns.tolist())
    df = df[using_cols]

    df = _pct_change(df, c_min=-2.0, c_max=2.0)
    df = _normalize_pct(df, groups=[['open', 'high', 'low', 'close'],
                                    ['vol']], n_pct=20)

    train_data, valid_data, test_data = _split_train_test(df, valid_pct=10, test_pct=10)
    label_col = options.get('label_cols', ['close'])
    label_idx = [using_cols.index(label) for label in label_col]
    seq_len = options['seq_len']
    pred_len = options['pred_len']

    X_train, y_train = _split_label(train_data, label_idx, seq_len, pred_len)
    X_valid, y_valid = _split_label(valid_data, label_idx, seq_len, pred_len)
    X_test, y_test = _split_label(test_data, label_idx, seq_len, pred_len)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), df