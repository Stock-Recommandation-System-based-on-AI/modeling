import numpy as np
import pandas as pd

def __market_stock_split(df):
    market_df = df[df.code=='U182']
    stock_df = df[df.code!='U182']
    return market_df, stock_df

def __stock_seperate(stocks):
    stock_names = stocks['name'].unique()
    res = []
    for stock in stock_names:
        res.append(stocks[stocks.name==stock])
    return res

def __train_test_split(stocks_df, test_pct=0.2, valid_pct=0.2):
    if not isinstance(stocks_df, (list, tuple)):
        return __train_test_split([stocks_df], test_pct, valid_pct)

    stocks_train, stocks_valid, stocks_test = [],[],[]
    for stock in stocks_df:
        test_size = int(len(stock)*test_pct)
        valid_size = int(len(stock)*valid_pct)
        train_size = len(stock) - test_size - valid_size

        stocks_train.append(stock[:train_size])
        stocks_valid.append(stock[train_size:train_size+valid_size])
        stocks_test.append(stock[-test_size:])

    return stocks_train, stocks_valid, stocks_test

def __input_target_split(stocks_df):
    target = [stock['close'].apply(lambda x: int(x>0)) for stock in stocks_df]
    return stocks_df, target

def __window_split(inputs, targets, window_size=10):
    inputs = inputs.values
    targets = targets.values
    X = []
    y = []
    for i in range(window_size, len(inputs)):
        X.append(inputs[i-window_size:i])
        y.append(targets[i])

    return np.array(X), np.array(y)

def __stocks_window_split(stocks_inputs, stocks_target, window_size=10):
    X,y = [],[]
    for j in range(window_size, len(stocks_inputs[0])):
        bucket_X = []
        bucket_y = []
        for i in range(len(stocks_inputs)):
            stock = stocks_inputs[i].values
            target = stocks_target[i].values
            bucket_X.append(stock[j-window_size:j])
            bucket_y.append(target[j])
        X.append(bucket_X)
        y.append(bucket_y)
    return np.array(X), np.array(y)

def prepare_data(path, **kwargs):
    df = pd.read_csv(path)
    market, stocks = __market_stock_split(df)
    stocks = __stock_seperate(stocks)
    print(len(stocks))
    for i in range(len(stocks)):
        stocks[i].drop(['code','name', 'date'], axis=1, inplace=True)
    market.drop(['code','name', 'date'], axis=1, inplace=True)

    market_train_valid_test = __train_test_split(market,
                                                 test_pct=kwargs.get('test_pct'),
                                                 valid_pct=kwargs.get('valid_pct')) # [m_train], [m_valid], [m_test]
    stock_train_valid_test = __train_test_split(stocks,
                                                test_pct=kwargs.get('test_pct'),
                                                valid_pct=kwargs.get('valid_pct')) # [s_train1, ...], [s_valid1, ...], [s_test1, ...]

    market_data = [__input_target_split(m) for m in market_train_valid_test] # [([m_train_X], [m_train_y]), (...), (...)]
    stock_data = [__input_target_split(s) for s in stock_train_valid_test] # [([s_train1_X, ...], [s_train1_y, ...]), (...), (...)]

    window_size = kwargs['window_size']
    market_data = [__stocks_window_split(*m, window_size) for m in market_data]
    stock_data = [__stocks_window_split(*s, window_size) for s in stock_data]
    return market_data, stock_data

