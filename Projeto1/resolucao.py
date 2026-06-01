#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

cols_to_drop_frete = ["order_id", "order_item_id", "product_id", "seller_id" , "shipping_limit_date","customer_id", "order_status",
                      "product_length_cm", "product_height_cm", "product_width_cm", "customer_city"]

def transform_for_frete (df: pd.DataFrame):
    
    df['product_volume_cm'] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"] 
    df = df.fillna(df.median(numeric_only=True))
    df.drop(columns=cols_to_drop_frete, inplace=True)
    
    return df

def limpeza_for_frete(df: pd.DataFrame):

    col_float = df.select_dtypes('float64').columns
    for col in col_float:
        if col != 'freight_value':
            df[col] = np.log1p(df[col])
    return df

df_train = transform_for_frete(df_train)
df_test = transform_for_frete(df_test)
df_train = limpeza_for_frete(df_train)
df_test = limpeza_for_frete(df_test)

df_train = pd.get_dummies(df_train, columns=["customer_state"])
df_test = pd.get_dummies(df_test, columns=["customer_state"])

X_train = df_train.drop("freight_value", axis=1)
y_train = df_train['freight_value']
X_test = df_test
y_test = df_test

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

submission = pd.DataFrame({
    'row_id': df_test['row_id'],
    'freight_value' : y_pred
})

submission.to_csv("submission.csv", index=False)
print(submission.shape)
print(pd.Series(y_pred).describe())
# %%