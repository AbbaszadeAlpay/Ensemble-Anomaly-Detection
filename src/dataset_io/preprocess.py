import numpy as np
import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder

DROP_COLUMNS = ["zipcode", "date"]
CATEGORICAL_COLUMNS = ["city", "state", "credit_card"]
TARGET = ["transaction_dollar_amount"]


class Preprocess:
    def transform(self, data):
        data["date"] = pd.to_datetime(data["date"])
        data["credit_card"] = data["credit_card"].astype("object")
        data = self._add_features(data)
        data[CATEGORICAL_COLUMNS] = CatBoostEncoder().fit_transform(
            data[CATEGORICAL_COLUMNS], data[TARGET]
        )
        data = data.drop(DROP_COLUMNS, axis=1)
        return data

    @staticmethod
    def _datetime_features(df_temp):
        df_temp["dayofweek"] = df_temp["date"].dt.dayofweek
        df_temp["hour"] = df_temp["date"].dt.hour
        df_temp["minute"] = df_temp["date"].dt.minute
        df_temp["second"] = df_temp["date"].dt.second

        return df_temp

    @staticmethod
    def _circular_encoding(df_temp, column, period):
        df_temp[f"{column}_sin"] = np.sin(2 * np.pi * df_temp[column] / period)
        df_temp[f"{column}_cos"] = np.cos(2 * np.pi * df_temp[column] / period)
        return df_temp

    @staticmethod
    def _seasonality_features(df_temp):
        df_temp = Preprocess._circular_encoding(df_temp, "hour", 24)
        df_temp = Preprocess._circular_encoding(df_temp, "minute", 60)
        df_temp = Preprocess._circular_encoding(df_temp, "second", 60)
        return df_temp

    def _add_features(self, df):
        df = self._datetime_features(df)
        df = self._seasonality_features(df)

        return df
