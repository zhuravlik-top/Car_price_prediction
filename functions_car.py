
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import phik
from phik import resources
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import random
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from category_encoders import TargetEncoder
from sklearn.preprocessing import FunctionTransformer
from IPython.display import display, Markdown
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone


def preprocess_features_before_pipeline(df):
    """
    Предобработка, которую можно сделать ДО пайплайна и кросс-валидации.
    Не использует целевую переменную, безопасно для CV.
    """
    df = df.copy()

    # 1. Очистка
    if "ID" in df.columns:
        df = df.drop_duplicates().drop(columns=["ID"], errors="ignore")

    # 2. Mileage в числовой тк он содержит числа
    if "Mileage" in df.columns:
        df["Mileage"] = (
            df["Mileage"]
            .str.replace(" km", "", regex=False)
            .str.replace(" ", "", regex=False)
            .astype(float)
        )

    # 3. Levy в числовой тк содержит числа
    if "Levy" in df.columns:
        df["Levy"] = df["Levy"].replace("-", "0").astype(float)
    # 4. Dooors в числовой 
    def convert_doors(val):
        val = str(val).strip()
        if "-" in val:
            try:
                return int(val.split("-")[0])
            except:
                return np.nan
        if val.startswith(">"):
            try:
                return int(val[1:]) + 1
            except:
                return np.nan
        try:
            return int(val)
        except:
            return np.nan

    if "Doors" in df.columns:
        df["Doors"] = df["Doors"].apply(convert_doors)


    if "Engine volume" in df.columns:
        df["is_turbo"] = df["Engine volume"].str.contains("Turbo", case=False, na=False).astype(int)
        df["engine_size"] = df["Engine volume"].str.extract(r"(\d+\.?\d*)").astype(float)
        df = df.drop(columns=["Engine volume"])


    if "Leather interior" in df.columns:
        df["Leather interior"] = df["Leather interior"].map({"Yes": 1, "No": 0})

    return df


def prepare_regression_data(df, seed=42, test_size=0.2, verbose=True):
    """
    Полная предобработка данных для задачи регрессии.
    Возвращает X, y, train/test split и предобработанные DataFrame с названиями признаков.
    """
    print("🔥 prepare_regression_data() called — CLEAN 8-VARIANT version")

    df = df.copy()

    # 1. Очистка
    if "ID" in df.columns:
        df = df.drop_duplicates().drop(columns=["ID"], errors="ignore")

    if "Mileage" in df.columns:
        df["Mileage"] = (
            df["Mileage"]
            .str.replace(" km", "", regex=False)
            .str.replace(" ", "", regex=False)
            .astype(int)
        )

    if "Levy" in df.columns:
        df["Levy"] = df["Levy"].replace("-", "0").astype(int)

    # 2. Целевая переменная
    y = df["Price"]
    X = df.drop(columns=["Price"])

    # 3. Обработка Doors
    def convert_doors(val):
        val = str(val).strip()
        if "-" in val:
            try:
                return int(val.split("-")[0])
            except:
                return np.nan
        if val.startswith(">"):
            try:
                return int(val[1:]) + 1
            except:
                return np.nan
        try:
            return int(val)
        except:
            return np.nan

    if "Doors" in X.columns:
        X["Doors"] = X["Doors"].apply(convert_doors)

    # 4. Engine volume
    if "Engine volume" in X.columns:
        X["is_turbo"] = X["Engine volume"].str.contains("Turbo", case=False, na=False).astype(int)
        X["engine_size"] = X["Engine volume"].str.extract(r"(\d+\.?\d*)").astype(float)
        X = X.drop(columns=["Engine volume"])

    # 5. Leather interior
    if "Leather interior" in X.columns:
        X["Leather interior"] = X["Leather interior"].map({"Yes": 1, "No": 0})

    # 6. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    y_train_log = np.log1p(y_train)

    # 7. Target Encoding
    cat_target = ["Manufacturer", "Model", "Color"]
    cat_onehot = ["Category", "Fuel type", "Gear box type", "Drive wheels", "Wheel"]

    target_cols = [c for c in cat_target if c in X_train.columns]
    if target_cols:
        te = TargetEncoder(
            handle_unknown="value",
            handle_missing="value",
            min_samples_leaf=5,
            smoothing=10
        )
        X_train[target_cols] = te.fit_transform(X_train[target_cols], y_train_log)
        X_test[target_cols] = te.transform(X_test[target_cols])

        global_mean = y_train_log.mean()
        X_train[target_cols] = X_train[target_cols].fillna(global_mean)
        X_test[target_cols] = X_test[target_cols].fillna(global_mean)

    # 8. ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("one_hot", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), [col for col in cat_onehot if col in X_train.columns]),
            ("num", SimpleImputer(strategy="median"), [
                col for col in X_train.columns if col not in cat_onehot
            ]),
        ],
        remainder="drop"
    )

    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out(input_features=X_train.columns)

    X_train_pre = pd.DataFrame(preprocessor.transform(X_train),
                               index=X_train.index,
                               columns=feature_names)
    X_test_pre = pd.DataFrame(preprocessor.transform(X_test),
                              index=X_test.index,
                              columns=feature_names)

    # 9. Финальная импьютация
    imputer = SimpleImputer(strategy="most_frequent")
    X_train_preprocessed = pd.DataFrame(imputer.fit_transform(X_train_pre),
                                        index=X_train.index,
                                        columns=feature_names)
    X_test_preprocessed = pd.DataFrame(imputer.transform(X_test_pre),
                                       index=X_test.index,
                                       columns=feature_names)

    # Проверка
    assert not X_train_preprocessed.isna().any().any(), "NaN в X_train_preprocessed!"
    assert not X_test_preprocessed.isna().any().any(), "NaN в X_test_preprocessed!"

    # 10. Информационное сообщение
    if verbose:
        display(Markdown(
            f"✅ **Preprocessing complete.**  \n"
            f"Train samples: **{len(X_train_preprocessed)}**, "
            f"Test samples: **{len(X_test_preprocessed)}**  \n"
            f"Features: **{len(feature_names)}**"
        ))
    print("➡ RETURN executed with 8 values")

    return (
        X, y,
        X_train, y_train,
        X_test, y_test,
        X_train_preprocessed,
        X_test_preprocessed
    )




def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def symmetric_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))




def evaluate_models_cv_regression_safe_new(
    models,
    preprocessor,
    X,
    y,
    cv=5,
    seed=None,
    log=False
):
    """
    Кросс-валидация для регрессионных моделей с готовым preprocessor.
    Любой feature engineering должен быть сделан внутри preprocessor.
    
    Эта версия безопасно подстраивается под отсутствующие колонки в X.
    """
    
    all_metrics = {}

    # Определяем реальные колонки из X
    X_cols = set(X.columns)

    # Создаем копию preprocessor, чтобы не менять оригинал
    preprocessor_safe = clone(preprocessor)

    # Фильтруем трансформеры внутри ColumnTransformer
    if isinstance(preprocessor_safe, ColumnTransformer):
        new_transformers = []
        for name, transformer, cols in preprocessor_safe.transformers:
            # Берем только колонки, которые реально есть в X
            cols_filtered = [c for c in cols if c in X_cols]
            if len(cols_filtered) > 0:
                new_transformers.append((name, transformer, cols_filtered))
        preprocessor_safe.transformers = new_transformers

    for name, base_model in models:
        print(f"\n{'='*60}\nКросс-валидация модели: {name}\n{'='*60}")

        if seed is not None and hasattr(base_model, "random_state"):
            base_model.set_params(random_state=seed)

        # Логарифмирование цели
        model = (
            TransformedTargetRegressor(
                regressor=base_model,
                func=np.log1p,
                inverse_func=np.expm1
            ) if log else base_model
        )

        # Полный пайплайн: preprocessor + модель
        pipeline = Pipeline([
            ("preprocessor", preprocessor_safe),
            ("model", model)
        ])

        # Кросс-валидация
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)

        # Метрики
        MAE = mean_absolute_error(y, y_pred)
        MSE = mean_squared_error(y, y_pred)
        RMSE = np.sqrt(MSE)
        R2 = r2_score(y, y_pred)
        #MAPE = np.mean(np.abs((y - y_pred) / np.maximum(y, 1e-8))) * 100
        SMAPE = 100 * np.mean(2 * np.abs(y - y_pred) / (np.abs(y) + np.abs(y_pred) + 1e-8))

        all_metrics[name] = {
            "MAE": MAE,
            "MSE": MSE,
            "RMSE": RMSE,
            "R2": R2,
            #"MAPE": MAPE,
            "SMAPE": SMAPE
        }

        print(f"R2:    {R2:.4f}")
        print(f"MSE:   {MSE:.2f}")
        print(f"MAE:   {MAE:.2f}")
        print(f"RMSE:  {RMSE:.2f}")
        #print(f"MAPE:  {MAPE:.2f}%")
        print(f"SMAPE: {SMAPE:.2f}%")

    # Таблица результатов
    df_results = pd.DataFrame(all_metrics).T
    df_results = df_results[["MAE", "MSE", "RMSE", "R2","SMAPE"]].sort_values(by="R2", ascending=False)

    print("\n=== Сводная таблица метрик (усреднённые по CV) ===")
    print(df_results.to_string(float_format="%.4f"))

    return df_results




def compare_regression_metrics(df1, df2, name1="Variant 1", name2="Variant 2", plot=True):
    """
    Сравнивает результаты метрик регрессии между двумя вариантами (например, обычные и нормализованные данные).

    Параметры:
        df1, df2: pd.DataFrame с одинаковыми индексами (модели) и метриками
        name1, name2: имена вариантов (используются в подписях)
        plot: если True — визуализирует разницу по R2 и MAPE

    Возвращает:
        diff_df — таблица с дельтами метрик (df2 - df1)
    """
    # Проверка соответствия моделей
    if not all(df1.index == df2.index):
        raise ValueError("Модели в df1 и df2 должны совпадать по порядку и названию")

    # Совпадение метрик
    common_cols = df1.columns.intersection(df2.columns)
    if len(common_cols) == 0:
        raise ValueError("Нет общих метрик между df1 и df2")

    # Разница
    diff = df2[common_cols] - df1[common_cols]
    diff.index.name = "Model"
    diff = diff.rename(columns=lambda x: f"Δ{x} ({name2}-{name1})")

    print(f"\n=== 📊 Сравнение метрик: {name2} против {name1} ===")
    print(diff.to_string(float_format="%.4f"))

    # Определяем направление улучшений
    better_higher = ["R2", "R2_CV"]
    better_lower = ["MSE", "RMSE", "MAE", "SMAPE"]

    trends = []
    for model in diff.index:
        notes = []
        for col in diff.columns:
            base = col.replace(f"Δ", "").split(" ")[0]
            val = diff.loc[model, col]
            if base in better_higher:
                notes.append("↑" if val > 0 else "↓")
            elif base in better_lower:
                notes.append("↑" if val < 0 else "↓")
        trends.append(" ".join(notes))
    diff["Trend"] = trends

    # Визуализация (по R2 и MAPE, если есть)
    if plot:
        metrics_to_plot = [m for m in ["R2", "MAPE"] if any(col.startswith(f"Δ{m}") for col in diff.columns)]
        if metrics_to_plot:
            fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 4))
            if len(metrics_to_plot) == 1:
                axes = [axes]
            for i, metric in enumerate(metrics_to_plot):
                col = [c for c in diff.columns if c.startswith(f"Δ{metric}")][0]
                diff[col].plot(kind='bar', ax=axes[i], color='steelblue', edgecolor='black')
                axes[i].axhline(0, color='black', linewidth=1)
                axes[i].set_title(f"{metric}: Δ({name2}-{name1})")
                axes[i].set_ylabel("Изменение")
                axes[i].set_xlabel("Модель")
            plt.tight_layout()
            plt.show()

    return diff

