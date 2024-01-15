# ライブラリの読み込み
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# Commented out IPython magic to ensure Python compatibility.
# Google Colaboratoryで作業する場合はこちらも実行してください。
from google.colab import drive
drive.mount('/content/drive')
# %cd 以降にこのnotebookを置いているディレクトリを指定してください。
# %cd "/content/drive/MyDrive/Final"

data = pd.read_csv("data2.csv")

# カテゴリカルな特徴量を数値に変換
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column != 'Attrition':  # 目的変数は後で処理
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# 目的変数（Attrition）をラベルに変換
le_attrition = LabelEncoder()
data['Attrition'] = le_attrition.fit_transform(data['Attrition'])

# 勤務年数が5年以下のデータに絞る
filtered_data_5_years = data[data['YearsAtCompany'] <= 5]

# 特徴量と目的変数を分離（勤務年数5年以下用）
X_filtered_5_years = filtered_data_5_years.drop('Attrition', axis=1)
y_filtered_5_years = filtered_data_5_years['Attrition']

# 訓練セットとテストセットに分割（勤務年数5年以下用）
X_train_filtered_5_years, X_test_filtered_5_years, y_train_filtered_5_years, y_test_filtered_5_years = train_test_split(
    X_filtered_5_years, y_filtered_5_years, test_size=0.3, random_state=42)

# ランダムフォレストモデルのトレーニング（勤務年数5年以下用）
rf_model_filtered_5_years = RandomForestClassifier(random_state=42)
rf_model_filtered_5_years.fit(X_train_filtered_5_years, y_train_filtered_5_years)

# 特徴量の重要度（勤務年数5年以下用）
feature_importances_filtered_5_years = rf_model_filtered_5_years.feature_importances_

# 特徴量名と重要度をデータフレームに格納（勤務年数5年以下用）
features_df_filtered_5_years = pd.DataFrame({'Feature': X_filtered_5_years.columns, 'Importance': feature_importances_filtered_5_years})

# 重要度の降順にソート（勤務年数5年以下用）
features_df_filtered_5_years = features_df_filtered_5_years.sort_values(by='Importance', ascending=False)

# 特徴量の重要度グラフの表示（勤務年数5年以下用）
plt.figure(figsize=(12, len(features_df_filtered_5_years) / 2))
sns.barplot(x='Importance', y='Feature', data=features_df_filtered_5_years)
plt.title('Feature Importances in Random Forest Model (YearsAtCompany ≤ 5)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# テストデータでの予測確率（勤務年数5年以下用）
y_pred_proba_filtered_5_years = rf_model_filtered_5_years.predict_proba(X_test_filtered_5_years)[:, 1]

# ROCカーブの計算（勤務年数5年以下用）
fpr_filtered_5_years, tpr_filtered_5_years, thresholds_filtered_5_years = roc_curve(y_test_filtered_5_years, y_pred_proba_filtered_5_years)
roc_auc_filtered_5_years = auc(fpr_filtered_5_years, tpr_filtered_5_years)

# ROCカーブを描画（勤務年数5年以下用）
plt.figure(figsize=(8, 8))
plt.plot(fpr_filtered_5_years, tpr_filtered_5_years, color='blue', lw=2, label='ROC curve (area = %0.3f)' % roc_auc_filtered_5_years)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (YearsAtCompany ≤ 5)')
plt.legend(loc="lower right")
plt.show()

# AUCを小数第三位まで表示（勤務年数5年以下用）
roc_auc_rounded_filtered_5_years = round(roc_auc_filtered_5_years, 3)
roc_auc_rounded_filtered_5_years

# Getting feature importances
feature_importances_full = rf_model_filtered_5_years.feature_importances_

# Creating a DataFrame to hold feature names and their importances
features_full = pd.DataFrame({
    'Feature': X_train_filtered_5_years.columns,
    'Importance': feature_importances_full
})

# Sorting the DataFrame based on importance to get the top 10 features (excluding 'OverTime')
features_full = features_full.sort_values(by='Importance', ascending=False)
top_10_features_full = features_full[features_full['Feature'] != 'OverTime']['Feature'].tolist()[:11]
top_10_features_full

X_train = filtered_data_5_years[top_10_features_full]
y_train = filtered_data_5_years['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# Creating and training the Random Forest model for overtime data
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train,y_train)

# Compute ROC curve and ROC area for the model
fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve for the overtime model
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Overtime Model')
plt.legend(loc="lower right")
plt.show()

no_overtime_data = filtered_data_5_years[filtered_data_5_years['OverTime'] == 0]
# Using the identified top 10 features to train a model for employees who do overtime
X_train_no_overtime = no_overtime_data[top_10_features_full]
y_train_no_overtime = no_overtime_data['Attrition']

# Splitting the overtime dataset into training and testing sets
X_train_no_overtime, X_test_no_overtime, y_train_no_overtime, y_test_no_overtime = train_test_split(
    X_train_no_overtime, y_train_no_overtime, test_size=0.2, random_state=42)

# Creating and training the Random Forest model for overtime data
rf_model_no_overtime = RandomForestClassifier(random_state=42)
rf_model_no_overtime.fit(X_train_no_overtime, y_train_no_overtime)

# Model trained successfully
"Model trained using top 10 features for employees who do not overtime"
# Compute ROC curve and ROC area for the no_overtime model
fpr_no_overtime, tpr_no_overtime, _ = roc_curve(y_test_no_overtime, rf_model_no_overtime.predict_proba(X_test_no_overtime)[:,1])
roc_auc_no_overtime = auc(fpr_no_overtime, tpr_no_overtime)

# Plotting the ROC curve for the no_overtime model
plt.figure(figsize=(8, 6))
plt.plot(fpr_no_overtime, tpr_no_overtime, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_no_overtime:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Overtime Model')
plt.legend(loc="lower right")
plt.show()

# 残業していない人のデータを抽出（OverTime = 0）
no_overtime_data = data[data['OverTime'] == 0]

# 指定された特徴量のリスト
features = [
    'MonthlyIncome', 'RemoteWork', 'Age', 'DailyAchievement',
    'MonthlyAchievement', 'Incentive', 'TotalWorkingYears',
    'DistanceFromHome', 'HourlyAchievement', 'YearsAtCompany',
    'NumCompaniesWorked'
]

# 指定された特徴量を持つサンプルを選択
sample_data_no_overtime = no_overtime_data[features].head()

# 退職確率の予測
predicted_probabilities = rf_model_no_overtime.predict_proba(sample_data_no_overtime)[:, 1]

# 予測された確率をデータフレームに追加
sample_data_no_overtime['Predicted Attrition Probability'] = predicted_probabilities

# 結果の表示
sample_data_no_overtime

employee_data = pd.DataFrame([{
    'MonthlyIncome':3485,
    'RemoteWork':2,
    'Age':28,
    'DailyAchievement':529,
    'MonthlyAchievement':14935,
    'Incentive':0,
    'TotalWorkingYears':5,
    'DistanceFromHome':2,
    'HourlyAchievement':79,
    'YearsAtCompany':0,
    'NumCompaniesWorked':2
}])

# Predicting the attrition probability for the given employee
attrition_probability = rf_model_no_overtime.predict_proba(employee_data)[0, 1]
attrition_probability

employee_data2 = pd.DataFrame([{
    'MonthlyIncome': 2174,
    'RemoteWork': 2,
    'Age': 21,
    'DailyAchievement': 756,
    'MonthlyAchievement': 9150,
    'Incentive':0,
    'TotalWorkingYears':3,
    'DistanceFromHome':1,
    'HourlyAchievement':99,
    'YearsAtCompany':3,
    'NumCompaniesWorked':1
}])

# Predicting the attrition probability for the given employee
attrition_probability = rf_model_no_overtime.predict_proba(employee_data2)[0, 1]
attrition_probability