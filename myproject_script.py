# %%
import os
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import math
#from sqlalchemy import create_engine
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
#from imblearn.under_sampling import RandomUnderSampler

# %%
df_account = pd.read_csv(r"C:\Users\toraL\dirisl_phase1\dataset\phase1_data\acocunt.csv", dtype_backend='numpy_nullable')
df_order = pd.read_csv(r"C:\Users\toraL\dirisl_phase1\dataset\phase1_data\order.csv", dtype_backend='numpy_nullable')
df_pex = pd.read_csv(r"C:\Users\toraL\dirisl_phase1\dataset\phase1_data\personnel_expenses.csv", dtype_backend='numpy_nullable')

# %%
#データの確認
print(df_account.head())
print(df_order.head())
print(df_pex.head())
# %%
print(df_account.shape)
print(df_order.shape)
print(df_pex.shape)
# %%
#データ型の確認
print(df_account.info())
print(df_order.info())
print(df_pex.info())

# %%
#データ加工
df_order2 = df_order.copy()
#print(df_order2.columns)
#print(df_order2.info())
df_order2['original_price'] = df_order2.apply(lambda row:0.35*row['total_price'] if row['category'] in ['food','wine'] else 0.25*row['total_price'], axis=1)
df_order2['benefit'] = df_order2['total_price'] - df_order2['original_price']
df_order2 = df_order2.groupby('payment_id').agg({'total_price':'sum',
                              'quantity':'sum',
                              'original_price':'sum',
                              'benefit':'sum'}).reset_index()
print(df_order2.head())
# %%
df_account2 = df_account.copy()
df_tmp = pd.merge(df_account2, 
                  df_order2, 
                  how='inner', 
                  on='payment_id')
print(df_tmp.head())
# %%
# 月ごとにまとめる
print(df_tmp.info())
# %%
df_tmp['visit_on_month'] = pd.to_datetime(df_tmp['visit_on']).dt.strftime('%Y%m')
#total_priceは合計でいいのだろうか？
df_tmp2 = df_tmp.groupby('visit_on_month').agg({'total_price':'sum',
                                                'original_price':'sum',
                                                'benefit':'sum'}).reset_index()
print(df_tmp2.head())
# %%
# 人件費データ
df_pex2 = df_pex.copy()
df_pex2['work_on_month'] = pd.to_datetime(df_pex2['work_on']).dt.strftime('%Y%m')
df_pex2_month = df_pex2.groupby('work_on_month').agg({'personnel_expense':'sum'}).reset_index()
print(df_pex2_month.head()) 
# %%
#月ごとの人件費データと売上データを結合
df_pex2_month = df_pex2_month.rename(columns={'work_on_month':'month'})
df_tmp2 = df_tmp2.rename(columns={'visit_on_month':'month'})
#print(df_tmp2.columns)
df_accounting = pd.merge(df_tmp2, df_pex2_month, how='inner', on='month')
print(df_accounting.head())
# %%
#コスト算出
df_accounting['fix_other_cost'] = 60000+180000+90000
df_accounting['var_other_cost'] = df_accounting['total_price']*0.1
df_accounting['total_cost'] = df_accounting['personnel_expense'] + df_accounting['fix_other_cost'] + df_accounting['var_other_cost']
df_accounting['true_benefit'] = df_accounting['benefit'] - df_accounting['total_cost']

print(df_accounting.head())
# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
# %%
#その１について可視化
fig, ax = plt.subplots()
fig.autofmt_xdate()#x軸を見やすくする
ax.plot(df_accounting['month'], df_accounting['total_price'], label='売上')
ax.plot(df_accounting['month'], df_accounting['benefit'], label='粗利')
ax.plot(df_accounting['month'], df_accounting['total_cost'], label='販管費')
ax.plot(df_accounting['month'], df_accounting['true_benefit'], label='営業利益')
ax.set_xlabel('month')
ax.set_ylabel('amount')
ax.set_title('各経営指標推移')
ax.legend()
plt.grid(color = "gray", linestyle="--")
plt.show()

# %%
# リピート率について
df_account2 = df_account.copy()

df_account2['weeknumber'] = pd.to_datetime(df_account2['visit_on']).dt.weekday
df_account2['day_of_week'] = pd.to_datetime(df_account2['visit_on']).dt.strftime('%A')
df_repeat = df_account2.groupby('day_of_week').apply(lambda x:x['survey_response'].value_counts()) #曜日ごとにsurvey_responceをカウントする
df_repeat = df_repeat.reindex(index=['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
print(df_repeat)

# %%
# 曜日ごとの売上平均
#print(df_account2.head())
df_order2 = df_order.copy()
df_order3 = df_order2.groupby('payment_id').agg({'total_price':'sum'}).reset_index()
df_repeat_amount = pd.merge(df_account2, df_order3, how='inner', on='payment_id')
df_repeat_amount = df_repeat_amount.groupby('day_of_week').agg({'total_price':'sum'}).reset_index()
print(df_repeat_amount)

# %%
#リピート率可視化
#
df_repeat_percentage = df_repeat.div(df_repeat.sum(axis=1), axis=0)*100
print(df_repeat_percentage)
# %%
df_repeat_comp = pd.merge(df_repeat_amount, df_repeat_percentage, how='inner', on='day_of_week')
# 曜日の順序を定義（カスタム順序: 月曜日始まり）
day_order = ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# 'day_of_week' 列をカテゴリ型に変換して順序を設定
df_repeat_comp['day_of_week'] = pd.Categorical(df_repeat_comp['day_of_week'], categories=day_order, ordered=True)

# カスタム順序に基づいてソート
df_repeat_comp = df_repeat_comp.sort_values('day_of_week').reset_index(drop=True)
#df_repeat_comp = df_repeat_comp.reindex(index=['4', '5', '3', '0', '1', '2'])
print(df_repeat_comp)
# %%
print(df_repeat_comp.columns)
print(df_repeat_comp.dtypes)
# %%
fig, ax1 = plt.subplots()

ax1.bar(df_repeat_comp['day_of_week'], df_repeat_comp['total_price'], label='売上', width=0.5, color='#8b0000')
ax1.set_ylabel('売上', color='#8b0000', rotation='horizontal')
ax1.tick_params(axis='y', labelcolor='#8b0000')
#ax.bar(df_repeat.index, df_repeat_percentage[0], label='アンケート回答なし')
#ax.bar(df_repeat.index, df_repeat_percentage[1], bottom = df_repeat_percentage[0], label='初来店')
#ax.bar(df_repeat.index, df_repeat_percentage[2], bottom = df_repeat_percentage[0] + df_repeat_percentage[1], label='リピーター')

ax2 = ax1.twinx()
ax2.plot(df_repeat_comp['day_of_week'], df_repeat_comp[2], color='orange', marker='o')
ax2.set_ylabel('リピート率', color='orange', rotation='horizontal')
ax2.tick_params(axis='y', labelcolor='orange')

#ax.set_ylabel('割合')
#ax.set_title('曜日とリピート率')
#ax.legend(title='リピート率')
plt.title('曜日別売り上げとリピート率')
plt.xticks(rotation=45)  # X軸ラベルの回転
plt.tight_layout()
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
plt.show()
# %%
#子供の有無が、一人当たりの注文数がカテゴリごとに与える影響を調べる
df_account2 = df_account.copy()
df_order2 = df_order.copy()

df_account2['child_flag'] = df_account2['child_count'].apply(lambda x:1 if x>0 else 0)
df_account2['people_count'] = df_account2['child_count'] + df_account2['adult_count']
df_order2 = pd.get_dummies(df_order2, columns=['category'], dtype=int)
df_order3 = df_order2.groupby('payment_id').agg(sum_price=('total_price','sum',),
                                                coffee_quantity=('category_coffee',lambda x:(x * df_order2.loc[x.index, 'quantity']).sum()),
                                                dessert_quantity=('category_dessert',lambda x:(x * df_order2.loc[x.index, 'quantity']).sum()),
                                                drink_quantity=('category_drink',lambda x:(x * df_order2.loc[x.index, 'quantity']).sum()),
                                                food_quantity=('category_food',lambda x:(x * df_order2.loc[x.index, 'quantity']).sum()),
                                                wine_quantity=('category_wine',lambda x:(x * df_order2.loc[x.index, 'quantity']).sum())).reset_index()
                                                
print(df_order3.head())

# %%
#確認
#print(df_order2.head(10))
#結合
df_child = pd.merge(df_order3, df_account2[['payment_id', 'child_flag', 'people_count']], how='inner', on='payment_id')
print(df_child.head())
# %%
df_child['coffee_quantity'] = df_child['coffee_quantity'] / df_child['people_count']
df_child['dessert_quantity'] = df_child['dessert_quantity'] / df_child['people_count']
df_child['drink_quantity'] = df_child['drink_quantity'] / df_child['people_count']
df_child['food_quantity'] = df_child['food_quantity'] / df_child['people_count']
df_child['wine_quantity'] = df_child['wine_quantity'] / df_child['people_count']

df_child2 = df_child.groupby('child_flag').agg({'coffee_quantity':'mean',
                                               'dessert_quantity':'mean',
                                               'drink_quantity':'mean',
                                               'food_quantity':'mean',
                                               'wine_quantity':'mean'}).reset_index()
print(df_child2)
                                    
# %%
#子供の有無が一人当たりの注文に与える影響の可視化
position = np.arange(len(df_child2.index))
gap = 0.15

plt.bar(position, df_child2['coffee_quantity'], width=0.15, color='#8b4513', label='coffee')
plt.bar(position+gap, df_child2['wine_quantity'], width=0.15, color='#ff1493', label='wine')
plt.bar(position+gap*2, df_child2['drink_quantity'], width=0.15, color='#b0e0e6', label='drink')
plt.bar(position+gap*3, df_child2['food_quantity'], width=0.15, color='#7cfc00', label='food')
plt.bar(position+gap*4, df_child2['dessert_quantity'], width=0.15, color='#dda0dd', label='dessert')
new_labels = ['子供なし', '子供有']
plt.xticks(position+gap*2, labels=new_labels)  #2つ目の系列の中心にラベルが表示される
plt.xlabel('')
plt.ylabel('一人当たりの注文数', rotation='horizontal') 
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# %%
#カテゴリ別の売上を、月ごとに確認したい
##カテゴリーのダミー変数化＋集計
df_order2 = df_order.copy()
df_order2 = pd.get_dummies(df_order2, columns=['category'], dtype=int)
df_order2['coffee_sum_amount'] = df_order2['total_price'] * df_order2['category_coffee'] 
df_order2['dessert_sum_amount'] = df_order2['total_price'] * df_order2['category_dessert']
df_order2['drink_sum_amount'] = df_order2['total_price'] * df_order2['category_drink']
df_order2['food_sum_amount'] = df_order2['total_price'] * df_order2['category_food']
df_order2['wine_sum_amount'] = df_order2['total_price'] * df_order2['category_wine']
df_order3 = df_order2.groupby('payment_id').agg({'coffee_sum_amount':'sum',
                                                 'dessert_sum_amount':'sum',
                                                 'drink_sum_amount':'sum',
                                                 'food_sum_amount':'sum',
                                                 'wine_sum_amount':'sum'}).reset_index()
                                                
print(df_order3.head())
# %%
##決済データの決済日から決済月を出して結合
df_account2 = df_account.copy()

df_account2['visit_on_month'] = pd.to_datetime(df_account2['visit_on']).dt.strftime('%Y%m')

df_amount_cat = pd.merge(df_order3, df_account2[['payment_id', 'visit_on_month']], how='inner', on='payment_id')
#print(df_amount_cat.head())
df_amount_cat = df_amount_cat.groupby('visit_on_month').agg(coffee=('coffee_sum_amount','sum'),
                                                            dessert=('dessert_sum_amount','sum'),
                                                            drink=('drink_sum_amount','sum'),
                                                            food=('food_sum_amount','sum'),
                                                            wine=('wine_sum_amount','sum')).reset_index()
print(df_amount_cat.head())
# %%
#カテゴリ別の売上を折れ線グラフで可視化

fig, ax = plt.subplots()
fig.autofmt_xdate()#x軸を見やすくする
ax.plot(df_amount_cat['visit_on_month'], df_amount_cat['coffee'], label='コーヒー', color='#8b4513')
ax.plot(df_amount_cat['visit_on_month'], df_amount_cat['wine'], label='ワイン', color='#ff1493')
ax.plot(df_amount_cat['visit_on_month'], df_amount_cat['drink'], label='飲み物', color='#b0e0e6')
ax.plot(df_amount_cat['visit_on_month'], df_amount_cat['food'], label='食べ物', color='#7cfc00')
ax.plot(df_amount_cat['visit_on_month'], df_amount_cat['dessert'], label='デザート', color='#dda0dd')
ax.set_xlabel('年月')
ax.set_ylabel('売上', rotation='horizontal')
ax.set_title('カテゴリ別の売上推移')
ax.legend()
plt.grid(color = "gray", linestyle="--")
plt.show() 

# %%
# 日毎の来店グループ数と人件費の関係を平日休日を含めて確認したい
#データ加工
df_account2 = df_account.copy()

df_account2['visitday'] = pd.to_datetime(df_account2['visit_on']).dt.strftime('%Y%m%d')
#print(df_account2.sort_values('visit_on').head(40))
df_account3 = df_account2.groupby('visitday').agg({'payment_id':'count'}).reset_index()
#デバック用コード
#print(df_account3.head)
#print(df_account3.dtypes)
#データ型の変換
df_account3['visitday'] = pd.to_datetime(df_account3['visitday'])
#曜日の追加
df_account3['week_of_day'] = pd.to_datetime(df_account3['visitday']).dt.strftime('%A')
df_account3['week_of_number'] = pd.to_datetime(df_account3['visitday']).dt.weekday
print(df_account3.head())
# %%
#平日と土日祝を判別
import jpholiday
df_account3['holiday'] = df_account3.apply(lambda x:1 if x['week_of_number']>=5 or jpholiday.is_holiday(x['visitday'].date()) else 0, axis=1)
df_account3 = df_account3.rename(columns={'payment_id':'coming_number'})
print(df_account3.head())
# %%
# 決済データと人件費データを日時で結合
df_pex2 = df_pex.copy()
df_pex2 = df_pex2.rename(columns={'work_on':'visitday'})
#print(df_pex2.head())
df_pex2['visitday'] = pd.to_datetime(df_pex2['visitday'])
df_pen = pd.merge(df_account3, df_pex2, how='inner', on='visitday')
print(df_pen.head())
# %%
# 来店グループ数と人件費についてまとめる
df_pen['humanrate'] = df_pen['personnel_expense'] / df_pen['coming_number']
print(df_pen.head())
# %%
#可視化
plt.figure(figsize=(12, 6))
# 折れ線グラフの描画
plt.plot(df_pen['visitday'], df_pen['humanrate'], label='Human Rate', color='blue')
# 祝日の背景色を変更
for _, row in df_pen.iterrows():
    if row['holiday'] == 1:  # 祝日や土日
        plt.axvspan(row['visitday'], row['visitday'], color='red', alpha=0.8)
plt.title('')
plt.xlabel('日付')
plt.ylabel('人件費割合', rotation='horizontal')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
# %%
#step2 
#仮説３
#以前のデータ加工のコードの再喝
#カテゴリ別の売上を、月ごとに確認したい
##カテゴリーのダミー変数化＋集計
df_order2 = df_order.copy()
df_order2 = pd.get_dummies(df_order2, columns=['category'], dtype=int)
df_order2['coffee_sum_amount'] = df_order2['total_price'] * df_order2['category_coffee'] 
df_order2['dessert_sum_amount'] = df_order2['total_price'] * df_order2['category_dessert']
df_order2['drink_sum_amount'] = df_order2['total_price'] * df_order2['category_drink']
df_order2['food_sum_amount'] = df_order2['total_price'] * df_order2['category_food']
df_order2['wine_sum_amount'] = df_order2['total_price'] * df_order2['category_wine']
df_order3 = df_order2.groupby('payment_id').agg({'coffee_sum_amount':'sum',
                                                 'dessert_sum_amount':'sum',
                                                 'drink_sum_amount':'sum',
                                                 'food_sum_amount':'sum',
                                                 'wine_sum_amount':'sum'}).reset_index()
                                                
print(df_order3.head())
# %%
##決済データの決済日から決済月を出して結合
df_account2 = df_account.copy()

df_account2['visit_on_month'] = pd.to_datetime(df_account2['visit_on']).dt.strftime('%Y%m')

df_amount_cat = pd.merge(df_order3, df_account2[['payment_id', 'visit_on_month']], how='inner', on='payment_id')
#print(df_amount_cat.head())
df_amount_cat = df_amount_cat.groupby('visit_on_month').agg(coffee=('coffee_sum_amount','sum'),
                                                            dessert=('dessert_sum_amount','sum'),
                                                            drink=('drink_sum_amount','sum'),
                                                            food=('food_sum_amount','sum'),
                                                            wine=('wine_sum_amount','sum')).reset_index()
print(df_amount_cat.head())
# %%
# 気象データを持ってくる
df_weather = pd.read_csv(r"C:\Users\toraL\dirisl_phase1\dataset\phase1_data\miura_weatherdata.csv", dtype_backend='numpy_nullable', encoding='utf-8')
df_weather = df_weather.drop(df_weather.index[[0,1,3]]).reset_index()
df_weather = df_weather.drop(df_weather.columns[[0]], axis=1)
target_string = ['品質情報', '均質番号']
df_weather = df_weather.loc[:,~df_weather.iloc[1].isin(target_string)]
df_weather = df_weather.drop(df_weather.index[[1]]).reset_index()
df_weather.columns = df_weather.iloc[0]
df_weather = df_weather.iloc[1:]
df_weather = df_weather.drop(df_weather.columns[[0]], axis=1)
df_weather = df_weather.drop(df_weather.columns[[6,7]], axis=1)
print(df_weather.head())
# %%
print(df_weather.info())
# %%
df_weather1 = df_weather.copy()
df_weather1['年月'] = pd.to_datetime(df_weather1['年月'], format='%b-%y').dt.strftime('%Y%m')
df_weather1 = df_weather1.rename(columns={'年月':'visit_on_month'})
print(df_weather1.head(2))
# %%
#結合
df_hyp3 = pd.merge(df_amount_cat, 
                   df_weather1[['visit_on_month', '平均気温(℃)']],
                   how='inner',
                   on='visit_on_month')
print(df_hyp3.head())
# %%
#df_hyp3['visit_on_month'] = pd.to_datetime(df_hyp3['visit_on_month'])
df_hyp3['平均気温(℃)'] = pd.to_numeric(df_hyp3['平均気温(℃)'], errors='coerce')
print(df_hyp3.info())
# %%
# カテゴリー別売上
fig, ax1 = plt.subplots()
fig.autofmt_xdate()#x軸を見やすくする
ax1.bar(df_hyp3['visit_on_month'], df_hyp3['coffee'], color='#8b0000', label='コーヒー')
ax1.bar(df_hyp3['visit_on_month'], df_hyp3['wine'], color='#ff69b4', label='ワイン', bottom=df_hyp3['coffee'])
ax1.bar(df_hyp3['visit_on_month'], df_hyp3['drink'], color='#ff4500', label='その他飲み物', bottom=df_hyp3['coffee']+df_hyp3['wine'])
ax1.bar(df_hyp3['visit_on_month'], df_hyp3['food'], color='#1e90ff', label='食べ物', bottom=df_hyp3['coffee']+df_hyp3['wine']+df_hyp3['drink'])
ax1.bar(df_hyp3['visit_on_month'], df_hyp3['dessert'],color='#191970', label='デザート', bottom=df_hyp3['coffee']+df_hyp3['wine']+df_hyp3['drink']+df_hyp3['food'])
ax1.set_ylabel('売上', rotation='horizontal')
ax1.set_yticks([500000, 1000000, 1500000, 2000000, 2500000])
ax1.set_yticklabels(['50万', '100万', '150万', '200万', '250万'])
ax1.tick_params(axis='y', labelcolor='#696969')

# 平均気温
ax2 = ax1.twinx()
ax2.plot(df_hyp3['visit_on_month'], df_hyp3['平均気温(℃)'], color='#ffd700', marker='o')
ax2.set_ylabel('平均気温(℃)', color='black', rotation='horizontal')
ax2.set_yticks( np.arange(0, 41, 4))
ax2.tick_params(axis='y', labelcolor='#696969')

plt.xlabel('年月')
plt.title('月毎カテゴリー別売上と平均気温')
plt.xticks(rotation=45)  # X軸ラベルの回転
plt.tight_layout()
plt.grid(axis='y', color='#a9a9a9', linestyle='--')
fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
plt.show()
# %%
# Appendix カテゴリごとの月毎の売上高の変動率（初月基準）
print(df_amount_cat.info())
# %%
df_amount_cat2 = df_amount_cat.copy()
print(df_amount_cat2.head())
# %%
for column in ['coffee', 'dessert', 'drink', 'food', 'wine']:
    initial_value = df_amount_cat2.loc[0, column]  # 初月の値を取得
    df_amount_cat2[column] = df_amount_cat2[column].apply(lambda x: x / initial_value)

# 結果を確認
print(df_amount_cat2.head())
# %%
# %%
#カテゴリ別の売上を折れ線グラフで可視化

fig, ax = plt.subplots()
fig.autofmt_xdate()#x軸を見やすくする
ax.plot(df_amount_cat2['visit_on_month'], df_amount_cat2['coffee'], label='コーヒー', color='#8b0000')
ax.plot(df_amount_cat2['visit_on_month'], df_amount_cat2['wine'], label='ワイン', color='#ff69b4')
ax.plot(df_amount_cat2['visit_on_month'], df_amount_cat2['drink'], label='飲み物', color='#ff4500')
ax.plot(df_amount_cat2['visit_on_month'], df_amount_cat2['food'], label='食べ物', color='#1e90ff')
ax.plot(df_amount_cat2['visit_on_month'], df_amount_cat2['dessert'], label='デザート', color='#191970')
ax.set_xlabel('年月')
ax.set_ylabel('売上', rotation='horizontal')
ax.set_title('カテゴリ別の売上推移')
ax.legend()
plt.grid(color = "gray", linestyle="--")
plt.show() 

# %%
# 日毎の来店グループ数と人件費の関係を平日休日を含めて確認したい
#データ加工
df_account2 = df_account.copy()

df_account2['visitday'] = pd.to_datetime(df_account2['visit_on']).dt.strftime('%Y%m%d')
#print(df_account2.sort_values('visit_on').head(40))
df_account3 = df_account2.groupby('visitday').agg({'payment_id':'count'}).reset_index()
#デバック用コード
#print(df_account3.head)
#print(df_account3.dtypes)
#データ型の変換
df_account3['visitday'] = pd.to_datetime(df_account3['visitday'])
#曜日の追加
df_account3['week_of_day'] = pd.to_datetime(df_account3['visitday']).dt.strftime('%A')
df_account3['week_of_number'] = pd.to_datetime(df_account3['visitday']).dt.weekday
#print(df_account3.head())
#平日と土日祝を判別
import jpholiday
df_account3['holiday'] = df_account3.apply(lambda x:1 if x['week_of_number']>=5 or jpholiday.is_holiday(x['visitday'].date()) else 0, axis=1)
df_account3 = df_account3.rename(columns={'payment_id':'coming_number'})
print(df_account3.head())
# %%
# 決済データと人件費データを日時で結合
df_pex2 = df_pex.copy()
df_pex2 = df_pex2.rename(columns={'work_on':'visitday'})
#print(df_pex2.head())
df_pex2['visitday'] = pd.to_datetime(df_pex2['visitday'])
df_pen = pd.merge(df_account3, df_pex2, how='inner', on='visitday')
print(df_pen.head())
# %%
# 来店グループ数と人件費についてまとめる
df_pen['humanrate'] = df_pen['personnel_expense'] / df_pen['coming_number']
df_pen['per_expense'] = df_pen['coming_number'] / df_pen['personnel_expense']
print(df_pen.head())
# %%
# 可視化
plt.figure(figsize=(12,6))
plt.hist(df_pen.loc[df_pen['holiday']==0]['humanrate'], alpha=0.5, bins=50, label='平日', color='blue')
plt.hist(df_pen.loc[df_pen['holiday']==1]['humanrate'], alpha=0.3, bins=50, label='休日', color='red')
#sns.histplot(data=df_pen, x='humanrate', kde=True, )
plt.xlabel('人件費/来店グループ数')
plt.ylabel('日数', rotation='horizontal')
plt.legend(title='曜日', labels=['平日', '休日'])
plt.title('来店グループ数と人件費の関係')
plt.show()
# %%
plt.figure(figsize=(12,6))
sns.histplot(data=df_pen, x='humanrate', stat='count',
             hue='holiday', multiple='stack', alpha=0.5, palette=['blue','red'])

plt.xlabel('人件費/来店グループ数')
plt.ylabel('日数', rotation='horizontal')
plt.legend(title='曜日', labels=['休日', '平日'])
plt.title('来店グループ数と人件費の関係')

plt.show()
# %%
# Appendix 
# 日毎の来店グループ数と人件費の関係を平日休日を含めて確認したい
#データ加工
df_account2 = df_account.copy()

df_account2['visitday'] = pd.to_datetime(df_account2['visit_on']).dt.strftime('%Y%m%d')
#print(df_account2.sort_values('visit_on').head(40))
print(df_account2.head())
# %%
df_account3 = df_account2.groupby('visitday').agg({'payment_id':'count'}).reset_index()
#デバック用コード
#print(df_account3.head)
#print(df_account3.dtypes)
#データ型の変換
df_account3['visitday'] = pd.to_datetime(df_account3['visitday'])
#曜日の追加
df_account3['week_of_day'] = pd.to_datetime(df_account3['visitday']).dt.strftime('%A')
df_account3['week_of_number'] = pd.to_datetime(df_account3['visitday']).dt.weekday
#print(df_account3.head())
#平日と土日祝を判別
import jpholiday
df_account3['holiday'] = df_account3.apply(lambda x:1 if x['week_of_number']>=5 or jpholiday.is_holiday(x['visitday'].date()) else 0, axis=1)
df_account3 = df_account3.rename(columns={'payment_id':'coming_number'})
print(df_account3.head())
# %%
# 曜日ごとに来店グループ数の平均を集計
df_account4 = df_account3.groupby('week_of_day').agg({'coming_number':'mean'})
df_account4 = df_account4.reindex(index=['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
print(df_account4)
# %%
# 月ごと
df_account3['month'] = pd.to_datetime(df_account3['visitday']).dt.strftime('%Y%m')
print(df_account3.head())
# %%
df_account5 = df_account3.groupby('month').agg({'coming_number':'mean'})
print(df_account5)
# %%
# 曜日別可視化
plt.figure(figsize=(12,6))
plt.plot(df_account4.index, df_account4['coming_number'], color='red', marker='o')
plt.xlabel('曜日')
plt.ylabel('来店者グループ数', rotation='horizontal')
plt.title('来店グループ数と曜日の推移')
plt.show()
# %%
#print(df_account4.columns)
# %%
plt.figure(figsize=(12,6))
plt.plot(df_account5.index, df_account5['coming_number'], color='navy', marker='o')
plt.xlabel('月')
plt.ylabel('来店者グループ数', rotation='horizontal')
plt.title('来店グループ数と月の推移')
plt.show()
# %%
# 来店時間帯も組み込む
df_account2 = df_account.copy()

df_account2['visitday'] = pd.to_datetime(df_account2['visit_on']).dt.strftime('%Y%m%d')
#print(df_account2.sort_values('visit_on').head(40))
print(df_account2.head())
# %%
print(df_account2.info())
# %%
df_account10 = df_account2.copy()
#デバック用コード
#print(df_account10.head())
#print(df_account3.dtypes)
#データ型の変換
#df_account10['visitday'] = pd.to_datetime(df_account3['visitday'])
# 
df_account10['visit_day_hour'] = pd.to_datetime(df_account10['order_datetime']).dt.strftime('%Y%m%d%H')
print(df_account10.head())
# %%
df_coming = df_account10.groupby('visit_day_hour').agg({'payment_id':'count'}).reset_index()
df_coming = df_coming.rename(columns={'payment_id':'coming_number'})
print(df_coming)
# %%
df_coming['visit_day_hour'] = pd.to_datetime(df_coming['visit_day_hour'], format='%Y%m%d%H')
df_coming['visit_day'] = pd.to_datetime(df_coming['visit_day_hour']).dt.strftime('%Y%m%d')
df_coming['visit_hour'] = pd.to_datetime(df_coming['visit_day_hour']).dt.strftime('%H')
df_coming['visit_month'] = pd.to_datetime(df_coming['visit_day_hour']).dt.strftime('%Y%m')
df_coming['visit_month_hour'] = pd.to_datetime(df_coming['visit_day_hour']).dt.strftime('%Y%m%H')
print(df_coming.head())
# %%
df_coming2 = df_coming.groupby('visit_day').agg({'coming_number':'sum'}).reset_index()
df_coming2['visit_month'] = pd.to_datetime(df_coming2['visit_day']).dt.strftime('%Y%m')
print(df_coming2.head(30))
# %%
#df_coming_month = pd.merge(df_coming[['visit_day', 'visit_month']], df_coming2, how='inner', on='visit_day')
#print(df_coming_month.head())
# %%
df_coming_month =df_coming2.copy()
df_coming_month = df_coming_month.groupby('visit_month').agg({'coming_number':['sum','mean']}).reset_index()
print(df_coming_month)
# %%
df_coming3 = df_coming.copy()
df_coming_month_hour = df_coming3.groupby('visit_month_hour').agg({'coming_number':'sum'}).reset_index()
df_coming_month_hour['visit_month'] = pd.to_datetime(df_coming_month_hour['visit_month_hour']).dt.strftime('%Y%m')
print(df_coming_month_hour.head(10))
# %%
df_