# 卡爾曼濾波（Kalman Filter)

在我目前所有的文章中使用的策略都是運用一個固定避險比率去做資產配置的。這個方法很好控制我們的投入資本，策略也相對簡單易寫。但他有一個問題是當我們遇到一個市場性的變動時，協整性也許會被打破，造成我們本來均值回歸的資產配置失去它原有的特質，變得難以預測，投資風險因此上升。解決這個問題的辦法有兩種：第一個就是用像是移動平均或是標準差類似的回觀期去找協整避險比率。在書中，Ernie Chan 說道這個方法的缺點就是每一次更新的回觀期如果比較短，因為最新的價格的增添以及最久以前價格的捨去，可能會造成避險比率的隨意大幅度變動，而這些變動是沒有實質上的意義，對部位的控管也會比較困難。這個方法，因為和之前的策略重複性較高，就不在這邊從事實作。第二個方法，就是卡爾曼濾波。

卡爾曼濾波是一個線性的優化程式。它會依照一個可觀察到的最新數值去更新一個無法觀察到的數值。其中的的數學真的不簡單了，我也不在這邊做太多解釋，有興趣研究細節可以去看參考書籍 P.74，或是[這個連結](https://youtu.be/mwn8xhgNpFY?si=5e7zWdLmfObe6GPL)。如果我們設定 $y(t)$ 是一個價格序列（在這篇文章中他是 ```00635U```），$x(t)$ 是另外一個價格序列（在這裏他代表 ```00642U```），$\beta(t)$ 是一個 $2\times n$ 的矩陣，第一行代表變動的避險比率，第二行代表變動的回歸常數。卡爾曼濾波會有以下關係式：

* $y(t) = x(t)\beta(t) + \epsilon (t)$ (Measurement equation)：在這裡，$x(t)$ 是向量 $\langle x(t), 1\rangle$，左邊是當下價格。$\epsilon(t)$ 是當日常態分佈誤差，方差為 $V_{\epsilon}$。
* $\beta(t) = \beta(t-1) + \omega(t-1)$ (State transition)：$\omega(t-1)$ 也是當日誤差，有共變異數 $V_{\omega}$。

接下來，我們就有六個迭代方程式：
* 預測：
    * $\hat{\beta}(t|t-1) = \hat{\beta}(t-1|t-1)$ (State prediction) (1)
    * $ R(t|t-1) = R(t-1|t-1) + V_{\omega}$ (State covariance prediction) (2) ($R(t|t-1) = cov(\beta(t)-\hat{\beta}(t|t-1))$)
    * $ \hat{y}(t) = x(t)\hat{\beta}(t|t-1)$ (Measurement prediction) (3)
    * $ Q(t) = x(t)R(t|t-1)x(t)' + V_{\epsilon}$ (Measurement variance prediction) (4)
* 更新：
    * $\hat{\beta}(t|t) = \hat{\beta}(t|t-1) + K(t)*e(t)$ (State update) ($K(t) = R(t|t-1)*x(t)/Q(t)$) (5)
    * $R(t|t) = R(t|t-1)-K(t)*x(t)*R(t|t-1)$ (State covariance update) (6)

而初始化設定會是 $\hat{\beta}(1|0), R(0|0) = 0$，至於 $V_{\omega}, V_{\epsilon}$，得去調整得到最優數值。




這次用的案例是使用期元大S&P黃金以及期元大S&P石油。我抓了2019年初到2023年底的資料進行策略優化，打算在2024年的歷史資訊上做回測。


```python
from fugle_marketdata import RestClient
import pandas as pd

client = RestClient(api_key = api_key.key) # api_key.key 填入你的富果 RestAPI 鑰匙（api_key = '你的鑰匙')
stock = client.stock

def get_data(symbol, year):
    s = stock.historical.candles(**{"symbol": symbol,
                                  "from": f"{year}-01-01",
                                  "to": f"{year}-12-31",
                                  "fields": "close",
                                  'sort': 'asc'})
    f = pd.DataFrame.from_dict(s['data'])
    f['date'] = pd.to_datetime(f['date'])
    f.set_index('date', inplace=True)
    f.rename(columns={'close':f'{symbol}'}, inplace=True)
    return f

s1 = '00635U'
s2 = '00642U'

s1_19 = get_data(s1, 2019)
s2_19 = get_data(s2, 2019)
s1_20 = get_data(s1, 2020)
s2_20 = get_data(s2, 2020)
s1_21 = get_data(s1, 2021)
s2_21 = get_data(s2, 2021)
s1_22 = get_data(s1, 2022)
s2_22 = get_data(s2, 2022)
s1_23 = get_data(s1, 2023)
s2_23 = get_data(s2, 2023)

df1 = pd.concat([s1_19, s1_20, s1_21, s1_22, s1_23])
df2 = pd.concat([s2_19, s2_20, s2_21, s2_22, s2_23])
df = pd.merge(df1, df2, on='date', how='inner')
```

兩者走勢如下：


```python
df.plot()
```

    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)





    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_1.png)
    



```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

results_johansen = coint_johansen(df, 0, 1)
```


```python
print(results_johansen.trace_stat)
print(results_johansen.trace_stat_crit_vals)
print(results_johansen.max_eig_stat)
print(results_johansen.max_eig_stat_crit_vals)
```

    [11.99777621  2.20965211]
    [[13.4294 15.4943 19.9349]
     [ 2.7055  3.8415  6.6349]]
    [9.7881241  2.20965211]
    [[12.2971 14.2639 18.52  ]
     [ 2.7055  3.8415  6.6349]]



```python
print(results_johansen.eig)
print(results_johansen.evec)
```

    [0.00803029 0.00181849]
    [[ 0.56741343 -0.12508233]
     [ 0.18202617  0.21932149]]



```python
df['port'] = 0.56741343*df[s1] + 0.18202617*df[s2]
df_BB = df.copy()
df.plot.line(y='port')
```




    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_2.png)
    


其實這篇裡面找的例子本來是希望可以使用較不具有協整性的案例經過卡爾曼濾波的自動避險比率調整出一個均值回歸的資產序列。誰知道他們倆的走勢也是具有相當的協整性。不過是有明顯向上飄移的趨勢。也許卡爾曼濾波可以幫助我們去除這個問題。


```python
from statsmodels.tsa.stattools import adfuller
adfuller(df['port'])
```




    (-2.7504610258468536,
     0.06570587770459516,
     8,
     1207,
     {'1%': -3.43577938005948,
      '5%': -2.863937543790164,
      '10%': -2.568046493171221},
     -1011.3774379817146)



開始使用卡爾曼濾波。在這裏我找到最合適的 $V_{\epsilon}$ 是0.1，而若設定 $V_{\omega} = \frac{\delta}{1-\delta}I$，最合適的 $\delta$ 是 0.0001。 對於這兩個數值的設定，書本說道有一個方法叫做 Autocovariance Least Squares，但因為太複雜且沒有套件，目前沒有能力去執行。


```python
import numpy as np
y = np.array(df[s1])
x = list(df[s2])
x = np.array([[i, 1] for i in x])
delta = 0.0001
yhat = np.array([np.nan for _ in range(len(y))])
e = np.array([np.nan for _ in range(len(y))])
Q = np.array([np.nan for _ in range(len(y))])
P = np.array(np.zeros([2, 2]))
beta = np.array([[np.nan for _ in range(len(y))], [np.nan for _ in range(len(y))]])
Vw = delta/(1-delta)*np.diag(np.ones(2))
Ve = 0.1
```


```python
beta[:,0] = 0
for i in range(1, len(y)):
    beta[:,i] = beta[:,i-1] # (1)
    R = P + Vw # (2)

    yhat[i] = x[i,:]@beta[:, i] # (3)
    Q[i] = x[i,:]@R@x[i,:].transpose() + Ve # (4)
    e[i] = y[i]-yhat[i]
    K = R@x[i,:].transpose()/Q[i]
    beta[:,i] = beta[:,i] + K*e[i] # (5)
    P = R - np.outer(K, x[i,:]@R) #(6)

initial_24 = beta[:,-1]
```


```python
beta
```




    array([[0.        , 0.23159211, 0.53780844, ..., 1.08509824, 1.0954453 ,
            1.10651198],
           [0.        , 0.01577603, 0.03588128, ..., 7.29101776, 7.29375062,
            7.30208259]])




```python
df['slope'] = beta[0,:]
df['intercept'] = beta[1,:]
df['control_port'] = df[s1] + df['slope']*df[s2]
```

經過每天避險比率的調整，我們的資產配置如下。可以看到，運來的向上飄移不見了，我們得到的是一個非常有均值回歸的資產線。


```python
df.plot.line(y='control_port')
```




    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_3.png)
    


回歸常數圖示如下：


```python
df.plot.line(y='intercept')
```




    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_4.png)
    


避險比率如下。在這裏，我們的期元大S&P黃金比例永遠是1。經過石油的去調整。


```python
df.plot.line(y='slope')
```




    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_5.png)
    



```python
adfuller(df['control_port'][1:])
```




    (-2.241308686274083,
     0.19159570282211802,
     19,
     1195,
     {'1%': -3.4358340188169882,
      '5%': -2.8639616494853217,
      '10%': -2.5680593310691338},
     1729.5694361045662)




```python
from Disillusion_v1_0 import *

df['sqrt_Q'] = np.sqrt(Q)
df['e'] = e
df['Q_up'] = 2*df['sqrt_Q']
df['Q_low'] = -2*df['sqrt_Q']
df.plot.line(y=['Q_up', 'e', 'Q_low'], ylim=[-4, 8])
```

    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)
    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)





    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_6.png)
    


有了這些結果後，要怎麼寫出策略呢？可以發現e代表的是每天的預測誤差值。而Q是這個誤差值的方差。根號Q就可以成為我們的交易訊號門檻。若把e跟根號Q的正負兩倍畫在圖表上，我們可以做出一個類似布林通道的圖形。接下來就不難了，寫寫策略，結果意外的亮眼！5年46%，年化利率為7.86%。不錯！當然，在這個狀況下我們的股票張數沒有整數單位，但這不是太大的問題。


```python
c = 2

class KF(Strategy):
    def __init__(self):
        super().__init__()
        self.active_position = None
        self.position_size = 0


    def on_bar(self):
        if self.data.loc[self.current_idx, 'e'] <= -c*self.data.loc[self.current_idx, 'sqrt_Q'] and not self.active_position:
            self.buy(s1, 1)
            self.buy(s2, self.data.loc[self.current_idx, 'slope'])
            self.active_position = 'long'
            self.position_size = self.data.loc[self.current_idx, 'slope']
        elif self.data.loc[self.current_idx, 'e'] >= c*self.data.loc[self.current_idx, 'sqrt_Q'] and not self.active_position:
            self.short_in(s1, 1)
            self.short_in(s2, self.data.loc[self.current_idx, 'slope'])
            self.active_position = 'short'
            self.position_size = self.data.loc[self.current_idx, 'slope']
        elif self.data.loc[self.current_idx, 'e'] >= c*self.data.loc[self.current_idx, 'sqrt_Q'] and self.active_position == 'long':
            self.sell(s1, 1)
            self.sell(s2, self.position_size)
            self.short_in(s1, 1)
            self.short_in(s2, self.data.loc[self.current_idx, 'slope'])
            self.active_position = 'short'
            self.position_size = self.data.loc[self.current_idx, 'slope']
        elif self.data.loc[self.current_idx, 'e'] <= -c*self.data.loc[self.current_idx, 'sqrt_Q'] and self.active_position == 'short':
            self.short_out(s1, 1)
            self.short_out(s2, self.position_size)
            self.buy(s1, 1)
            self.buy(s2, self.data.loc[self.current_idx, 'slope'])
            self.active_position = 'long'
            self.position_size = self.data.loc[self.current_idx, 'slope']
        else:
            pass

E = Engine(initial_cash=100000)
E.add_data(df)
E.add_strategy(KF())
E.run()
```

    100%|██████████| 1216/1216 [00:00<00:00, 9358.28it/s]

    Short Executed for 1 shares of 00635U at 2019-01-03 00:00:00.
    Short Executed for 0.23159211129474916 shares of 00642U at 2019-01-03 00:00:00.
    Short Closed for 1 shares of 00635U at 2019-06-26 00:00:00.
    Short Closed for 0.23159211129474916 shares of 00642U at 2019-06-26 00:00:00.
    Buy Executed for 1 shares of 00635U at 2019-06-26 00:00:00.
    Buy Executed for 1.2050730667303693 shares of 00642U at 2019-06-26 00:00:00.
    Sell Executed for 1 shares of 00635U at 2019-07-03 00:00:00.
    Sell Executed for 1.2050730667303693 shares of 00642U at 2019-07-03 00:00:00.
    Short Executed for 1 shares of 00635U at 2019-07-03 00:00:00.
    Short Executed for 1.1993221633339175 shares of 00642U at 2019-07-03 00:00:00.
    Short Closed for 1 shares of 00635U at 2019-09-10 00:00:00.
    Short Closed for 1.1993221633339175 shares of 00642U at 2019-09-10 00:00:00.
    Buy Executed for 1 shares of 00635U at 2019-09-10 00:00:00.
    Buy Executed for 1.2999167842326227 shares of 00642U at 2019-09-10 00:00:00.
    Sell Executed for 1 shares of 00635U at 2019-09-25 00:00:00.
    Sell Executed for 1.2999167842326227 shares of 00642U at 2019-09-25 00:00:00.
    Short Executed for 1 shares of 00635U at 2019-09-25 00:00:00.
    Short Executed for 1.283459570978833 shares of 00642U at 2019-09-25 00:00:00.
    Short Closed for 1 shares of 00635U at 2020-04-30 00:00:00.
    Short Closed for 1.283459570978833 shares of 00642U at 2020-04-30 00:00:00.
    Buy Executed for 1 shares of 00635U at 2020-04-30 00:00:00.
    Buy Executed for 2.7820204156076733 shares of 00642U at 2020-04-30 00:00:00.
    Sell Executed for 1 shares of 00635U at 2020-06-11 00:00:00.
    Sell Executed for 2.7820204156076733 shares of 00642U at 2020-06-11 00:00:00.
    Short Executed for 1 shares of 00635U at 2020-06-11 00:00:00.
    Short Executed for 2.408512317388625 shares of 00642U at 2020-06-11 00:00:00.
    Short Closed for 1 shares of 00635U at 2020-08-12 00:00:00.
    Short Closed for 2.408512317388625 shares of 00642U at 2020-08-12 00:00:00.
    Buy Executed for 1 shares of 00635U at 2020-08-12 00:00:00.
    Buy Executed for 3.018565486554029 shares of 00642U at 2020-08-12 00:00:00.
    Sell Executed for 1 shares of 00635U at 2020-09-01 00:00:00.
    Sell Executed for 3.018565486554029 shares of 00642U at 2020-09-01 00:00:00.
    Short Executed for 1 shares of 00635U at 2020-09-01 00:00:00.
    Short Executed for 3.039884729934385 shares of 00642U at 2020-09-01 00:00:00.
    Short Closed for 1 shares of 00635U at 2020-11-10 00:00:00.
    Short Closed for 3.039884729934385 shares of 00642U at 2020-11-10 00:00:00.
    Buy Executed for 1 shares of 00635U at 2020-11-10 00:00:00.
    Buy Executed for 3.3185539620135622 shares of 00642U at 2020-11-10 00:00:00.
    Sell Executed for 1 shares of 00635U at 2021-01-05 00:00:00.
    Sell Executed for 3.3185539620135622 shares of 00642U at 2021-01-05 00:00:00.
    Short Executed for 1 shares of 00635U at 2021-01-05 00:00:00.
    Short Executed for 2.836194568847844 shares of 00642U at 2021-01-05 00:00:00.
    Short Closed for 1 shares of 00635U at 2021-01-07 00:00:00.
    Short Closed for 2.836194568847844 shares of 00642U at 2021-01-07 00:00:00.
    Buy Executed for 1 shares of 00635U at 2021-01-07 00:00:00.
    Buy Executed for 2.796663398548491 shares of 00642U at 2021-01-07 00:00:00.
    Sell Executed for 1 shares of 00635U at 2021-03-19 00:00:00.
    Sell Executed for 2.796663398548491 shares of 00642U at 2021-03-19 00:00:00.
    Short Executed for 1 shares of 00635U at 2021-03-19 00:00:00.
    Short Executed for 1.8851062652187287 shares of 00642U at 2021-03-19 00:00:00.
    Short Closed for 1 shares of 00635U at 2021-03-30 00:00:00.
    Short Closed for 1.8851062652187287 shares of 00642U at 2021-03-30 00:00:00.
    Buy Executed for 1 shares of 00635U at 2021-03-30 00:00:00.
    Buy Executed for 1.9585639212988464 shares of 00642U at 2021-03-30 00:00:00.
    Sell Executed for 1 shares of 00635U at 2021-04-22 00:00:00.
    Sell Executed for 1.9585639212988464 shares of 00642U at 2021-04-22 00:00:00.
    Short Executed for 1 shares of 00635U at 2021-04-22 00:00:00.
    Short Executed for 1.9678395910367485 shares of 00642U at 2021-04-22 00:00:00.
    Short Closed for 1 shares of 00635U at 2021-05-05 00:00:00.
    Short Closed for 1.9678395910367485 shares of 00642U at 2021-05-05 00:00:00.
    Buy Executed for 1 shares of 00635U at 2021-05-05 00:00:00.
    Buy Executed for 1.9173711541369671 shares of 00642U at 2021-05-05 00:00:00.
    Sell Executed for 1 shares of 00635U at 2021-05-20 00:00:00.
    Sell Executed for 1.9173711541369671 shares of 00642U at 2021-05-20 00:00:00.
    Short Executed for 1 shares of 00635U at 2021-05-20 00:00:00.
    Short Executed for 1.9867153770064956 shares of 00642U at 2021-05-20 00:00:00.
    Short Closed for 1 shares of 00635U at 2021-06-04 00:00:00.
    Short Closed for 1.9867153770064956 shares of 00642U at 2021-06-04 00:00:00.
    Buy Executed for 1 shares of 00635U at 2021-06-04 00:00:00.
    Buy Executed for 1.9283238031267609 shares of 00642U at 2021-06-04 00:00:00.
    Sell Executed for 1 shares of 00635U at 2021-07-20 00:00:00.
    Sell Executed for 1.9283238031267609 shares of 00642U at 2021-07-20 00:00:00.
    Short Executed for 1 shares of 00635U at 2021-07-20 00:00:00.
    Short Executed for 1.737721744387592 shares of 00642U at 2021-07-20 00:00:00.
    Short Closed for 1 shares of 00635U at 2021-09-16 00:00:00.
    Short Closed for 1.737721744387592 shares of 00642U at 2021-09-16 00:00:00.
    Buy Executed for 1 shares of 00635U at 2021-09-16 00:00:00.
    Buy Executed for 1.6803735807762243 shares of 00642U at 2021-09-16 00:00:00.
    Sell Executed for 1 shares of 00635U at 2021-11-18 00:00:00.
    Sell Executed for 1.6803735807762243 shares of 00642U at 2021-11-18 00:00:00.
    Short Executed for 1 shares of 00635U at 2021-11-18 00:00:00.
    Short Executed for 1.518809120294583 shares of 00642U at 2021-11-18 00:00:00.
    Short Closed for 1 shares of 00635U at 2021-11-24 00:00:00.
    Short Closed for 1.518809120294583 shares of 00642U at 2021-11-24 00:00:00.
    Buy Executed for 1 shares of 00635U at 2021-11-24 00:00:00.
    Buy Executed for 1.5060457797325215 shares of 00642U at 2021-11-24 00:00:00.
    Sell Executed for 1 shares of 00635U at 2021-11-29 00:00:00.
    Sell Executed for 1.5060457797325215 shares of 00642U at 2021-11-29 00:00:00.
    Short Executed for 1 shares of 00635U at 2021-11-29 00:00:00.
    Short Executed for 1.5314619999080086 shares of 00642U at 2021-11-29 00:00:00.
    Short Closed for 1 shares of 00635U at 2021-12-28 00:00:00.
    Short Closed for 1.5314619999080086 shares of 00642U at 2021-12-28 00:00:00.
    Buy Executed for 1 shares of 00635U at 2021-12-28 00:00:00.
    Buy Executed for 1.5477537759408466 shares of 00642U at 2021-12-28 00:00:00.
    Sell Executed for 1 shares of 00635U at 2022-03-10 00:00:00.
    Sell Executed for 1.5477537759408466 shares of 00642U at 2022-03-10 00:00:00.
    Short Executed for 1 shares of 00635U at 2022-03-10 00:00:00.
    Short Executed for 1.0380309133772214 shares of 00642U at 2022-03-10 00:00:00.
    Short Closed for 1 shares of 00635U at 2022-03-18 00:00:00.
    Short Closed for 1.0380309133772214 shares of 00642U at 2022-03-18 00:00:00.
    Buy Executed for 1 shares of 00635U at 2022-03-18 00:00:00.
    Buy Executed for 1.1150521019125332 shares of 00642U at 2022-03-18 00:00:00.
    Sell Executed for 1 shares of 00635U at 2022-03-31 00:00:00.
    Sell Executed for 1.1150521019125332 shares of 00642U at 2022-03-31 00:00:00.
    Short Executed for 1 shares of 00635U at 2022-03-31 00:00:00.
    Short Executed for 1.0556432419478956 shares of 00642U at 2022-03-31 00:00:00.
    Short Closed for 1 shares of 00635U at 2022-04-14 00:00:00.
    Short Closed for 1.0556432419478956 shares of 00642U at 2022-04-14 00:00:00.
    Buy Executed for 1 shares of 00635U at 2022-04-14 00:00:00.
    Buy Executed for 1.1056486258888572 shares of 00642U at 2022-04-14 00:00:00.
    Sell Executed for 1 shares of 00635U at 2022-06-20 00:00:00.
    Sell Executed for 1.1056486258888572 shares of 00642U at 2022-06-20 00:00:00.
    Short Executed for 1 shares of 00635U at 2022-06-20 00:00:00.
    Short Executed for 0.8660380101850497 shares of 00642U at 2022-06-20 00:00:00.
    Short Closed for 1 shares of 00635U at 2022-10-11 00:00:00.
    Short Closed for 0.8660380101850497 shares of 00642U at 2022-10-11 00:00:00.
    Buy Executed for 1 shares of 00635U at 2022-10-11 00:00:00.
    Buy Executed for 0.8881700714122712 shares of 00642U at 2022-10-11 00:00:00.
    Sell Executed for 1 shares of 00635U at 2022-11-09 00:00:00.
    Sell Executed for 0.8881700714122712 shares of 00642U at 2022-11-09 00:00:00.
    Short Executed for 1 shares of 00635U at 2022-11-09 00:00:00.
    Short Executed for 0.8511480763167013 shares of 00642U at 2022-11-09 00:00:00.
    Short Closed for 1 shares of 00635U at 2023-03-28 00:00:00.
    Short Closed for 0.8511480763167013 shares of 00642U at 2023-03-28 00:00:00.
    Buy Executed for 1 shares of 00635U at 2023-03-28 00:00:00.
    Buy Executed for 1.249544827074747 shares of 00642U at 2023-03-28 00:00:00.
    Sell Executed for 1 shares of 00635U at 2023-04-27 00:00:00.
    Sell Executed for 1.249544827074747 shares of 00642U at 2023-04-27 00:00:00.
    Short Executed for 1 shares of 00635U at 2023-04-27 00:00:00.
    Short Executed for 1.1562374793112045 shares of 00642U at 2023-04-27 00:00:00.
    Short Closed for 1 shares of 00635U at 2023-06-05 00:00:00.
    Short Closed for 1.1562374793112045 shares of 00642U at 2023-06-05 00:00:00.
    Buy Executed for 1 shares of 00635U at 2023-06-05 00:00:00.
    Buy Executed for 1.1870788441743396 shares of 00642U at 2023-06-05 00:00:00.
    Sell Executed for 1 shares of 00635U at 2023-10-12 00:00:00.
    Sell Executed for 1.1870788441743396 shares of 00642U at 2023-10-12 00:00:00.
    Short Executed for 1 shares of 00635U at 2023-10-12 00:00:00.
    Short Executed for 0.8670231384205948 shares of 00642U at 2023-10-12 00:00:00.


    



```python
results = df.copy()
results['mv'] = E.portfolio_mv[1:]
results.plot.line(y='mv')
```




    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_7.png)
    



```python
(E.portfolio_mv[-1]-E.portfolio_mv[0])/E.max_cash_deployed
```




    0.4600982758251972



## 使用布林通道做比對

這裏，我們使用正常布林通道用先前計算出來的避險比率做資產配置。使用上一篇文章中的交易策略進行參數調整。



```python
import statsmodels.api as sm
# Regression
d_port = np.diff(df_BB['port'])
port_wc = sm.add_constant(df_BB['port'][:-1])
model = sm.OLS(d_port, port_wc)
results = model.fit()
halflife = -np.log(2)/results.params[1]
print(halflife)
```

    47.78681384753512


    /var/folders/r8/c399k3cj0l59gk4_hz40yfy40000gn/T/ipykernel_73891/1511807467.py:7: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      halflife = -np.log(2)/results.params[1]



```python
df_BB['mv_avg'] = df_BB['port'].rolling(48).mean()
df_BB['mv_std'] = df_BB['port'].rolling(48).std()
df_BB['BB_up'] = df_BB['mv_avg'] + 2*df_BB['mv_std']
df_BB['BB_low'] = df_BB['mv_avg'] - 2*df_BB['mv_std']
df_BB.plot.line(y=['port', 'BB_low', 'mv_avg', 'BB_up'])
```

    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)
    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)
    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)





    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_8.png)
    



```python
class BB(Strategy):
    def __init__(self):
        super().__init__()
        self.active_position = None


    def on_bar(self):
        if self.data.loc[self.current_idx, 'port'] <= self.data.loc[self.current_idx, 'BB_low'] and not self.active_position:
            self.buy(s1, 5.6741343)
            self.buy(s2, 1.8202617)
            self.active_position = 'long'
        elif self.data.loc[self.current_idx, 'port'] >= self.data.loc[self.current_idx, 'BB_up'] and not self.active_position:
            self.short_in(s1, 5.6741343)
            self.short_in(s2, 1.8202617)
            self.active_position = 'short'
        elif self.data.loc[self.current_idx, 'port'] >= self.data.loc[self.current_idx, 'BB_up'] and self.active_position == 'long':
            self.sell(s1, 5.6741343)
            self.sell(s2, 1.8202617)
            self.short_in(s1, 5.6741343)
            self.short_in(s2, 1.8202617)
            self.active_position = 'short'
        elif self.data.loc[self.current_idx, 'port'] <= self.data.loc[self.current_idx, 'BB_low'] and self.active_position == 'short':
            self.short_out(s1, 5.6741343)
            self.short_out(s2, 1.8202617)
            self.buy(s1, 5.6741343)
            self.buy(s2, 1.8202617)
            self.active_position = 'long'
        else:
            pass

E = Engine(initial_cash=200000)
E.add_data(df_BB)
E.add_strategy(BB())
E.run()
```

      0%|          | 0/1216 [00:00<?, ?it/s]

    Buy Executed for 5.6741343 shares of 00635U at 2019-05-03 00:00:00.
    Buy Executed for 1.8202617 shares of 00642U at 2019-05-03 00:00:00.
    Sell Executed for 5.6741343 shares of 00635U at 2019-06-20 00:00:00.
    Sell Executed for 1.8202617 shares of 00642U at 2019-06-20 00:00:00.
    Short Executed for 5.6741343 shares of 00635U at 2019-06-20 00:00:00.
    Short Executed for 1.8202617 shares of 00642U at 2019-06-20 00:00:00.
    Short Closed for 5.6741343 shares of 00635U at 2020-03-13 00:00:00.
    Short Closed for 1.8202617 shares of 00642U at 2020-03-13 00:00:00.
    Buy Executed for 5.6741343 shares of 00635U at 2020-03-13 00:00:00.
    Buy Executed for 1.8202617 shares of 00642U at 2020-03-13 00:00:00.
    Sell Executed for 5.6741343 shares of 00635U at 2020-07-09 00:00:00.
    Sell Executed for 1.8202617 shares of 00642U at 2020-07-09 00:00:00.
    Short Executed for 5.6741343 shares of 00635U at 2020-07-09 00:00:00.
    Short Executed for 1.8202617 shares of 00642U at 2020-07-09 00:00:00.
    Short Closed for 5.6741343 shares of 00635U at 2020-09-24 00:00:00.
    Short Closed for 1.8202617 shares of 00642U at 2020-09-24 00:00:00.
    Buy Executed for 5.6741343 shares of 00635U at 2020-09-24 00:00:00.
    Buy Executed for 1.8202617 shares of 00642U at 2020-09-24 00:00:00.
    Sell Executed for 5.6741343 shares of 00635U at 2021-01-05 00:00:00.

    100%|██████████| 1216/1216 [00:00<00:00, 9473.79it/s]

    
    Sell Executed for 1.8202617 shares of 00642U at 2021-01-05 00:00:00.
    Short Executed for 5.6741343 shares of 00635U at 2021-01-05 00:00:00.
    Short Executed for 1.8202617 shares of 00642U at 2021-01-05 00:00:00.
    Short Closed for 5.6741343 shares of 00635U at 2021-02-19 00:00:00.
    Short Closed for 1.8202617 shares of 00642U at 2021-02-19 00:00:00.
    Buy Executed for 5.6741343 shares of 00635U at 2021-02-19 00:00:00.
    Buy Executed for 1.8202617 shares of 00642U at 2021-02-19 00:00:00.
    Sell Executed for 5.6741343 shares of 00635U at 2021-05-07 00:00:00.
    Sell Executed for 1.8202617 shares of 00642U at 2021-05-07 00:00:00.
    Short Executed for 5.6741343 shares of 00635U at 2021-05-07 00:00:00.
    Short Executed for 1.8202617 shares of 00642U at 2021-05-07 00:00:00.
    Short Closed for 5.6741343 shares of 00635U at 2021-08-09 00:00:00.
    Short Closed for 1.8202617 shares of 00642U at 2021-08-09 00:00:00.
    Buy Executed for 5.6741343 shares of 00635U at 2021-08-09 00:00:00.
    Buy Executed for 1.8202617 shares of 00642U at 2021-08-09 00:00:00.
    Sell Executed for 5.6741343 shares of 00635U at 2021-10-14 00:00:00.
    Sell Executed for 1.8202617 shares of 00642U at 2021-10-14 00:00:00.
    Short Executed for 5.6741343 shares of 00635U at 2021-10-14 00:00:00.
    Short Executed for 1.8202617 shares of 00642U at 2021-10-14 00:00:00.
    Short Closed for 5.6741343 shares of 00635U at 2022-07-01 00:00:00.
    Short Closed for 1.8202617 shares of 00642U at 2022-07-01 00:00:00.
    Buy Executed for 5.6741343 shares of 00635U at 2022-07-01 00:00:00.
    Buy Executed for 1.8202617 shares of 00642U at 2022-07-01 00:00:00.
    Sell Executed for 5.6741343 shares of 00635U at 2022-11-11 00:00:00.
    Sell Executed for 1.8202617 shares of 00642U at 2022-11-11 00:00:00.
    Short Executed for 5.6741343 shares of 00635U at 2022-11-11 00:00:00.
    Short Executed for 1.8202617 shares of 00642U at 2022-11-11 00:00:00.
    Short Closed for 5.6741343 shares of 00635U at 2023-06-29 00:00:00.
    Short Closed for 1.8202617 shares of 00642U at 2023-06-29 00:00:00.
    Buy Executed for 5.6741343 shares of 00635U at 2023-06-29 00:00:00.
    Buy Executed for 1.8202617 shares of 00642U at 2023-06-29 00:00:00.
    Sell Executed for 5.6741343 shares of 00635U at 2023-10-20 00:00:00.
    Sell Executed for 1.8202617 shares of 00642U at 2023-10-20 00:00:00.
    Short Executed for 5.6741343 shares of 00635U at 2023-10-20 00:00:00.
    Short Executed for 1.8202617 shares of 00642U at 2023-10-20 00:00:00.


    



```python
df_BB['mv'] = E.portfolio_mv[1:]
df_BB.plot.line(y='mv')
```




    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_9.png)
    



```python
(E.portfolio_mv[-1]-E.portfolio_mv[0])/E.max_cash_deployed
```




    0.21539773252486658




```python
E.max_cash_deployed
```




    149738.54950668436



5年區間測試結果總收益21.5%，年化收益為約5%。沒有布林通道的好。

## 交叉驗證

有了調整完的參數後，就可以拿2024年的歷史資訊做一個公平的回測了。


```python
s1_24 = get_data(s1, 2024)
s2_24 = get_data(s2, 2024)
cv_kf = pd.merge(s1_24, s2_24, on='date', how='inner')
cv_BB = cv_kf.copy()
cv_kf.plot()
```

    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)





    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_10.png)
    



```python
cv_kf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>00635U</th>
      <th>00642U</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-01-02</th>
      <td>24.98</td>
      <td>15.87</td>
    </tr>
    <tr>
      <th>2024-01-03</th>
      <td>24.92</td>
      <td>15.36</td>
    </tr>
    <tr>
      <th>2024-01-04</th>
      <td>24.68</td>
      <td>15.96</td>
    </tr>
    <tr>
      <th>2024-01-05</th>
      <td>24.68</td>
      <td>15.86</td>
    </tr>
    <tr>
      <th>2024-01-08</th>
      <td>24.58</td>
      <td>15.86</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2024-12-25</th>
      <td>29.91</td>
      <td>16.63</td>
    </tr>
    <tr>
      <th>2024-12-26</th>
      <td>30.02</td>
      <td>16.69</td>
    </tr>
    <tr>
      <th>2024-12-27</th>
      <td>30.11</td>
      <td>16.58</td>
    </tr>
    <tr>
      <th>2024-12-30</th>
      <td>29.95</td>
      <td>16.81</td>
    </tr>
    <tr>
      <th>2024-12-31</th>
      <td>29.77</td>
      <td>17.04</td>
    </tr>
  </tbody>
</table>
<p>242 rows × 2 columns</p>
</div>




```python
y = np.array(cv_kf[s1])
x = list(cv_kf[s2])
x = np.array([[i, 1] for i in x])
delta = 0.0001
yhat = np.array([np.nan for _ in range(len(y))])
e = np.array([np.nan for _ in range(len(y))])
Q = np.array([np.nan for _ in range(len(y))])
P = np.array(np.zeros([2, 2]))
beta = np.array([[np.nan for _ in range(len(y))], [np.nan for _ in range(len(y))]])
Vw = delta/(1-delta)*np.diag(np.ones(2))
Ve = 0.1
beta[:,0] = 0
for i in range(1, len(y)):
    beta[:,i] = beta[:,i-1] # (1)
    R = P + Vw # (2)

    yhat[i] = x[i,:]@beta[:, i] # (3)
    Q[i] = x[i,:]@R@x[i,:].transpose() + Ve # (4)
    e[i] = y[i]-yhat[i]
    K = R@x[i,:].transpose()/Q[i]
    beta[:,i] = beta[:,i] + K*e[i] # (5)
    P = R - np.outer(K, x[i,:]@R) #(6)

cv_kf['slope'] = beta[0,:]
cv_kf['intercept'] = beta[1,:]
cv_kf['control_port'] = cv_kf[s1] + cv_kf['slope']*cv_kf[s2]
```


```python
cv_kf.plot.line(y='control_port')
```

    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/IPython/core/displayhook.py:281: UserWarning: Output cache limit (currently 1000 entries) hit.
    Flushing oldest 200 entries.
      warn('Output cache limit (currently {sz} entries) hit.\n'





    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_11.png)
    



```python
cv_kf.plot.line(y='slope')
```




    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_12.png)
    



```python
cv_kf.plot.line(y='intercept')
```




    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_13.png)
    



```python
cv_kf['sqrt_Q'] = np.sqrt(Q)
cv_kf['e'] = e
cv_kf['Q_up'] = 2*cv_kf['sqrt_Q']
cv_kf['Q_low'] = -2*cv_kf['sqrt_Q']
cv_kf.plot.line(y=['Q_up', 'e', 'Q_low'])
```

    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)
    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)





    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_14.png)
    



```python
c = 2

class KF(Strategy):
    def __init__(self):
        super().__init__()
        self.active_position = None
        self.position_size = 0


    def on_bar(self):
        if self.data.loc[self.current_idx, 'e'] <= -c*self.data.loc[self.current_idx, 'sqrt_Q'] and not self.active_position:
            self.buy(s1, 1)
            self.buy(s2, self.data.loc[self.current_idx, 'slope'])
            self.active_position = 'long'
            self.position_size = self.data.loc[self.current_idx, 'slope']
        elif self.data.loc[self.current_idx, 'e'] >= c*self.data.loc[self.current_idx, 'sqrt_Q'] and not self.active_position:
            self.short_in(s1, 1)
            self.short_in(s2, self.data.loc[self.current_idx, 'slope'])
            self.active_position = 'short'
            self.position_size = self.data.loc[self.current_idx, 'slope']
        elif self.data.loc[self.current_idx, 'e'] >= c*self.data.loc[self.current_idx, 'sqrt_Q'] and self.active_position == 'long':
            self.sell(s1, 1)
            self.sell(s2, self.position_size)
            self.short_in(s1, 1)
            self.short_in(s2, self.data.loc[self.current_idx, 'slope'])
            self.active_position = 'short'
            self.position_size = self.data.loc[self.current_idx, 'slope']
        elif self.data.loc[self.current_idx, 'e'] <= -c*self.data.loc[self.current_idx, 'sqrt_Q'] and self.active_position == 'short':
            self.short_out(s1, 1)
            self.short_out(s2, self.position_size)
            self.buy(s1, 1)
            self.buy(s2, self.data.loc[self.current_idx, 'slope'])
            self.active_position = 'long'
            self.position_size = self.data.loc[self.current_idx, 'slope']
        else:
            pass

E = Engine(initial_cash=100000)
E.add_data(cv_kf)
E.add_strategy(KF())
E.run()
```

    100%|██████████| 242/242 [00:00<00:00, 8272.05it/s]

    Short Executed for 1 shares of 00635U at 2024-01-03 00:00:00.
    Short Executed for 0.30947771625129283 shares of 00642U at 2024-01-03 00:00:00.
    Short Closed for 1 shares of 00635U at 2024-01-29 00:00:00.
    Short Closed for 0.30947771625129283 shares of 00642U at 2024-01-29 00:00:00.
    Buy Executed for 1 shares of 00635U at 2024-01-29 00:00:00.
    Buy Executed for 1.4525964347311366 shares of 00642U at 2024-01-29 00:00:00.
    Sell Executed for 1 shares of 00635U at 2024-02-02 00:00:00.
    Sell Executed for 1.4525964347311366 shares of 00642U at 2024-02-02 00:00:00.
    Short Executed for 1 shares of 00635U at 2024-02-02 00:00:00.
    Short Executed for 1.4821980164543735 shares of 00642U at 2024-02-02 00:00:00.
    Short Closed for 1 shares of 00635U at 2024-02-15 00:00:00.
    Short Closed for 1.4821980164543735 shares of 00642U at 2024-02-15 00:00:00.
    Buy Executed for 1 shares of 00635U at 2024-02-15 00:00:00.
    Buy Executed for 1.4742251031188003 shares of 00642U at 2024-02-15 00:00:00.
    Sell Executed for 1 shares of 00635U at 2024-03-11 00:00:00.
    Sell Executed for 1.4742251031188003 shares of 00642U at 2024-03-11 00:00:00.
    Short Executed for 1 shares of 00635U at 2024-03-11 00:00:00.
    Short Executed for 1.4992789503638382 shares of 00642U at 2024-03-11 00:00:00.
    Short Closed for 1 shares of 00635U at 2024-06-11 00:00:00.
    Short Closed for 1.4992789503638382 shares of 00642U at 2024-06-11 00:00:00.
    Buy Executed for 1 shares of 00635U at 2024-06-11 00:00:00.
    Buy Executed for 1.6073975405227292 shares of 00642U at 2024-06-11 00:00:00.
    Sell Executed for 1 shares of 00635U at 2024-07-17 00:00:00.
    Sell Executed for 1.6073975405227292 shares of 00642U at 2024-07-17 00:00:00.
    Short Executed for 1 shares of 00635U at 2024-07-17 00:00:00.
    Short Executed for 1.552279829542606 shares of 00642U at 2024-07-17 00:00:00.
    Short Closed for 1 shares of 00635U at 2024-08-27 00:00:00.
    Short Closed for 1.552279829542606 shares of 00642U at 2024-08-27 00:00:00.
    Buy Executed for 1 shares of 00635U at 2024-08-27 00:00:00.
    Buy Executed for 1.6729951034835877 shares of 00642U at 2024-08-27 00:00:00.
    Sell Executed for 1 shares of 00635U at 2024-09-04 00:00:00.
    Sell Executed for 1.6729951034835877 shares of 00642U at 2024-09-04 00:00:00.
    Short Executed for 1 shares of 00635U at 2024-09-04 00:00:00.
    Short Executed for 1.7228716646378675 shares of 00642U at 2024-09-04 00:00:00.
    Short Closed for 1 shares of 00635U at 2024-10-04 00:00:00.
    Short Closed for 1.7228716646378675 shares of 00642U at 2024-10-04 00:00:00.
    Buy Executed for 1 shares of 00635U at 2024-10-04 00:00:00.
    Buy Executed for 1.8555929009183463 shares of 00642U at 2024-10-04 00:00:00.
    Sell Executed for 1 shares of 00635U at 2024-10-15 00:00:00.
    Sell Executed for 1.8555929009183463 shares of 00642U at 2024-10-15 00:00:00.
    Short Executed for 1 shares of 00635U at 2024-10-15 00:00:00.
    Short Executed for 1.7778833546046238 shares of 00642U at 2024-10-15 00:00:00.
    Short Closed for 1 shares of 00635U at 2024-11-01 00:00:00.
    Short Closed for 1.7778833546046238 shares of 00642U at 2024-11-01 00:00:00.
    Buy Executed for 1 shares of 00635U at 2024-11-01 00:00:00.
    Buy Executed for 1.9316165975472275 shares of 00642U at 2024-11-01 00:00:00.


    



```python
cv_kf['mv'] = E.portfolio_mv[1:]
cv_kf.plot.line(y='mv')
```




    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_15.png)
    



```python
E.max_cash_deployed
```




    72744.02281472745




```python
(E.portfolio_mv[-1]-E.portfolio_mv[0])/E.max_cash_deployed
```




    -0.14209629323824305



看來，2024年的效果不是很好。不過我覺得，這並不代表這個策略是完全無法使用的，如果回觀道我們2019-2023年的資料上，其實可以看到總資產價值線也會經過幾段回撤期，有些也長至一年。也許2024也是回撤的一年。

## 布林通道交叉驗證績效比較

我們使用先前的布林通道策略在同一段時間內與我們的卡爾曼濾波做比較。


```python
cv_BB['port'] = 5.6741343 * cv_BB[s1] + 1.8202617 * cv_BB[s2]
cv_BB.plot.line(y='port')
```




    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_16.png)
    



```python
cv_BB['mv_avg'] = cv_BB['port'].rolling(48).mean()
cv_BB['mv_std'] = cv_BB['port'].rolling(48).std()
cv_BB['BB_up'] = cv_BB['mv_avg'] + 2*cv_BB['mv_std']
cv_BB['BB_low'] = cv_BB['mv_avg'] - 2*cv_BB['mv_std']
cv_BB.plot.line(y=['port', 'BB_low', 'mv_avg', 'BB_up'])
```

    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)
    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)
    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)





    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_17.png)
    



```python
class BB(Strategy):
    def __init__(self):
        super().__init__()
        self.active_position = None


    def on_bar(self):
        if self.data.loc[self.current_idx, 'port'] <= self.data.loc[self.current_idx, 'BB_low'] and not self.active_position:
            self.buy(s1, 5.6741343)
            self.buy(s2, 1.8202617)
            self.active_position = 'long'
        elif self.data.loc[self.current_idx, 'port'] >= self.data.loc[self.current_idx, 'BB_up'] and not self.active_position:
            self.short_in(s1, 5.6741343)
            self.short_in(s2, 1.8202617)
            self.active_position = 'short'
        elif self.data.loc[self.current_idx, 'port'] >= self.data.loc[self.current_idx, 'BB_up'] and self.active_position == 'long':
            self.sell(s1, 5.6741343)
            self.sell(s2, 1.8202617)
            self.short_in(s1, 5.6741343)
            self.short_in(s2, 1.8202617)
            self.active_position = 'short'
        elif self.data.loc[self.current_idx, 'port'] <= self.data.loc[self.current_idx, 'BB_low'] and self.active_position == 'short':
            self.short_out(s1, 5.6741343)
            self.short_out(s2, 1.8202617)
            self.buy(s1, 5.6741343)
            self.buy(s2, 1.8202617)
            self.active_position = 'long'
        else:
            pass

E = Engine(initial_cash=200000)
E.add_data(cv_BB)
E.add_strategy(BB())
E.run()
```

    100%|██████████| 242/242 [00:00<00:00, 9993.52it/s]

    Short Executed for 5.6741343 shares of 00635U at 2024-03-19 00:00:00.
    Short Executed for 1.8202617 shares of 00642U at 2024-03-19 00:00:00.


    



```python
cv_BB['mv'] = E.portfolio_mv[1:]
cv_BB.plot.line(y='mv')
```




    <Axes: xlabel='date'>




    
![png](5_KalmanFilter_files/5_KalmanFilter_18.png)
    



```python
E.max_cash_deployed
```




    161203.112847




```python
(E.portfolio_mv[-1]-E.portfolio_mv[0])/E.max_cash_deployed
```




    -0.13530411869701842



可以看到，因為資產組合的價格持續上升，除了策略績效比卡爾曼波濾波差，如果我們看到交易次數會發現，這個策略在2024年沒有完成過任何一次交易。

## 結語

卡爾曼濾波在歷史上最早是美國NASA科學家在定位火箭的時候做使用，是一個應用範圍非常廣範的模型。因為本人在看到書本上應用範例之前從來沒有聽過這門學問，這次照著書本運用來做策略的情況下，可能顯得較為生疏且不靈活。不過，在寫完這篇文章之後，發現自己對於卡爾曼的濾波不了解之處極為好奇，自己找了許多資源，希望可以徹底了解它的困難之處，並且內化方便自己在未來的使用。

這是我在均值回歸為核心主題文章的最後一篇。在寫這幾片的過程中，不知道讀者的感想會不會跟我一樣。我們在比較先前遇到是關於台灣高手續費交易稅的問題。在這個環境下，要找到波動率高的協整對，成為非常重要的決勝條件。在這篇文章的例子中，可以看到期元大黃金與期元大石油幾乎滿足了我們的條件了。因此，在台灣不是不可能，只是比較難。第二個主要問題，是當我們想要寫進階策略做動態的模型調整時的知識門檻，逐漸越來越高。嚴謹又有效的交易策略需要嚴謹的邏輯思考及技術作為基底，可能是我們做量化自動化交易程式夢想的最大詛咒與優勢。身為一個台灣人，除了懂數學，同時也因為目前的資源大多都還是英文，所以也要英文程度夠好。想要寫到一個可行的策略後躺著賺錢，背後所需的就是無數個小時學習與研發的成本。天下沒有白吃的午餐。

非常感謝我的讀者，以及提供我環境從事撰寫這些文章的群馥科技。這一系列文章記錄著我從一開始對量化交易的無知，慢慢開始了解許多基本必要知識，到後來建構出自己的回測架構以及交易策略的過程。希望這些文章可以讓所有看到的人在量化的技術上有幫助，因為自己在寫的過程中確實幫助了我許多。日後，也許還有更多文章出現。

最後，建議讀者有空的話也自己翻閱一下這些文章所使用的參考書籍。因為時間的限制，書中有許多有用的知識我只能稍微帶過或忽略，對於實際市場上的交易一定會有很多幫助。書中除了均值回歸的章節還有一派量化交易策略是以價格趨勢動能做分析，與均值回歸截然不同，也是因為時間的限制，我還沒有做這部分的實作。不過非常希望以後還有機會為大家呈現！


FJ Hsu
