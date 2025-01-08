# 布林通道

FJ Hsu

在之前的文章裡使用的部位調整策略時我們遇到了幾種問題。第一個是頻繁的部位調整在有交易手續費的市場中會造成利潤過度流失；第二個就是我們沒有設定最大部位的限制，因此有可能會用到過多的資金。這時候，布林通道指標就可以派上用場了。

相信繁是有稍微研究市場交易的人應該會在某個時間點聽說過布林通道。他的概念也不難。簡單來說就是在一個價格序列上畫出一個移動平均之後，在其上方一個移動標準差的固定倍數畫一個上區間，在平均線的下方也是畫出一個移動標準差固定區間。價格低於下區間代表是買點，高於上區間代表是賣點。不過若是用了布林通道在隨便一隻股票上做交易往往會發現，依照布林通道的訊號進出場，並不能確保收益。為什麼？因為布林通道的期待就是價格會回到均值上。咦？啊那不就很剛剛好我們在研究的就是均值回歸的序列。我們的期望是，在有均值回歸性的數列上，布林通道的訊號有效許多。





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

s1 = '0050'
s2 = '006203'

s1_22 = get_data(s1, 2022)
s2_22 = get_data(s2, 2022)
s1_23 = get_data(s1, 2023)
s2_23 = get_data(s2, 2023)

s1 = pd.concat([s1_22, s1_23])
s2 = pd.concat([s2_22, s2_23])
df = pd.merge(s1, s2, on='date', how='inner')
```

這次使用的協整交易對是元大台灣50（```0050```）以及元大MSCI台灣（```006203```）。


```python
df.plot()
```

    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)





    <Axes: xlabel='date'>




    
![png](4_BBands_files/4_BBands_4_2.png)
    



```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

results_johansen = coint_johansen(df, 0, 1)
print(results_johansen.trace_stat)
print(results_johansen.trace_stat_crit_vals)
print(results_johansen.max_eig_stat)
print(results_johansen.max_eig_stat_crit_vals)
print(results_johansen.eig)
print(results_johansen.evec)
```

    [20.74518092  5.52133096]
    [[13.4294 15.4943 19.9349]
     [ 2.7055  3.8415  6.6349]]
    [15.22384996  5.52133096]
    [[12.2971 14.2639 18.52  ]
     [ 2.7055  3.8415  6.6349]]
    [0.03115478 0.01141323]
    [[ 1.15746878 -0.13417468]
     [-2.47938089  0.49664202]]


經過 Johansen Test 找出他的避險比率，資產組合結果如下，可以看到有不錯的均值回歸性。


```python
result = df.copy()
result['port'] = 1.15746878*result['0050'] -2.47938089*result['006203']
result.plot.line(y='port')
```




    <Axes: xlabel='date'>




    
![png](4_BBands_files/4_BBands_7_1.png)
    


再經過簡單最小平方回歸，可以得到均值回歸半衰期。大約為5天。我們的移動平均以及標準差的回觀期可以依照半衰期大小做簡單倍數的調整。


```python
import numpy as np
import statsmodels.api as sm
# Regression
d_port = np.diff(result['port'])
port_wc = sm.add_constant(result['port'][:-1])
model = sm.OLS(d_port, port_wc)
results = model.fit()
halflife = -np.log(2)/results.params[1]
print(halflife)
```

    4.916218213041573


    /var/folders/r8/c399k3cj0l59gk4_hz40yfy40000gn/T/ipykernel_33229/2681593137.py:8: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      halflife = -np.log(2)/results.params[1]


接下來就可以畫出布林通道。網路上有許多套件可以自動幫你畫出布林通道。但其實，自己做一點也不難。算出移動平均及移動標準差之後（這裏回觀期選擇用10天），再將移動平均數列加上與扣除兩倍平方差分別在組成兩個數列。最後，將投資組合價值、移動平均、布林通道上下軌再一起畫到圖表上就好。


```python
result['mv_avg'] = result['port'].rolling(10).mean()
result['mv_std'] = result['port'].rolling(10).std()
result['bb_up'] = result['mv_avg'] + 2*result['mv_std']
result['bb_down'] = result['mv_avg'] - 2*result['mv_std']
result.plot(y=['port', 'bb_up', 'bb_down'])
```

    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)
    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)





    <Axes: xlabel='date'>




    
![png](4_BBands_files/4_BBands_11_2.png)
    


這邊要先做一個操作理念的討論。在沒有交易成本的環境中如果想要透過布林通道控制好使用資金，但同時捕捉到每次價格序列偏離又回歸到均值的過程，那麼，最正確的做法是當價格突破一個門檻時就入場，回到均值時就出場。甚至可以設定多個區間的門檻，價格偏離均值越多，進場的部位越大。這樣確保每一段該有的收益都可以賺到，交易頻率最頻繁，捕捉的利潤也比較大。在我們的例子中，意思就是當價格高於一個標準差或兩個標準差的區間時，我們「賣」這個資產組合（融券賣出1.15張```0050```，買進2.479張```006203```），回到均值以下就出場，而當價格低於下區間時，就可以「買」這個資產組合（買進1.15張```0050```並融券賣出2.479張```006203```）同樣的回到均值以上再出場。但在前一篇小插曲的文章的結果中，我們發現因為真實市場上的賣空交易仍需出90%的保證金，加上本來買進的資金成本，我們在建造投資組合的均值回歸性序列所需的成本往往非常高，相較於組成出來的序列的變異量太大，所以若光是從偏離均值到回歸均值的區間收益，往往沒辦法超過交易的手續費以及交易稅。在這個情況下，我們得選擇一個進出場區間更大的策略，收益盡可能超越每次交易的成本，否則每次交易都是一個虧損。於是，我們得從低於下軌買進，高於上軌賣出，且我們的軌道需要用比較大倍數的標準差。這個選擇做出的犧牲就是資產組合價值不是每次偏離均值的方向都會相反，因此會錯過不少次入場機會，而且當出場的時機距離入場太遠，也可能導致走勢均值偏離原先均值，我們的出場點收益依然不足以超越交易成本。

現在先找出一個實際整數的 Hedge Ratio。我決定用7張```0050```對上-15張```006203```。組合出來的走勢如下。


```python
result['real'] = 7*result['0050'] - 15*result['006203']
result.plot.line(y='real')
```




    <Axes: xlabel='date'>




    
![png](4_BBands_files/4_BBands_13_1.png)
    


策略有兩個數值可以做調整，第一個是移動均值標準差的回顧區間，Ernie Chan的書中建議設定在均值回歸半衰期的個位數倍數，一個是布林通道的寬度。這裏，我選擇用60天回觀期，2個標準差半徑。原因是在前面的敘述中，希望找到夠大的區間並且參考大一點的回觀期以避免變動太大的移動均值以及標準差。若是在較低交易成本的環境中，可以考慮將這些數值都降低。當然，別以為我是靠直覺就知道這些數值的，也是經過許多嘗試才找出自己覺得最優的數值。為了省去中間繁瑣的過程，我直接呈現我得到的最佳結果。


```python
test = df.copy()
test['port'] = 7*test['0050'] - 15*test['006203']
test['mv_avg'] = test['port'].rolling(60).mean()
test['mv_std'] = test['port'].rolling(60).std()
test['bb_up'] = test['mv_avg'] + 2*test['mv_std']
test['bb_low'] = test['mv_avg'] - 2*test['mv_std']
```

這第一個策略是在進場點，不論是買或賣，進場，遇到下一個出場點出場。然後等待下一個入場點重複。每一次買然後賣、賣然後買的過程不重複交疊或銜接。


```python
from Disillusion_v1_0 import *

class BB(Strategy):
    def __init__(self):
        super().__init__()
        self.active_up = False
        self.active_down = False


    def on_bar(self):
        if self.data.loc[self.current_idx, 'port'] <= self.data.loc[self.current_idx, 'bb_low'] and not self.active_up and not self.active_down:
            self.buy('0050', 7)
            self.short_in('006203', 15)
            self.active_up = True
        elif self.data.loc[self.current_idx, 'port'] >= self.data.loc[self.current_idx, 'bb_up'] and self.active_up:
            self.sell('0050', 7)
            self.short_out('006203', 15)
            self.active_up = False
        elif self.data.loc[self.current_idx, 'port'] >= self.data.loc[self.current_idx, 'bb_up'] and not self.active_up and not self.active_down:
            self.short_in('0050', 7)
            self.buy('006203', 15)
            self.active_down = True
        elif self.data.loc[self.current_idx, 'port'] <= self.data.loc[self.current_idx, 'bb_low'] and self.active_down:
            self.short_out('0050', 7)
            self.sell('006203', 15)
            self.active_down = False
        else:
            pass

e = Engine(initial_cash=3000000)
e.add_data(test)
e.add_strategy(BB())
e.run()
```

      0%|          | 0/483 [00:00<?, ?it/s]

    Buy Executed for 7 shares of 0050 at 2022-05-12 00:00:00.
    Short Executed for 15 shares of 006203 at 2022-05-12 00:00:00.
    Sell Executed for 7 shares of 0050 at 2022-08-04 00:00:00.
    Short Closed for 15 shares of 006203 at 2022-08-04 00:00:00.
    Buy Executed for 7 shares of 0050 at 2022-08-26 00:00:00.
    Short Executed for 15 shares of 006203 at 2022-08-26 00:00:00.
    Sell Executed for 7 shares of 0050 at 2022-10-20 00:00:00.
    Short Closed for 15 shares of 006203 at 2022-10-20 00:00:00.
    Short Executed for 7 shares of 0050 at 2022-11-17 00:00:00.
    Buy Executed for 15 shares of 006203 at 2022-11-17 00:00:00.
    Short Closed for 7 shares of 0050 at 2023-06-20 00:00:00.
    Sell Executed for 15 shares of 006203 at 2023-06-20 00:00:00.
    Buy Executed for 7 shares of 0050 at 2023-07-04 00:00:00.
    Short Executed for 15 shares of 006203 at 2023-07-04 00:00:00.
    Sell Executed for 7 shares of 0050 at 2023-11-02 00:00:00.
    Short Closed for 15 shares of 006203 at 2023-11-02 00:00:00.
    Buy Executed for 7 shares of 0050 at 2023-12-21 00:00:00.
    Short Executed for 15 shares of 006203 at 2023-12-21 00:00:00.


    100%|██████████| 483/483 [00:00<00:00, 12995.37it/s]


可以發現總資產價值走線沒有向上很多，但至少它沒有虧，我們到最後一次交易時，扣掉了手續費還在水上，實在難得了。不過在兩年內沒有賺多少，不是一個很有效的策略。


```python
test['pv'] = e.portfolio_mv[1:]
test.plot.line(y='pv')
```




    <Axes: xlabel='date'>




    
![png](4_BBands_files/4_BBands_19_1.png)
    


我們在策略裡使用的布林通道視覺化結果如下。


```python
test.plot.line(y=['port','mv_avg','bb_up','bb_low'])
```

    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)
    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)
    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)





    <Axes: xlabel='date'>




    
![png](4_BBands_files/4_BBands_21_2.png)
    


最大使用資金也控制在兩百萬以內。


```python
e.max_cash_deployed
```




    1791939.47362762



想了一下後，發現為什麼不在同一個訊號出場又再反方向進場？這樣可以盡可能的增加交易的頻率，資金不會閒置。於是我做出了第二個策略。


```python
class BB(Strategy):
    def __init__(self):
        super().__init__()
        self.active_position = None


    def on_bar(self):
        if self.data.loc[self.current_idx, 'port'] <= self.data.loc[self.current_idx, 'bb_low'] and not self.active_position:
            self.buy('0050', 7)
            self.short_in('006203', 15)
            self.active_position = 'long'
        elif self.data.loc[self.current_idx, 'port'] >= self.data.loc[self.current_idx, 'bb_up'] and not self.active_position:
            self.short_in('0050', 7)
            self.buy('006203', 15)
            self.active_position = 'short'
        elif self.data.loc[self.current_idx, 'port'] >= self.data.loc[self.current_idx, 'bb_up'] and self.active_position == 'long':
            self.sell('0050', 7)
            self.short_out('006203', 15)
            self.short_in('0050', 7)
            self.buy('006203', 15)
            self.active_position = 'short'
        elif self.data.loc[self.current_idx, 'port'] <= self.data.loc[self.current_idx, 'bb_low'] and self.active_position == 'short':
            self.short_out('0050', 7)
            self.sell('006203', 15)
            self.buy('0050', 7)
            self.short_in('006203', 15)
            self.active_position = 'long'
        else:
            pass

e = Engine(initial_cash=3000000)
e.add_data(test)
e.add_strategy(BB())
e.run()
```

    100%|██████████| 483/483 [00:00<00:00, 11446.90it/s]

    Buy Executed for 7 shares of 0050 at 2022-05-12 00:00:00.
    Short Executed for 15 shares of 006203 at 2022-05-12 00:00:00.
    Sell Executed for 7 shares of 0050 at 2022-08-04 00:00:00.
    Short Closed for 15 shares of 006203 at 2022-08-04 00:00:00.
    Short Executed for 7 shares of 0050 at 2022-08-04 00:00:00.
    Buy Executed for 15 shares of 006203 at 2022-08-04 00:00:00.
    Short Closed for 7 shares of 0050 at 2022-08-26 00:00:00.
    Sell Executed for 15 shares of 006203 at 2022-08-26 00:00:00.
    Buy Executed for 7 shares of 0050 at 2022-08-26 00:00:00.
    Short Executed for 15 shares of 006203 at 2022-08-26 00:00:00.
    Sell Executed for 7 shares of 0050 at 2022-10-20 00:00:00.
    Short Closed for 15 shares of 006203 at 2022-10-20 00:00:00.
    Short Executed for 7 shares of 0050 at 2022-10-20 00:00:00.
    Buy Executed for 15 shares of 006203 at 2022-10-20 00:00:00.
    Short Closed for 7 shares of 0050 at 2023-06-20 00:00:00.
    Sell Executed for 15 shares of 006203 at 2023-06-20 00:00:00.
    Buy Executed for 7 shares of 0050 at 2023-06-20 00:00:00.
    Short Executed for 15 shares of 006203 at 2023-06-20 00:00:00.
    Sell Executed for 7 shares of 0050 at 2023-11-02 00:00:00.
    Short Closed for 15 shares of 006203 at 2023-11-02 00:00:00.
    Short Executed for 7 shares of 0050 at 2023-11-02 00:00:00.
    Buy Executed for 15 shares of 006203 at 2023-11-02 00:00:00.
    Short Closed for 7 shares of 0050 at 2023-12-21 00:00:00.
    Sell Executed for 15 shares of 006203 at 2023-12-21 00:00:00.
    Buy Executed for 7 shares of 0050 at 2023-12-21 00:00:00.
    Short Executed for 15 shares of 006203 at 2023-12-21 00:00:00.


    



```python
print(e.cash)
```

    1221286.6956754765


這次的總資產走線表現好一些了！回撤也不會太大，雖然以兩年的時間來看，這賺的還是不怎麼樣，但至少我們可以看到正收益了。


```python
test['pv'] = e.portfolio_mv[1:]
test.plot.line(y='pv')
```




    <Axes: xlabel='date'>




    
![png](4_BBands_files/4_BBands_28_1.png)
    



```python
print(e.max_cash_deployed)
```

    1778713.3043245235


## 上虛擬戰場

做出我能找到最好的結果之後，我決定進行實際測試了。我拿2024年開始的資訊做一個實際的回測。


```python
def get_data(symbol, year):
    s = stock.historical.candles(**{"symbol": symbol,
                                  "from": f"{year}-01-01",
                                  "to": f"{year}-12-29",
                                  "fields": "close",
                                  'sort': 'asc'})
    f = pd.DataFrame.from_dict(s['data'])
    f['date'] = pd.to_datetime(f['date'])
    f.set_index('date', inplace=True)
    f.rename(columns={'close':f'{symbol}'}, inplace=True)
    return f

df1 = get_data('0050', '2024')
df2 = get_data('006203', '2024')

field = pd.merge(df1, df2, on='date', how='inner')
```


```python
field['port'] = 7*field['0050'] - 15*field['006203']
field['mv_avg'] = field['port'].rolling(60).mean()
field['mv_std'] = field['port'].rolling(60).std()
field['bb_up'] = field['mv_avg'] + 2*field['mv_std']
field['bb_low'] = field['mv_avg'] - 2*field['mv_std']
field.plot.line(y='port')
```




    <Axes: xlabel='date'>




    
![png](4_BBands_files/4_BBands_32_1.png)
    



```python
class BB(Strategy):
    def __init__(self):
        super().__init__()
        self.active_position = None


    def on_bar(self):
        if self.data.loc[self.current_idx, 'port'] <= self.data.loc[self.current_idx, 'bb_low'] and not self.active_position:
            self.buy('0050', 7)
            self.short_in('006203', 15)
            self.active_position = 'long'
        elif self.data.loc[self.current_idx, 'port'] >= self.data.loc[self.current_idx, 'bb_up'] and not self.active_position:
            self.short_in('0050', 7)
            self.buy('006203', 15)
            self.active_position = 'short'
        elif self.data.loc[self.current_idx, 'port'] >= self.data.loc[self.current_idx, 'bb_up'] and self.active_position == 'long':
            self.sell('0050', 7)
            self.short_out('006203', 15)
            self.short_in('0050', 7)
            self.buy('006203', 15)
            self.active_position = 'short'
        elif self.data.loc[self.current_idx, 'port'] <= self.data.loc[self.current_idx, 'bb_low'] and self.active_position == 'short':
            self.short_out('0050', 7)
            self.sell('006203', 15)
            self.buy('0050', 7)
            self.short_in('006203', 15)
            self.active_position = 'long'
        else:
            pass

e = Engine(initial_cash=3000000)
e.add_data(field)
e.add_strategy(BB())
e.run()
```

    100%|██████████| 240/240 [00:00<00:00, 12383.69it/s]

    Buy Executed for 7 shares of 0050 at 2024-05-31 00:00:00.
    Short Executed for 15 shares of 006203 at 2024-05-31 00:00:00.
    Sell Executed for 7 shares of 0050 at 2024-06-19 00:00:00.
    Short Closed for 15 shares of 006203 at 2024-06-19 00:00:00.
    Short Executed for 7 shares of 0050 at 2024-06-19 00:00:00.
    Buy Executed for 15 shares of 006203 at 2024-06-19 00:00:00.
    Short Closed for 7 shares of 0050 at 2024-09-05 00:00:00.
    Sell Executed for 15 shares of 006203 at 2024-09-05 00:00:00.
    Buy Executed for 7 shares of 0050 at 2024-09-05 00:00:00.
    Short Executed for 15 shares of 006203 at 2024-09-05 00:00:00.
    Sell Executed for 7 shares of 0050 at 2024-11-11 00:00:00.
    Short Closed for 15 shares of 006203 at 2024-11-11 00:00:00.
    Short Executed for 7 shares of 0050 at 2024-11-11 00:00:00.
    Buy Executed for 15 shares of 006203 at 2024-11-11 00:00:00.


    



總資產價值如下：


```python
field['pv'] = e.portfolio_mv[1:]
field.plot.line(y='pv')
```




    <Axes: xlabel='date'>




    
![png](4_BBands_files/4_BBands_35_1.png)
    



```python
field.plot.line(y=['port','mv_avg','bb_up','bb_low'])
```

    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)
    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)
    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)





    <Axes: xlabel='date'>




    
![png](4_BBands_files/4_BBands_36_2.png)
    



```python
e.cash
```




    362667.161629064




```python
print((e.portfolio_mv[-1]-e.portfolio_mv[0])/e.max_cash_deployed)
```

    0.0009450060875098868


可以看到，說好，不是很好，說爛，也不爛，因為我至少打敗了邪惡的交易手續費和交易稅了。但這個策略有實際使用的意義嗎？完全沒有。光持有投資級別公司債的收益高多了也穩定多了。

這篇文章研究了布林通道操作上的細節。即使這次使用案例的結果並不亮眼，我們需要記得這不是因為布林通道的本質無效，而是投資環境的限制。若是在回測中設定沒有交易手續費，會發現收益是持續穩定上漲的，而且在選擇合適的回觀期以及通道半徑下，我們的收益甚至可以非常好看。因此，引申出了一些想法，供大家參考與討論。交易成本是否是一個讓台灣股市不能趨向完整市場的一個障礙點？交易成本的存在，讓較有科學性的量化交易較難獲利，原因不是因為市場上的資產與預期數值誤差太小，而是阻礙了市場上交易者去讓資產價值得到該有的平衡的障礙？另外，在尋找最合適的回觀期與布林通道半徑的過程中，是否有更巧妙的數學方法得出最優解，而不是用撒網捕魚班的方法將所有數值填入並看結果？
