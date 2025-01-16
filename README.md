# 這是一個使用 Ernie Chan 的 《Algorithmic Trading - Winning Strategies and Their Rationale》上的知識在台股探索的實作以及策略研發

過程中使用富果API抓取資料以及進行交易

```Disclaimer: 本文僅供教學與參考之用，實務交易應自行評估並承擔相關風險```

## 書本連結

[Algorithmic Trading: Winning Strategies and Their Rationale](https://www.google.com/search?q=algorithmic+trading%3A+winning+strategies+and+their+rationale&rlz=1C5CHFA_enTW990TW990&oq=al&gs_lcrp=EgZjaHJvbWUqCAgAEEUYJxg7MggIABBFGCcYOzIGCAEQRRhAMgYIAhBFGDkyBggDEEUYOzIGCAQQRRg8MgYIBRBFGDwyBggGEEUYPDIGCAcQRRg80gEHOTcyajBqNKgCALACAQ&sourceid=chrome&ie=UTF-8)<br>

## Fugle API Documentation

* [交易API](https://developer.fugle.tw/docs/trading/intro)<br>
* [行情API](https://developer.fugle.tw/docs/data/intro)<br>

## 小實作

1. [回測](1_回測.md)
2. [均值回歸](2_均值回歸.md)
3. [協整關係](3_協整關係.md)
4. [小插曲](01_小插曲.md)
5. [布林通道](4_BBands.md)
6. [卡爾曼濾波](5_KalmanFilter.md)

## 回測套件
另外，我自己寫了一個回測架構。適用於模擬台股交易。
* [回測套件 Version 1.0 說明文件](00_回測套件.md)   ||  [原始檔案](Disillusion_v1_0.py)（2024-12-31更新，修正 Bug）
