# ‘rbf’, ‘poly’ 和‘sigmoid’ 核函数的系数

作用: 1. auto: gamma = 1 / n_features  2. scale: gamma = 1 / (n_features * X.var())  3. 只作用于 rbf， poly，sigmoid 三个核函数
参数: gamma
数据: 1. string类型，默认值为‘scale’  2. {‘auto’, ‘scale’}