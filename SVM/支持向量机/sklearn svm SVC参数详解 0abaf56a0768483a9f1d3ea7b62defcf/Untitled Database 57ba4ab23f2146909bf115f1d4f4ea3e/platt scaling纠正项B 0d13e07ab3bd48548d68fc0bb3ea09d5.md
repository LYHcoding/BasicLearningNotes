# platt scaling纠正项B

作用: 1. 只有当 probability=True.时，这一纠正项B才会被计算，probability=False，则为空数组。
属性: probB_
数据: 1. ndarray， 一维数组， (n_classes * (n_classes - 1) / 2, )