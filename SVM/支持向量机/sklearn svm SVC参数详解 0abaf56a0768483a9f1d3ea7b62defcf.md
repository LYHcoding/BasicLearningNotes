# sklearn.svm.SVC参数详解

[https://blog.csdn.net/weixin_42279212/article/details/121504641](https://blog.csdn.net/weixin_42279212/article/details/121504641)

# 1. 前言

- 转载请注明出处
- 文章中有一部分内容是个人理解，所以内容仅供参考
- 这篇文章是讲解**sklearn库中SVM部分中SVC这一API**.
- 关于实战部分可以参考这篇文章（有源码，可直接运行）：[【Sklearn】【实战】【SVM】乳腺癌检测，模拟线上部署（1）](https://blog.csdn.net/weixin_42279212/article/details/121481117)
- 这里是官方说明文档传送门：[sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
- 本文约7000字，阅读完毕大约需要10分钟
- 本文可当做开发时的开发手册作为参考，建议收藏

# 2. 简介

- SVC为Support Vector Classification的简写，顾名思义，其是基于支持向量的分类器
- SVC是基于**libsvm**实现的
- SVC的拟合时间是和样本数量呈二次方指数关系，因此这一分类模型**适用于样本较小情况**，如果**样本数量过大（超过1W），建议使用其他模型，例如`LinearSVC` 或者 `SGDClassifier`**

# 3. 语法

## 3.1 API形式

- 形式如下，里面的参数均为默认参数

```
SVC( C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001,
cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr',
break_ties=False, random_state=None)

```

## 3.2 参数说明

[Untitled](sklearn%20svm%20SVC%E5%8F%82%E6%95%B0%E8%AF%A6%E8%A7%A3%200abaf56a0768483a9f1d3ea7b62defcf/Untitled%20Database%200834cf5f1bd349c69f88e18f2fa03671.csv)

## 3.3 属性说明

[Untitled](sklearn%20svm%20SVC%E5%8F%82%E6%95%B0%E8%AF%A6%E8%A7%A3%200abaf56a0768483a9f1d3ea7b62defcf/Untitled%20Database%2057ba4ab23f2146909bf115f1d4f4ea3e.csv)

# 4. 方法说明

## 4.1 decision_function(X)

**1. 描述：** 计算所有样本X的决策函数**2. 参数：** X为所有样本组成的二维数组，大小为(n_samples, n_features)**3. 返回值：** 返回模型中每个类的样本决策函数，大小为 (n_samples, n_classes * (n_classes-1) / 2)**4. 注意：** 如果decision_function_shape =“ ovr”，则返回值的大小为（n_samples，n_classes）

## 4.2 fit(X, y, sample_weight=None)

**1. 描述：** 用训练数据拟合模型**2. 参数：** X: 训练数据; y: 训练数据标签； sample_weight: 每个样本的权重，(n_samples,)**3. 返回值：** 自身，拟合好的模型**4. 注意：** 无

## 4.3 get_params(deep=True)

**1. 描述：** 获取模型的所有参数**2. 参数：** 如果为真，则将返回此模型和作为模型的所包含子对象的参数**3. 返回值：** 字典类型， 所有的参数**4. 注意：** 无

## 4.4 predict(X)

**1. 描述：** 用拟合好的模型对所有样本X进行预测**2. 参数：** 所有预测样本，二维数组（n_samples, n_features)**3. 返回值：** 所有预测 X的预测标签，一维数组，(n_sample, )**4. 注意：** 无

## 4.5 predict_log_proba(X)

**1. 描述：** 计算所有预测样本在每个类别上的对数概率**2. 参数：** 所有预测样本，二维数组（n_samples, n_features)**3. 返回值：** 返回模型中每个类的样本的对数概率，二维数组，(n_samples, n_classes)**4. 注意：** 在模型训练时，需要将 **probability**参数设置为True，才能使用此方法

## 4.6 predict_proba(X)

**1. 描述：** 计算所有预测样本在每个类别上的概率**2. 参数：** 所有预测样本，二维数组（n_samples, n_features)**3. 返回值：** 返回模型中每个类的样本的对数概率，二维数组，(n_samples, n_classes)**4. 注意：** 在模型训练时，需要将 **probability**参数设置为True，才能使用此方法

## 4.7 score(X, y, sample_weight=None)

**1. 描述：** 返回给定测试数据上的平均准确度**2. 参数：** X: 训练数据; y: 训练数据标签； sample_weight: 每个样本的权重，(n_samples,)**3. 返回值：** 浮点类型，平均准确度**4. 注意：** 无

## 4.8 set_params(**params)

**1. 描述：** 重置当前模型的参数**2. 参数：** 字典类型，内容为当前模型的参数**3. 返回值：** 重置参数后的模型**4. 注意：** 无

# 5. 总结

不知不觉六个小时已经过去了，这会儿已经凌晨四点了。本以为这篇文章两个小时就能结束。无奈中间遇到了一个又一个不懂的知识点，整个过程就像升级打怪一样，在整理整个API的过程中，个人对模型的理解更深刻了，颇有收获。

有时间再继续往下更新~

希望这篇文档能对各位看官产生一定的帮助， 如有不妥，欢迎评论区指正~

# 6. 参考资料

1. [LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)
2. [英文版：sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
3. [RBF径向基函数](https://blog.csdn.net/qq_15295565/article/details/80888607?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.no_search_link&spm=1001.2101.3001.4242.1)
4. [Gram矩阵和核函数](https://blog.csdn.net/qq_35866736/article/details/97289151)
5. [SVM的概率输出（Platt scaling）](https://blog.csdn.net/giskun/article/details/49329095)