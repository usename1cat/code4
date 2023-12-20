import my_xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载示例数据集（乳腺癌数据集）
data = load_breast_cancer()
X = data.data
y = data.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用XGBoost的DMatrix数据结构
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置XGBoost的参数
params = {
    'objective': 'binary:logistic',  # 二分类问题
    'max_depth': 3,                   # 树的最大深度
    'learning_rate': 0.1,             # 学习率
    'eval_metric': 'logloss'          # 评估指标
}

# 训练模型
num_round = 100  # 迭代次数
model = xgb.train(params, dtrain, num_round)

# 在测试集上进行预测
y_pred = model.predict(dtest)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy}")
