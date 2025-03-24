import os


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import report_handle

classifiers = {
    "Logistic Regression":  LogisticRegression(
    multi_class="multinomial",  # 使用多项逻辑回归
    solver="lbfgs",            # 使用 LBFGS 求解器
    max_iter=1000,             # 最大迭代次数
    C=1.0,                     # 正则化强度
    tol=1e-4,                  # 收敛阈值
    fit_intercept=True,        # 拟合截距项
    class_weight=None          # 不调整类别权重
),
    "SVM": SVC(kernel="rbf", decision_function_shape="ovo"),
    "Random Forest-50": RandomForestClassifier(n_estimators=50),
    # "Random Forest-100": RandomForestClassifier(n_estimators=100),
    # "Random Forest-30": RandomForestClassifier(n_estimators=30),
    # "k-NN-3": KNeighborsClassifier(n_neighbors=3),
    "k-NN-5": KNeighborsClassifier(n_neighbors=5),
    # "k-NN-8": KNeighborsClassifier(n_neighbors=8)
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
}


def process(X_train_split, X_predict_split, y_train_split, y_predict_split, param, p_num):

    for name, clf in classifiers.items():
        scaler = StandardScaler()
        X_train_split = scaler.fit_transform(X_train_split)
        X_predict_split = scaler.fit_transform(X_predict_split)
        # 训练
        clf.fit(X_train_split, y_train_split)

        # 预测
        y_pred = clf.predict(X_predict_split)
        y_true = y_predict_split
        accuracy,f1,report = report_handle.get_report(y_true,y_pred)
        report_model_name = report_handle.reportDirDev + os.sep + param + '_' +name +'_'+ p_num + '_report.csv'
        report_handle.save(report, report_model_name)


