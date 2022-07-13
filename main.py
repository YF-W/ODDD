import ODDD
import scipy.io as scio

# 运行代码
start_k = 3
end_k = 5  # 修改1：k值 注意左闭右开
for k in range(start_k,end_k):
    # load data
    dataFile = 'datasets/glass.mat'  # 修改2：数据集
    dataName = "glass"  # 修改3：数据集
    input_data = scio.loadmat(dataFile)
    X_train = input_data['X']

    max_k = k
    omega = 0.8
    eta = 0.8
    mu = 3
    niu = 2
    theta = 2
    y_predict_scores = ODDD.Model(X_train, max_k, omega, eta, mu, niu, theta)  # 结果