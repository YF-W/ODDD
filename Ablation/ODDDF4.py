import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import time
import pandas as pd
# 5_1的基础上修改评价指标

class Point:
    def __init__(self, attributes, bright=0, density=0, connectivity=0, indegree=0, outdegree=0, status=0,
                 move_distance=0, outlier_degree=0, min_bright=9999,max_bright=0,avg_bright=0, previous_connection_list=[]):
        """
        这个函数用来初始化Point类，作为它的构造函数。当我们要构造一个点对象时，就需要使用到这个函数
        使用方法： point_1=Point(attributes, bright, density,connectivity,indegree,outdegree,status,move_distance)
        :param attributes: 数据的属性，比如x,y,z这些。就是传入的原始数据如 [1,2,3; 4,5,6]
        :param bright: 点的亮度 如 0.9
        :param density: 密度————通过计算 k个近邻到当前点的平均距离/数量 得到的
        :param connectivity: 连接性
        :param indegree: 入度，其它点指向当前点。 指向———— A密度比B大，那么B就会指向A  B-->A
        :param outdegree: 出度 ，当前点指向其它点。 指向———— A密度比B小，那么A就会指向B A-->B
        :param status: 状态，决定点是否休眠。 1表示要休眠，0表示不休眠
        :param move_distance: 累计移动距离，用来画决策图
        :param outlier_degree:孤立程度
        :param min_bright: 最小的亮度
        :param previous_connection_list：以前的连接性列表
        """
        self.attributes = attributes
        self.bright = bright
        self.density = density
        self.connectivity = connectivity
        self.indegree = indegree
        self.outdegree = outdegree
        self.status = status
        self.move_distance = move_distance
        self.outlier_degree = outlier_degree
        self.min_bright = min_bright
        self.max_bright = max_bright
        self.avg_bright = avg_bright
        self.previous_connection_list = previous_connection_list


def Init(X):
    """

    这个函数用来进行初始化，也就是输入点数据及其属性
    使用方法：point_list=LightDetection.init(data) 这里的data指存有数据文件，视具体情况而填写即可
    :param X: 传入的所有数据属性，比如x,y,z...
    :return:  返回存有点对象的属性的列表
    """
    point_list = []  # 创建空列表
    # 对每个点的属性进行存储
    for i in range(X.shape[0]):  # shape[0]表示矩阵的行数，shape[1]表示矩阵的列数
        temp_point = Point(X[i])  # 传入属性
        point_list.append(temp_point)  # 将属性增加进列表中
        # num表示参数数量
    return point_list  # 返回存有数据的列表


def Model(X, max_k, omega, eta, mu, niu, contamination):
    """
    模型函数，调用后可以返回孤立程度集合。【暂定不排序】
    [100,12,13,20,...] 表示 第一个点的孤立程度为100，第二个点为12，第三个点是13...

    :param X: 输入数据的表达矩阵
    :param max_k: 数据集遍历大小，对应论文的K，为了避免混淆，就没有用max_k
    :param omega: 吸引力系数
    :param eta: 扰动系数
    :param mu: 吸引频度（大的步进次数）
    :param niu: 探测频度（小的探测次数）
    :return:
    """
    point_list = Init(X)  # 1. 初始化（参数初始化、变量初始化）；
    # NOTE 这里每个点有要有数据的属性，比如x,y,z这些,这些都用attributes属性表示,调用方法 point_list[0].attributes[0] 表示0号点的第0个属性的值
    distance_matrix = Get_Distance_Matrix(point_list)  # 调用距离矩阵的方法
    # 对密度进行更新
    Get_Density(distance_matrix, point_list, max_k)  # 点越稀疏，密度越大
    neighbor_matrix = Get_Neighbor_Matrix(distance_matrix)  # 调用近邻矩阵的方法
    sorted_bright_list = Data_Association_Module(point_list, neighbor_matrix, max_k, 1)  # [更新亮度]得到排序后的亮度列表，里面存储的是点序号


    # 3. 计算数据关联，确定亮度、休眠。


    for i in range(1, mu+1):  # 4. 外层循环，遍历大步进
        # [大移动] 移动后更新的状态都在小移动的循环里保存
        previous_point_list= Data_Displacement_Module(sorted_bright_list, point_list, distance_matrix,
                                                       neighbor_matrix, max_k, omega)  # 5.执行数据位移模块（算法3）；根据算法2的亮度等，执行位移

        # # [更新密度] 每次移动会导致距离变化，因此需要更新密度
        Get_Density(distance_matrix, point_list, max_k)  # TODO 点越稀疏，密度越大
        # [更新亮度] 因为密度变化了，所以亮度也需要修改
        sorted_bright_list = Data_Association_Module(point_list, neighbor_matrix, max_k,
                                                     1)  # [更新亮度]得到排序后的亮度列表，里面存储的是点序号
        # [更新近邻]
        neighbor_matrix = Get_Neighbor_Matrix(distance_matrix)

        for j in range(1, niu+1):  # 6. 内层循环，扰动探测，寻找周围的最佳解
            sorted_bright_list, outlier_score, outlier_score_list, point_list = Interference_Detection_Module(
                point_list, previous_point_list, max_k, eta, contamination)
            # 输出离群值评分的列表


   # TODO 生成决策图 （亮度，移动总距离）

    y_predict_type = [0] * len(point_list)
    sorted_outlier_list_id = sorted(range(len(outlier_score_list)), key=lambda k: outlier_score_list[k],
                                    reverse=True)  # 排序后的序号列表。sorted_outlier_list_id[0]=5 表示5号点的孤立度最大

    outlier_num = int(len(point_list) * contamination)
    for i in range(0, outlier_num):
        outlier_index = sorted_outlier_list_id[i]
        y_predict_type[outlier_index] = 1

    return outlier_score_list,y_predict_type


# 更新亮度
def Data_Association_Module(point_list, neighbor_matrix, max_k, type):
    """
     计算点的关联
    :param point_list: 点对象的列表
    :param neighbor_matrix: 近邻矩阵
    :param max_k: 最大的k
    :param type: type==0小扰动，type==1大扰动
    :return: 排序好的亮度列表。如：sorted_bright_list_id[0]=5 表示5号点的亮度是最高的
    """
    bright_list = []  # 首先创建一个亮度的空列表

    # 更新连接性
    # 1. 确定每个点的密度
    # 通过X,确定每个点密度。然后把这个密度作为点的属性更新
    # NOTE 到这里每个点有要有密度的属性
    for i in range(0, len(point_list)):
        # 遍历每一个点
        # 计算其它k个近邻点与i点的出入度
        point_list[i].outdegree = 0
        point_list[i].indegree = 0
        for j in range(1, max_k + 1):
            if point_list[neighbor_matrix[i][j]].density > point_list[i].density:  # 如果比较的点密度比当前点大，那么当前点的出度+1
                point_list[i].outdegree = point_list[i].outdegree + 1  # 执行当前点的出度+1
            else:  # 否则，那么当前点的入度+1
                point_list[i].indegree = point_list[i].indegree + 1  # 执行当前点的入度+1
    # NOTE 到这里每个点有要有出入度的属性

    # 2. 通过密度确定数据集点的连接性xc；（list）
    for i in range(0, len(point_list)):
        point_list[i].connectivity = point_list[i].indegree - point_list[i].outdegree  # 计算当前点的连接性
        # TODO 连接性正负数
        # 通过连接性算亮度
        if type == 0:  # 如果是小扰动
            point_list[i].bright = sigmoid(point_list[i].connectivity)  # 求小扰动的亮度

        else:  # 其它情况就是公式二
            if len(point_list[i].previous_connection_list) == 0:  # 如果是初次扰动
                point_list[i].bright = sigmoid(point_list[i].connectivity)  # 直接计算点的亮度，点的连接性的sigmoid值
            else:
                avg_disturbance = 0  # 计算平均扰动的值
                for v in range(0, len(point_list[
                                          i].previous_connection_list)):  # point_list[i].last_disturbance_list 中的 last_disturbance_list 表示点i对应上一次小扰动的连接性集合
                    avg_disturbance += point_list[i].previous_connection_list[v]  # 上次进行扰动的点的连接性的总和
                avg_disturbance = avg_disturbance / len(point_list[i].previous_connection_list)  # 求得上次扰动的平均连接性的值
                point_list[i].bright = sigmoid((avg_disturbance + point_list[i].connectivity) / 2)  # 获得这次的连接性

        if point_list[i].bright < point_list[i].min_bright:  # 更新每个点的最小亮度
            point_list[i].min_bright = point_list[i].bright

        point_list[i].avg_bright=(point_list[i].avg_bright+point_list[i].bright)/2

        if point_list[i].max_bright < point_list[i].bright:  # 更新每个点的最小亮度
            point_list[i].max_bright = point_list[i].bright
        # NOTE 到这里每个点有要有亮度的属性
        bright_list.append(point_list[i].bright)  # 存储亮度到list里面
    # 6.    通过当前c和上次扰动c的均值确定数据点的亮度xb，并进行排序；
    # TODO
    sorted_bright_list_id = sorted(range(len(bright_list)), key=lambda k: bright_list[k],
                                   reverse=False)  # 亮度排序后的序号列表。sorted_bright_list_id[0]=5 表示5号店的亮度最小
    sleep_thershold = 0.9  # 判断点是否为休眠点的临界值
    # 7. 确定休眠点；
    for i in range(0, len(point_list)):
        if point_list[i].bright > sleep_thershold:  # 比较点亮度是否大于临界值
            point_list[i].status = 1  # 该点为休眠点
    return sorted_bright_list_id  # 返回排序好的亮度列表


# 大移动
def Data_Displacement_Module(sorted_bright_list, point_list, distance_matrix, neighbor_matrix, max_k, omega):
    """
    每次大移动会自动更新点对象的属性位置，和距离矩阵对应的行列
    :param sorted_bright_list:
    :param point_list:
    :param distance_matrix:
    :param neighbor_matrix:
    :param max_k:
    :param omega:
    :param big_move_report_list: 保存用的报告list
    :param report_list: 总报告list
    :return: last_disturbance_list 扰动的连通性列表
    """

    previous_point_list = copy.deepcopy(point_list)  # 保存还没动时的状态，用于后面小移动的目标替换
    # 判断是否有休眠点并且找到现况下本轮最亮的点
    for i in range(0,len(sorted_bright_list)):
        now_point_order=sorted_bright_list[i]
        now_point = point_list[sorted_bright_list[i]]  # 获取本轮点对象亮度的升序列表
        if now_point.status == 1:  # 该点休眠，则跳过本轮
            now_point.status = 0
            point_list[now_point_order].status = 0
            continue  # 终止执行本次循环中剩下的代码，直接从下一次循环继续执行
        max_bright_point = point_list[neighbor_matrix[now_point_order][1]]  # 亮度最大点为列表中点对象0号点的1号近邻

        for j in range(1, max_k + 1):  # 遍历前k个近邻[不考虑自身]
            if point_list[neighbor_matrix[now_point_order][j]].bright >= max_bright_point.bright:  # 找到最亮的点
                max_bright_point = point_list[neighbor_matrix[now_point_order][j]]  # 得到本轮最亮的点
        # 计算吸引力
        xi = now_point  # 当前点
        xj = max_bright_point  # 最亮的点
        a_xi2xj_ = xj.bright - xi.bright  # 得到两点的吸引力大小
        # //TODO Move
        distance = 0  # 先将距离初始化为0
        for k in range(0, len(xi.attributes)):
            distance += (xi.attributes[k] - xj.attributes[k]) ** 2  # 计算xi->xj两点之间对应的属性的数据的差值的平方和
            # k表示点的每一列数据
        distance = math.sqrt(distance)  # 平方和求开方得距离
        move = distance * a_xi2xj_ * omega  # 计算xi->xj两点间的大移动
        if distance != 0:
            r = move / distance  # 计算移动距离占两个点的总距离的多少
        else:
            r = 0
        # 计算xi的属性移动后的值的大小

        for j in range(0, len(xi.attributes)):  # j表示点的每一列数据
            xi.attributes[j] = r * (xj.attributes[j] - xi.attributes[j]) + xi.attributes[j]  # 值的计算
        xi.move_distance += move  # 两点的大移动的距离的叠加


        for j in range(0, len(point_list)):  # 更新距离矩阵对应的i行j列
            temp = 0
            for k in range(0, len(point_list[i].attributes)):  # 计算距离
                temp += (point_list[i].attributes[k] - point_list[j].attributes[k]) ** 2
                # 计算现在点对象的状态下i点和j点的两点对应属性的数据的差值的平方和之和
            temp = math.sqrt(temp)  # 对平方和的总和求开方
            # 将得到的值更新到矩阵的对应位置

            distance_matrix[i][j] = temp
            distance_matrix[j][i] = temp



    return previous_point_list # 返回扰动的连通性列表


# 小移动
def Interference_Detection_Module(point_list, previous_point_list, max_k, eta, contamination):
    """
    执行小扰动，并且记录每次小扰动的连接性到点的属性previous_connection_list
    :param point_list: 传入的点对象列表
    :param previous_point_list: 上一步点的状态、位置等。 是个对象列表
    :param max_k: 最大的k
    :param eta: eta用于移动计算，是给定的扰动系数
    :return: 连接性
    """

    # 首先将本轮的点对象列表的值进行复制存储
    point_list_left = copy.deepcopy(point_list)  # 复制point_list，用来后面评测得分
    point_list_right = copy.deepcopy(point_list)  # 复制point_list，用来后面评测得分

    for i in range(0, len(point_list)):

        # 左移动的初始点位
        xi = copy.deepcopy(point_list[i])  # 对xi赋值
        xj = copy.deepcopy(previous_point_list[i])  # 对xj赋值
        # 右移动的初始点位
        xi_right = copy.deepcopy(point_list[i])  # 对xi_right赋值
        xj_right = copy.deepcopy(previous_point_list[i])  # 对xj_right赋值


        # TODO 这里小移动的当前点和比较点的按代码捕获都是0点，在调试的时候发现运行中出现了点亮度一样导致吸引力一样没有移动的情况
        # 左移动
        left_a_xi2xj_ = xj.bright - xi.bright  # xi xj之间的连接性
        distance = 0  # 先将距离初始化为0
        for k in range(0, len(xi.attributes)):
            distance += (xi.attributes[k] - xj.attributes[k]) ** 2  # 计算xi->xj两点之间对应的属性的数据的差值的平方和
            # k表示点的每一列数据
        distance = math.sqrt(distance)  # 平方和求开方得距离
        left_move = distance * left_a_xi2xj_ * eta  # 计算xi->xj两点间的小移动
        if distance != 0:
            r = left_a_xi2xj_ * eta  # 计算移动距离占两个点的总距离的多少
        else:
            r = 0
        # 计算xi的属性移动后的值
        for j in range(0, len(xi.attributes)):
            xi.attributes[j] = -r * (xj.attributes[j] - xi.attributes[j]) + xi.attributes[j]  # 左移动后的xi属性值的计算

        xi.move_distance += left_move  # 左移动的移动距离的累加
        point_list_left[i] = xi  # 更新左移动的点对象集合


        # 右移动
        right_a_xi2xj_ = xj_right.bright - xi_right.bright  # 计算xi xj之间的连接性
        distance = 0  # 先将距离初始化为0
        for k in range(0, len(xi_right.attributes)):
            distance += (xi_right.attributes[k] - xj_right.attributes[
                k]) ** 2  # 计算xi_right->xj_right两点之间对应的属性的数据的差值的平方和
            # k表示点的每一列数据
        distance = math.sqrt(distance)  # 平方和求开方得距离
        right_move = distance * right_a_xi2xj_ * eta  # 计算xi->xj两点间的大移动
        if distance != 0:
            r = right_a_xi2xj_ * eta  # 计算移动距离占两个点的总距离的多少
        else:
            r = 0

        # 计算xi的属性移动后的值
        for j in range(0, len(xi_right.attributes)):
            xi_right.attributes[j] = r * (xj_right.attributes[j] - xi_right.attributes[j]) + xi_right.attributes[
                j]  # 右移动后的xi属性值的计算
        xi_right.move_distance += right_move  # 右移动的移动距离的累加
        point_list_right[i] = xi_right  # 更新左移动的点对象集合

#-----left-----------
    # [更新距离矩阵]
    distance_matrix_left = Get_Distance_Matrix(point_list_left)
    # [更新密度] 每次移动会导致距离变化，因此需要更新密度
    Get_Density(distance_matrix_left, point_list_left, max_k)  # TODO 点越稀疏，密度越大
    # [更新近邻] 因为点移动，所以导致近邻排序变化
    neighbor_matrix_left = Get_Neighbor_Matrix(distance_matrix_left)
    # [更新亮度] 因为密度和近邻的变化，所以连接性（出入度也会不同）
    sorted_bright_list_id_left = Data_Association_Module(point_list_left, neighbor_matrix_left, max_k, 0)
    # 将点的连接性添加到一个储存已完成左移动的点的连接性值的列表中
    point_list_left[i].previous_connection_list=[]
    for i in range(0,len(point_list_left)):
        point_list_left[i].previous_connection_list.append(point_list_left[i].connectivity)
    # 返回本轮评估得到的数据：整体的孤立点得分（一个值，用来评估），所有点的孤立分数（画图，不排序）和孤立点的列表（取交集）
    left_score, outlier_list_left = Assessment(point_list_left, contamination)


#-----right-----------
    # [更新距离矩阵]
    distance_matrix_right = Get_Distance_Matrix(point_list_right)
    # [更新密度] 每次移动会导致距离变化，因此需要更新密度
    Get_Density(distance_matrix_right, point_list_right, max_k)  # TODO 点越稀疏，密度越大
    # [更新近邻] 因为点移动，所以导致近邻排序变化
    neighbor_matrix_right = Get_Neighbor_Matrix(distance_matrix_right)
    # [更新亮度] 因为密度和近邻的变化，所以连接性（出入度也会不同）
    sorted_bright_list_id_right = Data_Association_Module(point_list_right, neighbor_matrix_right, max_k, 0)
    # 将点的连接性添加到一个储存已完成右移动的点的连接性值的列表中
    point_list_right[i].previous_connection_list=[]
    for i in range(0,len(point_list_right)):
        point_list_right[i].previous_connection_list.append(point_list_right[i].connectivity)
    # 返回本轮评估得到的数据：整体的孤立点得分（一个值，用来评估），所有点的孤立分数（画图，不排序）和孤立点的列表（取交集）
    right_score, outlier_list_right = Assessment(point_list_right, contamination)
    # ----------

    if left_score >= right_score:  # 如果左移动的孤立点得分大于等于右移动的孤立点得分
        point_list=copy.copy(point_list_left)

        return sorted_bright_list_id_left, left_score, outlier_list_left,point_list
        # 返回排序好的亮度列表，点的得分，左移动的离群值，左移动的部分异常值
    else:
        point_list=copy.copy(point_list_right)
        return sorted_bright_list_id_right, right_score, outlier_list_right, point_list  # 返回排序好的亮度列表，点的得分，右移动的离群值，右移动的部分异常值


# sigmoid函数
def sigmoid(x):
    """

    这个函数表示的是sigmoid函数的公式计算
    使用方法：直接调用函数并传入参数即可得到其sigmoid函数值 -> sigmoid(x)
    :param x: 需要计算sigmoid值的数据
    :return: 返回的是传入x的sigmoid函数值
    """

    sigmoid = 1 / (1 + math.e ** (-x))  # sigmoid函数公式
    return sigmoid  # 返回函数值


# 评估
def Assessment(point_list, contamination):
    """
    :param point_list: 点对象列表
    :param theta: 分割的次数，自定义
    :return: 整体的孤立点得分（一个值，用来评估），所有点的孤立分数（画图，不排序）和孤立点的列表（取交集）
    """
    outlier_list = []  # 记录所有点的孤立度
    point_num=len(point_list)
    outlier_num=int(point_num*contamination)
    score=0
    # 更新每个点的孤立度
    for i in range(0, point_num):
        point_list[i].outlier_degree = point_list[i].move_distance / point_list[i].min_bright  # 孤立度=移动距离/能量
        outlier_list.append(point_list[i].outlier_degree)  # 将每个点的孤立度存入列表
    sorted_outlier_list=sorted(outlier_list,reverse=True)
    for i in range(0,outlier_num):
        score+=sorted_outlier_list[i]
    return score, outlier_list  # 返回分数，即评估当前的移动好坏


# 距离矩阵
def Get_Distance_Matrix(point_list):
    """
    密度为K近邻内的点，到所选点的距离均值。
    :param point_list: 点对象列表。 因为算密度矩阵的同时，可以把密度作为点的属性加到点对象里面。
    :return: 距离矩阵
    """
    distance_matrix = np.zeros((len(point_list), len(point_list)), float)  # 开辟一个行列都为点对象列表长度的矩阵，浮点型

    for i in range(0, len(point_list)):  # i表示当前点
        for j in range(0, len(point_list)):  # j表示另外一个被计算的点
            distance = 0
            for k in range(0, len(point_list[i].attributes)):
                # point_list[i] 表示一个点，它是一个对象,eg: ([1,2,3,4,5],0,0,0,0,0,0)
                # 我们可以从对象里面去取属性，
                # point_list[i].attributes 就把 attributes 这个属性给取出来了
                # 取出来的 attributes 是一个列表，eg: [1,2,3,4,5]
                # 列表就有长度， 通过 len() 获得长度， len(attributes) 就得到了那个列表的长度
                distance += (point_list[i].attributes[k] - point_list[j].attributes[k]) ** 2
            # k表示点的每一列数据
            distance = math.sqrt(distance)  # 计算欧式距离，开方
            distance_matrix[i][j] = distance  # 当前点与目标点的距离

    return distance_matrix


# 密度矩阵
def Get_Density(distance_matrix, point_list, max_k):
    """
    密度为K近邻内的点，到所选点的距离均值。
    :param distance_matrix: 距离矩阵。
    :param point_list: 点对象列表。 因为算密度矩阵的同时，可以把密度作为点的属性加到点对象里面。
    :param max_k: 最大的k近邻。
    :return: 因为密度是点的属性，所以更新点的属性就行，不用花费资源用list存储
    """
    for i in range(distance_matrix.shape[0]):  # 遍历每一个点
        now_density = 0  # 初始化密度
        now_point_distance_list = distance_matrix[i]  # 找到当前点的近邻列表
        sorted_now_point_distance_list = sorted(now_point_distance_list)  # 对该列表进行排序
        for j in range(0, max_k + 1):  # 找到前k个近邻
            now_density += sorted_now_point_distance_list[j]  # 当前点与前k个近邻的总距离
        now_density = max_k / now_density  # 求平均距离作为密度
        point_list[i].density = now_density  # 更新密度属性


# 近邻矩阵
def Get_Neighbor_Matrix(distance_matrix):
    """
    获得近邻矩阵
    :param distance_matrix: 距离矩阵
    :return: 近邻矩阵
    """
    neighbor_matrix = np.zeros((distance_matrix.shape[0], distance_matrix.shape[0]), int)  # 开一个近邻矩阵
    for i in range(0, distance_matrix.shape[0]):
        now_list = distance_matrix[i]  # 找到当前点的距离列表，距离矩阵的第x行
        sorted_id = sorted(range(len(now_list)), key=lambda k: now_list[k], reverse=False)  # 对距离列表从小到大排序，然后得到排序后的id
        neighbor_matrix[i] = sorted_id  # neighbor_matrix[0][1]=5 表示0号点的1号近邻为5号点。 neighbor_matrix[0][0]=0 一般来说，0号近邻是自身
    return neighbor_matrix


# 图谱
# 点图像
def scatterPlot(data):
    """
    Data visualization ( scatter plot )
    :param data: data set
    :return: No return
    """
    x = data[:, 0]
    y = data[:, 1]
    n = np.arange(data.shape[0])

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, txt in enumerate(n):
        ax.annotate(i+1, (x[i], y[i]))

    str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    plt.savefig("res_img/" + str + ".svg", dpi=600)
    plt.show()


# 决策图评分
def scatterPlotScore(data, pheList):
    """
    Data visualization ( pheromone map )
    :param data: Data set
    :return: No return
    """

    x = data[:, 0]
    y = data[:, 1]
    fig, ax = plt.subplots()

    for i in range(0, len(pheList)):
        now_pheList = normalize(pheList, pheList[i])
        plt.scatter(x[i], y[i], marker=".", s=now_pheList * 1000, c='orange')
        ax.annotate(i, (x[i], y[i]))

    str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    plt.savefig("res_img/" + str + ".svg", dpi=600)
    plt.show()


def normalize(list, value):
    range = max(list) - min(list)
    if range == 0:
        return 1
    else:
        value2 = (value - min(list)) / range
        return value2


# 亮度图像
def brightnessimage(data, point_list):
    '''
    Data visualization ( pheromone map )
    :param data: Data set
    :return: No return
    '''

    x = data[:, 0]
    y = data[:, 1]
    fig, ax = plt.subplots()

    for i in range(0, len(point_list)):
        now_bright = point_list[i].bright * 800
        plt.scatter(x[i], y[i], marker=".", alpha=0.3, s=now_bright * 1, c='red')
        # ax.annotate(round(point_list[i].bright, 2), (x[i], y[i]))
        ax.annotate(i+1, (x[i], y[i]))
    str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    plt.savefig("re_img/"+ str + ".svg", dpi=600)
    plt.show()


def movingimage(data, point_list, file_name):
    """
    Data visualization ( scatter plot )
    :param data: data set
    :return: No return
    """
    x = data[:, 0]
    y = data[:, 1]
    n = np.arange(data.shape[0])

    fig, ax = plt.subplots()
    # ax.scatter(x, y)
    for i, txt in enumerate(n):
        now_bright = point_list[i].bright * 800
        plt.scatter(x[i], y[i], marker=".", alpha=0.3, s=now_bright, c='blue')
        # ax.annotate(round(point_list[i].bright, 2), (x[i], y[i]))
        ax.annotate(i+1, (x[i], y[i]))
    plt.savefig("res_img/" + file_name + ".svg", dpi=600)
    plt.show()

