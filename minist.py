import numpy as np
import matplotlib.pyplot as plt

def sigmod(z):
    h = 1. / (1 + np.exp(-z))
    return h


def de_sigmoid(z, h):
    return h * (1 - h)


def relu(z):
    h = np.maximum(z, 0)
    return h


def de_relu(z, h):
    z[z <= 0] = 0
    z[z > 0] = 1.0
    return z


def no_active(z):
    h = z
    return h


def de_no_active(z, h):
    return np.ones(h.shape)


# o Nxc
# lab Nxc
def loss_L2(o, lab):
    diff = lab - o
    sqrDiff = diff ** 2
    return 0.5 * np.sum(sqrDiff)


def de_loss_L2(o, lab):
    return o - lab


def loss_CE(o, lab):
    p = np.exp(o) / np.sum(np.exp(o), axis=1, keepdims=True)
    loss_ce = np.sum(-lab * np.log(p))
    return loss_ce


def de_loss_CE(o, lab):
    p = np.exp(o) / np.sum(np.exp(o), axis=1, keepdims=True)
    return p - lab

def bulid_net(dim_in, list_num_hidden, list_act_funs, list_de_act_funs):
    layers = []

    # 逐层的进行网络构建
    for i in range(len(list_num_hidden)):
        layer = {}

        # 定义每一层的权重
        if i == 0:
            # layer["w"]= 0.2*np.random.randn(dim_in,list_num_hidden[i])-0.1 # 用sigmoid激活函数
            layer["w"] = 0.01 * np.random.randn(dim_in, list_num_hidden[i])  # 用relu 激活函数
        else:
            # layer["w"]= 0.2*np.random.randn(list_num_hidden[i-1],list_num_hidden[i])-0.1 # 用sigmoid激活函数
            layer["w"] = 0.01 * np.random.randn(list_num_hidden[i - 1], list_num_hidden[i])  # 用relu 激活函数

        # 定义每一层的偏置
        layer["b"] = 0.1 * np.ones([1, list_num_hidden[i]])
        layer["act_fun"] = list_act_funs[i]
        layer["de_act_fun"] = list_de_act_funs[i]
        layers.append(layer)

    return layers

def load_mnist(file_data, file_lab):
    # 加载训练数据
    data = np.load(file_data)
    lab = np.load(file_lab)
    N, D = np.shape(data)

    # 构造 one-hot 标签
    lab_onehot = np.zeros([N, 13])
    for i in range(N):
        id = int(lab[i, 0])
        lab_onehot[i, id] = 1
    data = (data.astype(float) / 255.0)
    return data, lab_onehot

def fead_forward(datas, layers):
    input_layers = []
    input_acfun = []
    for i in range(len(layers)):
        layer = layers[i]
        if i == 0:
            inputs = datas
            z = np.dot(inputs, layer["w"]) + layer["b"]  # 线性变化
            h = layer['act_fun'](z)  # 激活函数
            input_layers.append(inputs)
            input_acfun.append(z)
        else:
            inputs = h
            z = np.dot(inputs, layer["w"]) + layer["b"]
            h = layer['act_fun'](z)
            input_layers.append(inputs)
            input_acfun.append(z)
    return input_layers, input_acfun, h

def updata_wb(datas, labs, layers, loss_fun, de_loss_fun, alpha=0.01):
    # N, D = np.shape(datas)
    # 进行前馈操作
    inputs, input_acfun, output = fead_forward(datas, layers)
    # 计算 loss
    loss = loss_fun(output, labs)
    # 从后向前计算
    deltas0 = de_loss_fun(output, labs)
    # 从后向前计算误差
    deltas = []
    for i in range(len(layers)):
        index = -i - 1
        if i == 0:
            h = output
            z = input_acfun[index]
            delta = deltas0 * layers[index]["de_act_fun"](z, h)
        else:
            h = inputs[index + 1]
            z = input_acfun[index]
            # print(layers[index]["de_act_fun"](z,h)[1])
            delta = np.dot(delta, layers[index + 1]["w"].T) * layers[index]["de_act_fun"](z, h)

        deltas.insert(0, delta)

    # 利用误差 对每一层的权重进行修正
    for i in range(len(layers)):
        # 计算 dw 与 db
        dw = np.dot(inputs[i].T, deltas[i])
        db = np.sum(deltas[i], axis=0, keepdims=True)
        # 梯度下降
        layers[i]["w"] = layers[i]["w"] - alpha * dw
        layers[i]["b"] = layers[i]["b"] - alpha * db

    return layers, loss

def test_accuracy(datas, labs_true, layers):
    _, _, output = fead_forward(datas, layers)
    lab_det = np.argmax(output, axis=1)
    labs_true = np.argmax(labs_true, axis=1)
    N_error = np.where(np.abs(labs_true - lab_det) > 0)[0].shape[0]

    error_rate = N_error / np.shape(datas)[0]
    return error_rate

if __name__ == "__main__":

    # 加载训练数据
    train_data, train_lab_onehot = load_mnist("./npy/train_data3.npy", "./npy/train_label3.npy")
    N, D = np.shape(train_data)
    # 加载测试数据
    test_data, test_lab_onehot = load_mnist("./npy/test_data3.npy", "./npy/test_label3.npy")
    # 载入先前训好的权重
    # layers = np.load("model.npy", allow_pickle=True)

    # 搭建网络
    # 定义网络结构
    list_num_hidden = [70, 30, 13]
    #
    # list_act_funs =[sigmod,sigmod,sigmod,no_active]
    # list_de_act_funs=[de_sigmoid,de_sigmoid,de_sigmoid,de_no_active]

    # # 定义损失函数
    # loss_fun = loss_L2
    # de_loss_fun=de_loss_L2

    list_act_funs = [relu, relu, relu, no_active]
    list_de_act_funs = [de_relu, de_relu, de_relu, de_no_active]
    # 定义损失函数
    loss_fun = loss_CE
    de_loss_fun = de_loss_CE

    layers = bulid_net(D, list_num_hidden,
                       list_act_funs, list_de_act_funs)

    # 进行训练
    n_epoch = 200
    batchsize = 8
    N_batch = N // batchsize
    ERR = 100
    E_test = []
    E_train = []
    for i in range(n_epoch):
        # 数据打乱
        rand_index = np.random.permutation(N).tolist()
        # 每个batch 更新一下weight
        loss_sum = 0
        for j in range(N_batch):
            index = rand_index[j * batchsize:(j + 1) * batchsize]
            batch_datas = train_data[index]
            batch_labs = train_lab_onehot[index]
            layers, loss = updata_wb(batch_datas, batch_labs, layers, loss_fun, de_loss_fun, alpha=0.001)
            # print("epoch %d  batch %d  loss %.2f"%(i,j,loss/batchsize))
            loss_sum = loss_sum + loss

        error_train = test_accuracy(train_data, train_lab_onehot, layers)
        E_train.append(error_train)
        error_test = test_accuracy(test_data, test_lab_onehot, layers)
        E_test.append(error_test)
        print("epoch %d  error_train  %.2f%%  loss_all %.2f error_test %.2f%%" % (i, error_train * 100, loss_sum / (N_batch * batchsize), error_test * 100))
        if(error_test < ERR):
            ERR = error_test
            np.save("model.npy", layers)
    print(ERR)
    layers = np.load("model.npy", allow_pickle=True)
    error = test_accuracy(test_data, test_lab_onehot, layers)
    print("Accuarcy on Test Data %.2f %%" % ((1 - error) * 100))

    fig, ax = plt.subplots()
    lw = 1
    # 在生成的坐标系下画折线图
    x = np.array(range(0, n_epoch))
    ax.plot(x, E_train, color = "red", linewidth=lw)
    ax.plot(x, E_test, color = "blue", linewidth=lw)
    # 显示图形
    plt.show()

