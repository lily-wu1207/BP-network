import numpy as np
import os
from minist import *
layers = np.load("model_91.26.npy", allow_pickle=True)
# 处理数据

def rgb2gray(rgb):
    # return np.dot(rgb[...,:3],[0.229,0.587,0.114])
    return np.dot(rgb[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
# 指向测试文件夹
cwd = './'
# data_root = os.path.join(cwd, "words")
origin_path = os.path.join(cwd, "test_data")
assert os.path.exists(origin_path), "path '{}' does not exist.".format(origin_path)

# 将test的图片转化为npy文件
test_data = []
test_label = []
word_class = [cla for cla in os.listdir(origin_path)
              if os.path.isdir(os.path.join(origin_path, cla))]
for cla in word_class:
    label = cla
    cla_path = os.path.join(origin_path, cla)
    images = os.listdir(cla_path)
    num = len(images)
    for index, image in enumerate(images):
        l = [np.array([label])]

        image_path = os.path.join(cla_path, image)
        t = plt.imread(image_path)

        gray_word = [rgb2gray(t).flatten()]
        if (test_data == []):
            test_data = gray_word
        else:
            test_data = np.concatenate((test_data, gray_word), axis=0)
        if (test_label == []):
            test_label = l
        else:
            test_label = np.concatenate((test_label, l), axis=0)

# shuffle
N_test = len(test_data)
Test_data = []
Test_label = []
rand_index = np.random.permutation(N_test).tolist()
for i in range(N_test):
    Test_data.append(test_data[rand_index[i]])
    Test_label.append((test_label[rand_index[i]]))

np.save('test_data' , Test_data)
np.save('test_label' , Test_label)

print("processing done!")

test_data, test_lab_onehot = load_mnist("test_data.npy", "test_label.npy")
# train_data, train_lab_onehot = load_mnist("train_data1.npy", "train_label1.npy")
error_test = test_accuracy(test_data, test_lab_onehot, layers)
# error_train = test_accuracy(train_data, train_lab_onehot, layers)
print('test:', 1 - error_test)
