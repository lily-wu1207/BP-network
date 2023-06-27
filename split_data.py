import os
from shutil import copy, rmtree
import random
import matplotlib.pyplot as plt
import numpy as np


def rgb2gray(rgb):
    # return np.dot(rgb[...,:3],[0.229,0.587,0.114])
    return np.dot(rgb[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)


def main():
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.1


    cwd = os.getcwd()
    data_root = os.path.join(cwd, "words")
    origin_path = os.path.join(data_root, "data")
    assert os.path.exists(origin_path), "path '{}' does not exist.".format(origin_path)

    word_class = [cla for cla in os.listdir(origin_path)
                    if os.path.isdir(os.path.join(origin_path, cla))]
    # 分了5个训练集+测试集验证准确性是否与数据选取有关
    for j in range(5):
        test_data = []
        test_label = []
        train_data = []
        train_label = []
        for cla in word_class:
            label = cla
            cla_path = os.path.join(origin_path, cla)
            images = os.listdir(cla_path)
            num = len(images)
            # 随机采样验证集的索引
            eval_index = random.sample(images, k=int(num*split_rate))
            for index, image in enumerate(images):
                l = [np.array([label])]
                if image in eval_index:
                    # 将分配至验证集中的文件复制到相应目录
                    image_path = os.path.join(cla_path, image)
                    t = plt.imread(image_path)

                    gray_word = [rgb2gray(t).flatten()]
                    if(test_data == []):
                        test_data = gray_word
                    else:
                        test_data = np.concatenate((test_data, gray_word), axis=0)
                    if(test_label == []):
                        test_label = l
                    else:
                        test_label = np.concatenate((test_label, l), axis=0)


                else:
                    # 将分配至训练集中的文件复制到相应目录
                    image_path = os.path.join(cla_path, image)
                    t = plt.imread(image_path)

                    gray_word = [rgb2gray(t).flatten()]
                    if (train_data == []):
                        train_data = gray_word
                    else:
                        train_data = np.concatenate((train_data, gray_word), axis=0)
                    if (train_label == []):
                        train_label = l
                    else:
                        train_label = np.concatenate((train_label, l), axis=0)
                print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar

            print()
        # shuffle
        N_test = len(test_data)
        Test_data = []
        Test_label = []
        rand_index = np.random.permutation(N_test).tolist()
        for i in range(N_test):
            Test_data.append(test_data[rand_index[i]])
            Test_label.append((test_label[rand_index[i]]))
        N_train = len(train_data)
        Train_data = []
        Train_label = []
        rand_index = np.random.permutation(N_train).tolist()
        for i in range(N_train):
            Train_data.append(train_data[rand_index[i]])
            Train_label.append((train_label[rand_index[i]]))
        np.save('test_data'+str(j), Test_data)
        np.save('train_data'+str(j), Train_data)
        np.save('test_label'+str(j), Test_label)
        np.save('train_label'+str(j), Train_label)
    print("processing done!")


if __name__ == '__main__':
    main()
