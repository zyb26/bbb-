import numpy as np
import matplotlib.pyplot as plt

def smooth_data(p, eps=0.0001):
    is_zeros = (p==0).astype(np.float32)
    is_nonezeros = (p!=0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonezeros = p.shape - n_zeros

    eps1 = eps*n_zeros/n_nonezeros
    hist = p.astype(np.float32)
    hist += eps*is_zeros + (-eps1)*is_nonezeros
    return hist



def cal_kl(p, q):
    KL = 0.
    for i in range(len(p)):
        KL += p[i]*np.log(p[i]/q[i])
    return KL

# def kl_test(x, kl_thresold=0.01):
#     y_out = []
#     while True:
#         y = [np.random.uniform(1, size+1) for i in range(size)]
#         y = y / np.sum(y)
#         kl_result = cal_kl(x, y)
#         if kl_result < kl_thresold:
#             print(kl_result)
#             y_out = y
#             plt.plot(x)
#             plt.plot(y)
#             break
#     return y

if __name__ == '__main__':
    # np.random.seed(1)
    # size = 10
    # x = [np.random.uniform(1, size+1) for i in range(size)]
    # x = x / np.sum(x)  # 概率生成
    # y_out = kl_test(x, kl_thresold=0.03)  # kl散度越小越接近
    # plt.show()
    # print(x, y_out)

    p = [1, 0, 2, 3, 5, 3, 1, 7]#参数的值
    bin = 4
    split_p = np.array_split(p, bin)  # 等频切割，两两一组。
    q = []
    for arr in split_p:
        avg = np.sum(arr)/np.count_nonzero(arr)#均值
        for item in arr:
            if item != 0:
                q.append(avg)
                continue
            q.append(0)#len(q)=len(p)
    print(q)
    p = p / np.sum(p)#归一化
    q = q / np.sum(q)#归一化
    p = smooth_data(p)
    q = smooth_data(q)
    print(cal_kl(p, q))
