import numpy as np

def saturate(x, int_min, int_max):
    return np.clip(x, int_min, int_max)

# 最大的数字除127
def scale_cal(x, int_max):
    max_val = np.max(np.abs(x))
    return max_val/127

# 归一化后 乘127 取整
def quant_float_data(x, scale, int_min, int_max):
    xq = np.round(x / scale)
    return saturate(xq, int_min=int_min, int_max=int_max)

# 还原
def dequant_data(xq, scale):
    return (xq*scale).astype('float32')

if __name__ == '__main__':
    data_float32 = np.random.randn(3).astype('float32')
    int_max = 127
    int_min = -128
    scale = scale_cal(data_float32, int_max)
    xq = quant_float_data(data_float32, scale, int_min, int_max)
    dq = dequant_data(xq, scale)
    print(data_float32)
    print(dq)
# [0.2,0.4,0.43534534543,0.5]
# [0.2/0.5,0.4-/0.5,...,1]
# []
