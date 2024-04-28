import numpy as np

def saturate(x, int_max, int_min):
    return np.clip(x, int_min, int_max)

# 求缩放 偏移
def scale_z_cal(x, int_max, int_min):
    scale = (x.max() - x.min())/(int_max - int_min)
    z = int_max - np.round(x.max()/scale)
    return scale, z

# 进行量化
def quant_float_data(x, scale, z, int_max, int_min):
    xq = saturate(np.round(x/scale + z), int_min = int_min, int_max = int_max)
    # xq = np.round(x/scale + z)
    return xq

# 进行反量化
def dequant_data(xq, scale, z):
    x = ((xq - z) * scale).astype('float32')
    return x


if __name__ == '__main__':
    data_float32 = np.random.randn(3).astype('float32')
    print("input:", data_float32)
    int_max = 127
    int_min = -128
    scale, z = scale_z_cal(data_float32, int_max, int_min)
    print("scale and z", scale, z)
    data_int8 = quant_float_data(data_float32, scale, z, int_max, int_min)
    print("quant result", data_int8)
    data_dequant_float = dequant_data(data_int8, scale, z)
    print("dequant result", data_dequant_float)

    print('diff', data_dequant_float - data_float32)
