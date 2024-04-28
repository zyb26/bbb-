import numpy as np
import matplotlib.pyplot as plt

data = np.random.rand(1000)
plt.hist(data, bins=20)

# plt.title("histogram")
# plt.xlabel('value')
# plt.ylabel('freq')
#
# plt.show()

def histgram_range(x):
    hist, range = np.histogram(x, 100)
    total = len(x)
    left = 0
    right = len(hist) - 1
    limit = 0.99
    # print(hist)
    # print(range)
    while True:
        cover_percent = hist[left:right].sum()/total
        if cover_percent <= limit:
            break
        if hist[left] > hist[right]:
            right -= 1
        else:
            left += 1
    left_val = range[left]
    right_val = range[right]
    dynamic_range = max(abs(left_val), abs(right_val))
    return dynamic_range/127

print(histgram_range(data))






