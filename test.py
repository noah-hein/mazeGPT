import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

arr = []
for i in range(100):
    c = np.random.rand(10, 10)
    arr.append(c)
plt.imshow(arr[45])
plt.show()

fig = plt.figure()
i=0
im = plt.imshow(arr[0], animated=True)
def updatefig(*args):
    global i
    if (i<99):
        i += 1
    else:
        i=0
    im.set_array(arr[i])
    return im,


ani = animation.FuncAnimation(fig, updatefig,  blit=True)
plt.show()
