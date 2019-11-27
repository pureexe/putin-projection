import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

image_id = '0000'
mat = loadmat('data/{}_mesh.mat'.format(image_id))

#position
position = mat['vertices']
# position = np.load('data/canonical_vertices_righthand_far.npy')

#Intrinsic
focal = 2000
height = 256
intrinsic = np.array([
    [focal*height / 2,                 0,   height/2],
    [               0,  focal*height / 2,   height/2],
    [               0,                 0,        1]
])

#create color
color = mat['colors']
"""
z_max = np.max(position[:,2])
z_min = np.min(np.min(position[:,2]))
z_len = z_max - z_min
color = position[:,2]
color = (color - z_min)/ z_len
"""

#projection
position = position - 128
position = position / 128.0
position[:,2] = position[:,2] + 2000

print("0_min: {}".format(np.min(position[:,0])))
print("0_max: {}".format(np.max(position[:,0])))
print("1_min: {}".format(np.min(position[:,1])))
print("1_max: {}".format(np.max(position[:,1])))
print("2_min: {}".format(np.min(position[:,2])))
print("2_max: {}".format(np.max(position[:,2])))
projected = np.matmul(intrinsic,position.T)
projected = (projected / projected[2,:]).T
projected = projected.astype(np.int32)
image = np.zeros((256,256,3)) 
print(projected)
for i in range(len(projected)):
    try:
        u,v,_ = projected[i]
        image[v,u,:] = color[i]
    except:
        pass
plt.imshow(image)
plt.imsave('output/out_{}.png'.format(image_id),image)
plt.show()
