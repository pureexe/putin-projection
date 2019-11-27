import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from math import sqrt

def compute_similarity_transform(points_static, points_to_transform):
    #http://nghiaho.com/?page_id=671
    p0 = np.copy(points_static).T
    p1 = np.copy(points_to_transform).T

    t0 = -np.mean(p0, axis=1).reshape(3,1)
    t1 = -np.mean(p1, axis=1).reshape(3,1)
    t_final = t1 -t0

    p0c = p0+t0
    p1c = p1+t1

    covariance_matrix = p0c.dot(p1c.T)
    U,S,V = np.linalg.svd(covariance_matrix)
    R = U.dot(V)
    if np.linalg.det(R) < 0:
        R[:,2] *= -1

    #rms_d0 = np.sqrt(np.mean(np.linalg.norm(p0c, axis=0)**2))
    #rms_d1 = np.sqrt(np.mean(np.linalg.norm(p1c, axis=0)**2))

    #s = (rms_d0/rms_d1)
    #P = np.c_[s*np.eye(3).dot(R), t_final] #original code
    P = np.c_[R, t_final]
    return P


def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    num_rows, num_cols = A.shape

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = B.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # subtract mean
    # Am = A - np.tile(centroid_A, (1, num_cols))
    # Bm = B - np.tile(centroid_B, (1, num_cols))
    Am = A.copy()
    Bm = B.copy()
    Am[0,:] = Am[0,:] - centroid_A[0]
    Am[1,:] = Am[1,:] - centroid_A[1]
    Am[2,:] = Am[2,:] - centroid_A[2]
    
    Bm[0,:] = Bm[0,:] - centroid_B[0]
    Bm[1,:] = Bm[1,:] - centroid_B[1]
    Bm[2,:] = Bm[2,:] - centroid_B[2]

    #rms_a = np.sqrt(np.mean(np.linalg.norm(Am, axis=0)**2))
    #rms_b = np.sqrt(np.mean(np.linalg.norm(Bm, axis=0)**2))
    rms_a = np.mean(np.linalg.norm(Am, axis=0))
    rms_b = np.mean(np.linalg.norm(Bm, axis=0))
    scale = rms_b / rms_a 
    Am = Am * scale

    # dot is matrix multiplication for array
    #H = np.matmul(Am,Bm.T)
    H = np.matmul(Bm,Am.T)
    # find rotation
    U, S, V = np.linalg.svd(H)
    #R = np.matmul(V,U.T)
    R = U.dot(V)
    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n");
        R[:,2] *= -v1
        #V[2,:] *= -1
        #R = np.matmul(V.T,U.T)
        
        """
    print(centroid_A)
    print(centroid_B)
    print(np.matmul(R,centroid_A) + t)
    exit()
    """
    #t = np.matmul(-R,centroid_A) + centroid_B
    t = centroid_B
    return R, t, scale



image_id = '0011'
mat = loadmat('data/{}_mesh.mat'.format(image_id))

# position
# position = mat['vertices']
canonical = np.load('data/canonical_vertices_righthand_far.npy')
canonical[:,2] = canonical[:,2] - 2000 
canonical = canonical * 107.0 / 128.0

#Intrinsic
focal = 2000
height = 256
intrinsic = np.array([
    [focal*height / 2,                 0,   height/2],
    [               0,  focal*height / 2,   height/2],
    [               0,                 0,        1]
])

# vertices
predict = mat['vertices']
predict = predict - 128
predict = predict / 128.0
predict[:2] = -predict[:2]
predict[:,2] = predict[:,2]  + 2000


#Extrinsic
#R,T = rigid_transform_3D(predict.T,canonical.T)
R,T,scale = rigid_transform_3D(canonical.T,predict.T)
print(R)
exit()
cam_matrix = np.c_[R, T]
#exit()
#cam_matrix = compute_similarity_transform(vertices,position)
extrinsic = np.eye(4)
extrinsic[:3,:4] = cam_matrix

#create color
color = mat['colors']

#projection
canonical_subtract = canonical.copy()
centroid_canonical = np.mean(canonical, axis=1)
canonical_subtract[:,0] = canonical_subtract[:,0] - centroid_canonical[0]
canonical_subtract[:,1] = canonical_subtract[:,1] - centroid_canonical[1]
canonical_subtract[:,2] = canonical_subtract[:,2] - centroid_canonical[2]
canonical_subtract = canonical_subtract * scale
n_position = np.ones((4,canonical_subtract.shape[0]))
n_position[:3,:] = canonical.T


#projected = np.matmul(intrinsic,position.T)

projected = np.matmul(intrinsic,np.matmul(extrinsic, n_position)[0:3,:])
projected = (projected / projected[2,:]).T
projected = projected.astype(np.int32)
#projected[:,0] = projected[:,0] - 56649
#projected[:,1] = projected[:,1] - 35975

image = np.zeros((256,256,3)) 
print(projected)
for i in range(len(projected)):
    try:
        u,v,_ = projected[i]
        image[v,u,:] = color[i]
    except:
        pass
plt.imshow(image)
#plt.imsave('output/out_{}.png'.format(image_id),image)
plt.show()
