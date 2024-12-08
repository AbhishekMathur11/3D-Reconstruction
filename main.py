import utils
import reconstruct
import visualization
import bundle_adjustment
import multi_view_reconstruct


#Data
if not os.path.exists('data'):
  !wget https://www.andrew.cmu.edu/user/eweng/data.zip -O data.zip
  !unzip -qq "data.zip"
  print("downloaded and unzipped data")

# the points in im1, whose correnponding epipolar line in im2 you'd like to verify
# point = [(50,190),(200, 200), (400,180), (350,350), (520, 220)]
point = [(100, 165), (180, 250), (330, 220), (341, 332), (520, 110)]
# feel free to change these point, to verify different point correspondences
displayEpipolarF(im1, im2, F, point)
correspondence = np.load('data/some_corresp.npz') # Loading correspondences
intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
K1, K2 = intrinsics['K1'], intrinsics['K2']
pts1, pts2 = correspondence['pts1'], correspondence['pts2']
im1 = plt.imread('data/im1.png')
im2 = plt.imread('data/im2.png')

F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
E = essentialMatrix(F, K1, K2)
print(f'recovered E:\n{E.round(4)}')

assert(E[2, 2] == 1)
assert(np.linalg.matrix_rank(E) == 2)

correspondence = np.load('data/some_corresp.npz') 
intrinsics = np.load('data/intrinsics.npz') 
K1, K2 = intrinsics['K1'], intrinsics['K2']
pts1, pts2 = correspondence['pts1'], correspondence['pts2']
im1 = plt.imread('data/im1.png')
im2 = plt.imread('data/im2.png')

F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

M2, C2, P = findM2(F, pts1, pts2, intrinsics)

M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
C1 = K1.dot(M1)
C2 = K2.dot(M2)
P_test, err = triangulate(C1, pts1, C2, pts2)
assert(err < 500)
correspondence = np.load('data/some_corresp.npz') # Loading correspondences
intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
K1, K2 = intrinsics['K1'], intrinsics['K2']
pts1, pts2 = correspondence['pts1'], correspondence['pts2']
im1 = plt.imread('data/im1.png')
im2 = plt.imread('data/im2.png')


F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

# Simple Tests to verify your implementation:
x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
assert(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10)


# the points in im1 whose correnponding epipolar line in im2 you'd like to verify
points = [(50,190), (200, 200), (400,180), (350,350), (520, 220)]
# feel free to change these points to verify different point correspondences
epipolarMatchGUI(im1, im2, F, points, epipolarCorrespondence)
temple_coords = np.load('data/templeCoords.npz') # Loading temple coordinates
correspondence = np.load('data/some_corresp.npz') # Loading correspondences
intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
K1, K2 = intrinsics['K1'], intrinsics['K2']
pts1, pts2 = correspondence['pts1'], correspondence['pts2']
im1 = plt.imread('data/im1.png')
im2 = plt.imread('data/im2.png')


temple_pts1 = np.hstack([temple_coords['x1'], temple_coords['y1']])

M = max(im1.shape)
F = eightpoint(pts1, pts2, M)


P = compute3D_pts(temple_pts1, {'K1': K1, 'K2': K2}, F, im1, im2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=10, c='c', depthshade=True)
plt.draw()

# also show a different viewpoint
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=10, c='c', depthshade=True)
ax.view_init(30, 0)
plt.draw()
# Visualization:
np.random.seed(1)
correspondence = np.load('data/some_corresp_noisy.npz') # Loading noisy correspondences
intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
K1, K2 = intrinsics['K1'], intrinsics['K2']
pts1, pts2 = correspondence['pts1'], correspondence['pts2']
im1 = plt.imread('data/im1.png')
im2 = plt.imread('data/im2.png')
M=np.max([*im1.shape, *im2.shape])




F, inliers = ransacF(pts1, pts2, M, nIters=1000, tol=1)


M1 = np.hstack((np.eye(3), np.zeros((3, 1))))

M2_init, C2_init, P_init = findM2(F, pts1[inliers.flatten()], pts2[inliers.flatten()], intrinsics)

M2_opt, P_opt, obj_start, obj_end = bundleAdjustment(K1, M1, pts1[inliers.flatten()], K2, M2_init, pts2[inliers.flatten()], P_init)


print(f"Before reprojection error: {obj_start}, After: {obj_end}")

plot_3D_dual(P_init, P_opt, azim=0, elev=0)
plot_3D_dual(P_init, P_opt, azim=70, elev=40)
plot_3D_dual(P_init, P_opt, azim=40, elev=40)


pts_3d_video = []
for loop in range(10):
  print(f"processing time frame - {loop}")

  data_path = os.path.join('data/q6/','time'+str(loop)+'.npz')
  image1_path = os.path.join('data/q6/','cam1_time'+str(loop)+'.jpg')
  image2_path = os.path.join('data/q6/','cam2_time'+str(loop)+'.jpg')
  image3_path = os.path.join('data/q6/','cam3_time'+str(loop)+'.jpg')

  im1 = plt.imread(image1_path)
  im2 = plt.imread(image2_path)
  im3 = plt.imread(image3_path)

  data = np.load(data_path)
  pts1 = data['pts1']
  pts2 = data['pts2']
  pts3 = data['pts3']

  K1 = data['K1']
  K2 = data['K2']
  K3 = data['K3']

  M1 = data['M1']
  M2 = data['M2']
  M3 = data['M3']

  if loop == 0 or loop==9:  # feel free to modify to visualize keypoints at other loop timesteps
    img = visualize_keypoints(im2, pts2)

  C1 = K1 @ M1
  C2 = K2 @ M2
  C3 = K3 @ M3


  pts_3d, err = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3)

  pts_3d_video.append(pts_3d)

  print(f"Reprojection error for frame {loop}: {err}")



  if loop == 0:
    plot_3d_keypoint(pts_3d)

plot_3d_keypoint_video(pts_3d_video)
