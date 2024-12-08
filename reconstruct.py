def eightpoint(pts1, pts2, M):



    x1 = pts1[:,0] / M
    y1 = pts1[:,1] / M
    x2 = pts2[:,0] / M
    y2 = pts2[:,1] / M


    U = np.vstack((x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, np.ones(np.shape(x1)))).T


    _, _, Vt = np.linalg.svd(U)
    F = Vt[-1].reshape(3, 3)


    F = _singularize(F)
    F = refineF(F, pts1 / M, pts2 / M)


    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    F = T.T @ F @ T


    F /= F[2, 2]

    return F


def essentialMatrix(F, K1, K2):

  E = K2.T @ F @ K1


  E = E / np.linalg.norm(E)


  E = E / E[2,2]

  return E

def triangulate(C1, pts1, C2, pts2):
    N = pts1.shape[0]
    P = np.zeros((N, 3))
    err = 0
    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A = np.vstack([
            x1 * C1[2] - C1[0],
            y1 * C1[2] - C1[1],
            x2 * C2[2] - C2[0],
            y2 * C2[2] - C2[1]
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]
        P[i] = X[:3]

        p1_homogen = C1 @ X
        p2_homogen = C2 @ X
        p1 = p1_homogen[:2] / p1_homogen[2]
        p2 = p2_homogen[:2] / p2_homogen[2]
        err += np.sum((pts1[i] - p1)**2) + np.sum((pts2[i] - p2)**2)


    err = np.sqrt(err / (2 * N))

    return P, err
def camera2(E):
  """helper function to find the 4 possibile M2 matrices"""
  U,S,V = np.linalg.svd(E)
  m = S[:2].mean()
  E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
  U,S,V = np.linalg.svd(E)
  W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

  if np.linalg.det(U.dot(W).dot(V))<0:
      W = -W

  M2s = np.zeros([3,4,4])
  M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  return M2s


def findM2(F, pts1, pts2, intrinsics):


    K1, K2 = intrinsics['K1'], intrinsics['K2']

 
    #E = essentialMatrix(F, K1, K2)

    E = K2.T @ F @ K1


    M2s =  camera2(E)
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    C1 = K1.dot(M1)

    best_error = float('inf')
    best_M2 = None
    best_C2 = None
    best_P = None

    for i in range(4):
        M2 = M2s[:, :, i]
        C2 = K2.dot(M2)

        P, err = triangulate(C1, pts1, C2, pts2)
        # print(f'Size of P: {np.size(P)}')
        # print(f'Shape of P: {P.shape}')
        # print(f'P:{P}')

        if np.all(P[:, 2] > 0):

            best_M2 = M2
            best_C2 = C2
            best_P = P
            # print(P)
    M2 = best_M2
    C2 = K2.dot(best_M2)
    # P, err = triangulate(C1, pts1, C2, pts2)

    # print(f'best error: {best_error}')
    # print(f'M2:\n{M2}')
    # print(f'C2:\n{C2}')
    # print(f'P:\n{P}')
    ###



    return M2, C2, best_P

def epipolarCorrespondence(im1, im2, F, x1, y1, window_size=15):

    coord_x = int(x1)
    coord_y = int(y1)
    window_dim = 20

    img1_window = im1[coord_y - window_dim//2 : coord_y + window_dim//2 + 1,
                      coord_x - window_dim//2 : coord_x + window_dim//2 + 1, :]

    height_img2, width_img2, _ = im2.shape

    point = np.array([coord_x, coord_y, 1])

    epi_line = np.dot(F, point)
    coef_a, coef_b, coef_c = epi_line

    y_range = np.arange(height_img2)
    x_range = np.rint(-(coef_b*y_range + coef_c) / coef_a)

    window_range = np.arange(-window_dim//2, window_dim//2+1, 1)
    grid_x, grid_y = np.meshgrid(window_range, window_range)
    sigma = 7
    gaussian = np.exp(-((grid_x**2 + grid_y**2) / (2 * (sigma**2))))
    gaussian /= np.sum(gaussian)
    min_error = float('inf')
    best_x, best_y = -1, -1

    gaussian_3d = np.repeat(gaussian[:, :, np.newaxis], 3, axis=2)

    for y_candidate in range(max(0, coord_y - 100), min(height_img2, coord_y + 100)):
        x_candidate = int((-coef_b * y_candidate - coef_c) / coef_a)
        if window_dim//2 <= x_candidate < width_img2 - window_dim//2 and window_dim//2 <= y_candidate < height_img2 - window_dim//2:
            img2_window = im2[y_candidate-window_dim//2:y_candidate+window_dim//2+1,
                              x_candidate-window_dim//2:x_candidate+window_dim//2+1, :]
            error = np.sum(np.linalg.norm((img1_window - img2_window) * gaussian_3d))
            if error < min_error:
                min_error = error
                best_x = x_candidate
                best_y = y_candidate

    return best_x, best_y

def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):

    K1, K2 = intrinsics['K1'], intrinsics['K2']


    temple_pts2 = np.zeros_like(temple_pts1)
    for i, (x1, y1) in enumerate(temple_pts1):

        x2, y2 = epipolarCorrespondence(im1, im2, F, x1, y1)
        temple_pts2[i] = [x2, y2]





    M2, C2, P = findM2(F, temple_pts1, temple_pts2, intrinsics)



    return P
