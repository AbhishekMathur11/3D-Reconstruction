
def ransacF(pts1, pts2, M, nIters=100, tol=10):
  '''
  Input:  pts1, Nx2 Matrix
          pts2, Nx2 Matrix
          M, a scaler parameter
          nIters, Number of iterations of the Ransac
          tol, tolerence for inliers
  Output: F, the fundamental matrix
          inliers, Nx1 bool vector set to true for inliers

  '''
  N = pts1.shape[0]
  pts1_homo, pts2_homo = toHomogenous(pts1), toHomogenous(pts2)
  best_inlier = 0
  inlier_curr = None

  for i in range(nIters):
      choice = np.random.choice(range(pts1.shape[0]), 8)
      pts1_choice = pts1[choice, :]
      pts2_choice = pts2[choice, :]
      F = eightpoint(pts1_choice, pts2_choice, M)
      ress = calc_epi_error(pts1_homo, pts2_homo, F)
      curr_num_inliner = np.sum(ress < tol)
      if curr_num_inliner > best_inlier:
          F_curr = F
          inlier_curr = (ress < tol)
          best_inlier = curr_num_inliner
  inlier_curr = inlier_curr.reshape(inlier_curr.shape[0], 1)
  indixing_array = inlier_curr.flatten()
  pts1_inlier = pts1[indixing_array]
  pts2_inlier = pts2[indixing_array]
  F = eightpoint(pts1_inlier, pts2_inlier, M)
  return F, inlier_curr
def rodrigues(r):
  '''
      Input:  r, a 3x1 vector
      Output: R, a rotation matrix
  '''

  r = np.array(r).flatten()
  I = np.eye(3)
  theta = np.linalg.norm(r)
  if theta == 0:
      return I
  else:
      U = (r/theta)[:, np.newaxis]
      Ux, Uy, Uz = r/theta
      K = np.array([[0, -Uz, Uy], [Uz, 0, -Ux], [-Uy, Ux, 0]])
      R = I * np.cos(theta) + np.sin(theta) * K + \
          (1 - np.cos(theta)) * np.matmul(U, U.T)
  return R


def invRodrigues(R):
  '''
  Input:  R, a rotation matrix
  Output: r, a 3x1 vector
  '''

  def s_half(r):
      r1, r2, r3 = r
      if np.linalg.norm(r) == np.pi and (r1 == r2 and r1 == 0 and r2 == 0 and r3 < 0) or (r1 == 0 and r2 < 0) or (r1 < 0):
          return -r
      else:
          return r

  A = (R - R.T)/2
  ro = [A[2, 1], A[0, 2], A[1, 0]]
  s = np.linalg.norm(ro)
  c = (np.sum(np.matrix(R).diagonal()) - 1)/2
  if s == 0 and c == 1:
      r = np.zeros(3)
  elif s == 0 and c == -1:
      col = np.eye(3) + R
      col_idx = np.nonzero(
          np.array(np.sum(col != 0, axis=0)).flatten())[0][0]
      v = col[:, col_idx]
      u = v/np.linalg.norm(v)
      r = s_half(u * np.pi)
  else:
      u = ro/s
      theta = np.arctan2(s, c)
      r = u * theta

  return r

def rodriguesResidual(K1, M1, p1, K2, p2, x):
  '''
  Q5.1: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
  '''
  N = p1.shape[0]

  def project(K, M, P):
    """
    Projects 3D points P into 2D using the camera intrinsics (K) and extrinsics (M).

    Parameters:
    K : numpy.ndarray
        Camera intrinsic matrix (3x3).
    M : numpy.ndarray
        Camera extrinsic matrix (3x4).
    P : numpy.ndarray
        3D points (Nx3).

    Returns:
    numpy.ndarray
        2D projected points (Nx2).
    """

    P_homo = np.hstack((P, np.ones((P.shape[0], 1))))


    p_homo = K @ M @ P_homo.T


    p = p_homo[:2, :] / p_homo[2, :]

    return p.T
  P = x[:3*N].reshape(N, 3)
  r2 = x[3*N:3*N+3]
  t2 = x[3*N+3:3*N+6]


  p1_hat = project(K1, M1, P)


  R2 = rodrigues(r2)
  M2 = np.hstack((R2, t2.reshape(3, 1)))


  p2_hat = project(K2, M2, P)


  residuals = np.concatenate([(p1 - p1_hat).reshape(-1), (p2 - p2_hat).reshape(-1)])
  return residuals



def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):

  obj_start = obj_end = 0

  r2_init, t2_init = invRodrigues(M2_init[:, :3]), M2_init[:, 3]
  x_init = np.hstack((P_init.flatten(), r2_init.flatten(), t2_init.flatten()))


  obj_start = np.sum(rodriguesResidual(K1, M1, p1, K2, p2, x_init) ** 2)


  def objective_function(x):
      return np.sum(rodriguesResidual(K1, M1, p1, K2, p2, x) ** 2)

  result = minimize(objective_function, x_init, method='L-BFGS-B')


  x_optimized = result.x
  P_optimized = x_optimized[:3 * P_init.shape[0]].reshape(P_init.shape)
  r2_optimized = x_optimized[3 * P_init.shape[0]:3 * P_init.shape[0] + 3]
  t2_optimized = x_optimized[3 * P_init.shape[0] + 3:]


  obj_end = result.fun


  R2_optimized = rodrigues(r2_optimized)
  M2_optimized = np.hstack((R2_optimized, t2_optimized[:, np.newaxis]))

  return M2_optimized, P_optimized, obj_start, obj_end

def plot_3D_dual(P_before, P_after, azim=70, elev=45):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    ax.view_init(azim=azim, elev=elev)
    plt.draw()
