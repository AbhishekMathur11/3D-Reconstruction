def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=100):

    N = pts1.shape[0]
    P = np.zeros((N, 3))
    err = 0

    for i in range(N):

        if pts1[i, 2] > Thres and pts2[i, 2] > Thres and pts3[i, 2] > Thres:

            x1, y1 = pts1[i, :2]
            x2, y2 = pts2[i, :2]
            x3, y3 = pts3[i, :2]


            A = np.vstack([
                x1 * C1[2] - C1[0],
                y1 * C1[2] - C1[1],
                x2 * C2[2] - C2[0],
                y2 * C2[2] - C2[1],
                x3 * C3[2] - C3[0],
                y3 * C3[2] - C3[1]
            ])


            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X / X[3]


            P[i] = X[:3]


            p1 = C1 @ X
            p2 = C2 @ X
            p3 = C3 @ X
            p1 = p1[:2] / p1[2]
            p2 = p2[:2] / p2[2]
            p3 = p3[:2] / p3[2]
            err += np.sum((pts1[i, :2] - p1)**2 + (pts2[i, :2] - p2)**2 + (pts3[i, :2] - p3)**2)
        else:

            P[i] = [0, 0, 0]


    err = np.sqrt(err / (N * 3))


    return P, err

def plot_3d_keypoint_video(pts_3d_video):
  '''
  Plot Spatio-temporal (3D) keypoints
      :param car_points: np.array points * 3
  '''

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for pts_3d in pts_3d_video:
      num_points = pts_3d.shape[1]
      for j in range(len(connections_3d)):
          index0, index1 = connections_3d[j]
          xline = [pts_3d[index0,0], pts_3d[index1,0]]
          yline = [pts_3d[index0,1], pts_3d[index1,1]]
          zline = [pts_3d[index0,2], pts_3d[index1,2]]
          ax.plot(xline, yline, zline, color=colors[j])
  np.set_printoptions(threshold=1e6, suppress=True)
  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')
  plt.show()
