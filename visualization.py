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
