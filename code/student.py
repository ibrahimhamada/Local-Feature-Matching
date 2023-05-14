import numpy as np
from cmath import pi
from skimage.filters import gaussian
from skimage.feature import peak_local_max



def get_interest_points(image, feature_width):
    """
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    """

    k = 0.04
    threshold = 0.01

    smoothed_image = gaussian(image, 0.1)
    dy, dx = np.gradient(smoothed_image)

    Ixx = gaussian(dx ** 2, 0.1)
    Ixy = gaussian(dy * dx, 0.1)
    Iyy = gaussian(dy ** 2, 0.1)

    y_range = image.shape[0] - feature_width
    x_range = image.shape[1] - feature_width

    out = np.zeros([image.shape[0], image.shape[1]])
    for y in range(0, y_range, 2):
        for x in range(0, x_range, 2):
            # Values of sliding window
            start_y = y
            end_y = y + feature_width + 1
            start_x = x
            end_x = x + feature_width + 1

            # The variable names are representative to
            # the variable of the Harris corner equation
            windowIxx = Ixx[start_y: end_y, start_x: end_x]
            windowIxy = Ixy[start_y: end_y, start_x: end_x]
            windowIyy = Iyy[start_y: end_y, start_x: end_x]

            # Sum of squares of intensities of partial derevatives
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            # Calculate determinant and trace of the matrix
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy

            # Calculate r for Harris Corner equation
            r = det - k * (trace ** 2)

            if r > threshold:
                out[y + feature_width // 2, x + feature_width // 2] = r

    s = peak_local_max(out, 3, 0.01)
    xs = s[:, 1]
    ys = s[:, 0]
    return np.round(xs).astype(int), np.round(ys).astype(int)


def get_features(image, x, y, feature_width):
    """
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions
    features = np.zeros((len(x), 4, 4, 8))          # Initialize features tensor
    smoothed_img = gaussian(image, 0.1)     # Remove the noise from the image using LPF
    dy, dx = np.gradient(smoothed_img)              # Find Gradient of the image
    grad_magnitude = np.sqrt(dx ** 2 + dy ** 2)     # Magnitude of the Gradient
    grad_direction = np.arctan2(dy, dx)             # Direction of the Gradient
    grad_direction[grad_direction < 0] += 2 * pi

    for featureLoc, (x_, y_) in enumerate(zip(x, y)):
        y_ind = y_ - feature_width // 2
        y2_ind = y_ + feature_width // 2 + 1
        x_ind = x_ - feature_width // 2
        x2_ind = x_ + feature_width // 2 + 1
        i = (y_ind, y2_ind)
        j = (x_ind, x2_ind)

        if i[0] < 0:
            i = (0, feature_width + 1)
        if j[0] < 0:
            j = (0, feature_width + 1)

        if i[1] > image.shape[0]:
            i = (image.shape[0] - feature_width - 1, image.shape[0] - 1)
        if j[1] > image.shape[1]:
            j = (image.shape[1] - feature_width - 1, image.shape[1] - 1)

        window_mag = grad_magnitude[i[0]:i[1], j[0]:j[1]]
        window_dir = grad_direction[i[0]:i[1], j[0]:j[1]]
        window_mag = gaussian(window_mag, 0.4)
        window_dir = gaussian(window_dir, 0.4)
        c = feature_width // 4

        for i in range(c):
            for j in range(c):
                start_i = i * c
                end_i = (i + 1) *c
                start_j = j * c
                end_j = (j + 1) * c
                mag_c = window_mag[start_i: end_i, start_j: end_j]
                dir_c = window_dir[start_i: end_i, start_j: end_j]
                features[featureLoc, i, j] = np.histogram(dir_c.reshape(-1), bins=8, range=(0, 2 * pi), weights=mag_c.reshape(-1))[0]

    features = features.reshape((len(x), -1,))
    mag = np.sqrt((features**2).sum(axis=1))
    mag = mag.reshape(-1, 1)
    features = features / mag
    features[features >= 0.2] = 0.2
    mag = np.sqrt((features**2).sum(axis=1))
    mag = mag.reshape(-1, 1)
    features = features / mag

    return features


def match_features(im1_features, im2_features):
    """
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions
    matches = []
    confidences = []

    for i in range(im1_features.shape[0]):
        distances = np.sqrt(((im1_features[i, :] - im2_features) ** 2).sum(axis=1))
        sorted_index = np.argsort(distances)
        ratio = distances[sorted_index[0]] / distances[sorted_index[1]]

        if ratio < 0.8:
            matches.append([i, sorted_index[0]])
            confidences.append(1.0 - ratio)

    return np.asarray(matches), np.asarray(confidences)
