import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import msvcrt  # to detect keystrokes (windows)


class ThresholdParams:
    sobel = (True, False, 3)  # x, y, kernel size
    HLS = (False, False, True)
    YUV = (False, False, False)
    LAB = (False, False, False)
    thresSobel = ((40, 150), (70, 150))  # threshold X, Y
    thresHLS = ((100, 250), (100, 200), (130, 240))
    thresYUV = ((100, 250), (100, 200), (130, 240))
    thresLAB = ((100, 250), (100, 200), (130, 240))


# loads an rgb image
def loadRGBImage(path):
    # print("Load RGB image")
    img = mpimg.imread(path)
    # img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# converts an rgb image to grayscale
def rgb2gray(img):
    # print("Convert image to grayscale")
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# display an image, opcionally with color map = gray
def displayImage(img, gray=False, title=""):
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.title(title)
    plt.show()


# image distortion correction and returns the undistorted image
# inputs
#   img: rgb image
#   mtx, dist: calibration arrays obtained from calibrate_camera
# outputs
#   the image undistorted
def undistort_image(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# compute distortion matrix and save to pickle file
# inputs
#   img_dir: path to calibration images
#   nx, ny: number of inner corners on each image, horizontal and vertical
def calibrate_camera(img_dir, nx, ny):
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(img_dir + '/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = loadRGBImage(fname)
        gray = rgb2gray(img)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # # Draw and display the corners
            # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # write_name = 'corners_found'+str(idx)+'.jpg'
            # save images for the writeup
            # cv2.imwrite(write_name, img)
            #
            # cv2.imshow('img', img)
            # cv2.waitKey(1000)
        else:
            print("corners not found")
    # calibrating camera using test image
    test_img = loadRGBImage(base_dir + "/test_images/straight_lines1.jpg")
    displayImage(test_img)
    img_size = (test_img.shape[1], test_img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(base_dir + "/img_dir/calibration.p", "wb"))


# binary thresholding
def thresholding_img(img, sobel=(True, False, 3), thresSobel=((0, 255), (0, 255)),
                     HLS=(False, False, True), thresHLS=((0, 255), (0, 255), (0, 255)),
                     YUV=(False, False, False), thresYUV=((0, 255), (0, 255), (0, 255)),
                     LAB=(False, False, False), thresLAB=((0, 255), (0, 255), (0, 255))):
    sbin = np.zeros_like(img[:, :, 0])  # final image will have only one channel
    if sobel[0] or sobel[1]:
        img_gray = rgb2gray(img)
        if sobel[0]:
            sx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel[2])
            abs_sobelx = np.absolute(sx)
            scaled_sx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
            sbin[(scaled_sx >= thresSobel[0][0]) & (scaled_sx <= thresSobel[0][1])] = 1
            if debug: displayImage(sbin, gray=True, title="Sobel X thresholding")
        if sobel[1]:
            sy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel[2])
            abs_sobely = np.absolute(sy)
            scaled_sy = np.uint8(255 * abs_sobely / np.max(abs_sobely))
            sbin[(scaled_sy >= thresSobel[1][0]) & (scaled_sy <= thresSobel[1][1])] = 1
            if debug: displayImage(sbin, gray=True, title="Sobel Y thresholding")
    H = HLS[0];
    L = HLS[1];
    S = HLS[2]
    thresH = thresHLS[0];
    thresL = thresHLS[1];
    thresS = thresHLS[2]
    if H or L or S:
        img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        if H:
            h = img_hls[:, :, 0]
            sbin[(h >= thresH[0]) & (h <= thresH[1])] = 1
            if debug: displayImage(sbin, gray=True, title="Hue thresholding")
        if L:
            l = img_hls[:, :, 1]
            sbin[(l >= thresL[0]) & (l <= thresL[1])] = 1
            if debug: displayImage(sbin, gray=True, title="Luminosity thresholding")
        if S:
            s = img_hls[:, :, 2]
            sbin[(s >= thresS[0]) & (s <= thresS[1])] = 1
            if debug: displayImage(sbin, gray=True, title="Saturation thresholding")
    Y = YUV[0];
    U = YUV[1];
    V = YUV[2]
    thresY = thresYUV[0];
    thresU = thresYUV[1];
    thresV = thresYUV[2]
    if Y or U or V:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        if Y:
            y = img_yuv[:, :, 0]
            sbin[(y >= thresY[0]) & (y <= thresY[1])] = 1
            if debug: displayImage(sbin, gray=True, title="Y thresholding")
        if U:
            u = img_yuv[:, :, 1]
            sbin[(u >= thresU[0]) & (u <= thresU[1])] = 1
            if debug: displayImage(sbin, gray=True, title="U thresholding")
        if V:
            v = img_yuv[:, :, 2]
            sbin[(v >= thresV[0]) & (v <= thresV[1])] = 1
            if debug: displayImage(sbin, gray=True, title="V thresholding")
    LC = LAB[0];
    A = LAB[1];
    B = LAB[2]
    thresLC = thresLAB[0];
    thresA = thresLAB[1];
    thresB = thresLAB[2]
    if LC or A or B:
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        if LC:
            lc = img_lab[:, :, 0]
            sbin[(lc >= thresLC[0]) & (lc <= thresLC[1])] = 1
            if debug: displayImage(sbin, gray=True, title="LumColor thresholding")
        if A:
            a = img_lab[:, :, 1]
            sbin[(a >= thresA[0]) & (a <= thresA[1])] = 1
            if debug: displayImage(sbin, gray=True, title="A thresholding")
        if B:
            b = img_lab[:, :, 2]
            sbin[(b >= thresB[0]) & (b <= thresB[1])] = 1
            if debug: displayImage(sbin, gray=True, title="B thresholding")

    return sbin


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region_img of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# pipeline
def pipeline(img, thresholdParams=ThresholdParams(), mask=None, roi=None):
    global image_number, search, left_fit, right_fit, left_curverad, right_curverad, offset
    image_number += 1

    # region 1- Distortion correction
    img_und = undistort_image(img, mtx, dist)
    # region visualization
    if debug:
        displayImage(img_und, title="undistorted image")
    # endregion
    # endregion

    # region 2- Thresholding
    img_thres = thresholding_img(img_und, sobel=thresholdParams.sobel, thresSobel=thresholdParams.thresSobel,
                                 HLS=thresholdParams.HLS, thresHLS=thresholdParams.thresHLS,
                                 YUV=thresholdParams.YUV, thresYUV=thresholdParams.thresYUV,
                                 LAB=thresholdParams.LAB, thresLAB=thresholdParams.thresLAB)
    # region visualization
    if debug:
        plt.imshow(img_thres, cmap='gray')
        plt.title("thresholded")
        plt.show()
    # endregion
    # endregion

    # region 3- Perspective transform
    M, invM, warped = perspective_transform(img_thres)
    if debug:
        displayImage(warped, title="warped", gray=True)
    # endregion

    #region 4- lane finding
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # width of the windows +/- margin
    margin = 70
    # margin around the line from previous frame to consider
    marginLine = 25
    if search:
        # take histogram of the last part of the image to know where to start fitting the lane lines
        histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
        # region visualization
        if debug:
            plt.plot(histogram)
            plt.show()
        # endregion

        midpoint = np.int(histogram.shape[0] / 2)
        # base point of the left lane line
        leftx_base = np.argmax(histogram[:midpoint])
        # base point of the right lane line
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # region visualization
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((warped, warped, warped)) * 255
        # endregion

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(warped.shape[0] / nwindows)
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window + 1) * window_height
            win_y_high = warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # region visualization
            if debug:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # endregion

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        search = False

    else:
        # we have already one polynomial for each lane line, just look around them
        left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - marginLine)) & (
                nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + marginLine)))
        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - marginLine)) & (
                nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + marginLine)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if len(lefty) > 0 and len(leftx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        search = True
    if len(righty) > 0 and len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        search = True

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # region visualization
    if debug:
        # paint nonzero pixels inside the windows: red for left lane, blue for right one
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        # add the lines of the polinomials in yellow
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        # set the limits for the axes in the plot
        plt.xlim(0, out_img.shape[1])
        plt.ylim(out_img.shape[0], 0)
        plt.show()
    # endregion

    #endregion

    #region 5- compute curvature and distance from center
    if len(leftx) > 0 and len(lefty) > 0 and len(rightx) > 0 and len(righty) > 0:
        # calculate curvature at the bottom of the screen
        y_eval = np.max(ploty)
        # scale factor to get the curvature in meters
        ym_per_pix = 10.0 / 738  # meters from top to bottom of the image
        xm_per_pix = 3.7 / 760  # 4 meters between the lanes, covered by 760 pixels approx
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature in meters
        y_eval_cr = y_eval * ym_per_pix
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval_cr + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval_cr + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # calculate the distance between the lines
        leftx = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        rightx = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
        distance = rightx - leftx
        posx = leftx + distance // 2
        offset = img.shape[1] // 2 - posx
    #endregion

    # 6- graphic
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, invM, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img_und, 1, newwarp, 0.3, 0)
    # draw the curvature radius
    cv2.putText(result, "curvature: %.1fm" % ((left_curverad + right_curverad) / 2.), (10, 50), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, "dist.from center: %.1fm" % (offset * xm_per_pix), (10, 90), font, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    return result


def perspective_transform(image):
    # left bottom
    lb = (round(0.22 * image.shape[1]), round(0.93 * image.shape[0]))
    # print("lb: (%d, %d)" % (lb[0], lb[1]))
    # right bottom
    rb = (round(0.88 * image.shape[1]), round(0.93 * image.shape[0]))
    # print("rb: (%d, %d)" % (rb[0], rb[1]))
    # left center
    lc = (round(0.46 * image.shape[1]), round(0.63 * image.shape[0]))
    # print("lc: (%d, %d)" % (lc[0], lc[1]))
    # right center
    rc = (round(0.56 * image.shape[1]), round(0.63 * image.shape[0]))
    # print("rc: (%d, %d)" % (rc[0], rc[1]))

    vertices = np.array([lb, lc, rc, rb], dtype=np.float32)
    if debug:
        img2 = img.copy()
        cv2.polylines(img2, np.int32([vertices]), True, (255, 0, 0), thickness=4)
        displayImage(img2, gray=True)

    vertices_transf = np.array([[lb[0], image.shape[0] - 10], [lb[0], 10], [rb[0], 10], [rb[0], image.shape[0] - 10]], dtype=np.float32)
    print(vertices_transf)
    M = cv2.getPerspectiveTransform(vertices, vertices_transf)
    invM = cv2.getPerspectiveTransform(vertices_transf, vertices)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return M, invM, warped


base_dir = "D:/archi/ernesto/cursos/self-driving car/proj4/CarND-Advanced-Lane-Lines"
calibration = False
debug = True
image = True

if calibration:
    calibrate_camera(base_dir + "camera_cal", 9, 6)

print("load camera calibration data")
with open(base_dir + "/camera_cal/calibration.p", mode='rb') as f:
    calibration_data = pickle.load(f)
mtx, dist = calibration_data["mtx"], calibration_data["dist"]

# region generate image undistorted for writeup
# img = loadRGBImage(base_dir + "/camera_cal/calibration1.jpg")
# fig = plt.figure()
# a = fig.add_subplot(1, 2, 1)
# plt.imshow(img)
# plt.title("Original image")
# fig.add_subplot(1, 2, 2)
# plt.imshow(undistort_image(img, mtx, dist))
# plt.title("Undistorted image")
# plt.show()
# exit(0)
# endregion

image_number = 0
search = True
font = cv2.QT_FONT_NORMAL
left_fit = []
right_fit = []
left_curverad = 0
right_curverad = 0
distance = 0

params = ThresholdParams()
# params.sobel = (True, False, 3)
# params.thresSobel = ((50, 150), (0, 255))
params.HLS = (False, True, True)
params.thresHLS = ((0, 255), (220, 240), (170, 220))
# params.LAB = (False, False, True)
# params.thresLAB = ((0,255),(0,255),(150,250))
params.YUV = (False, False, True)
params.thresYUV = ((0, 255), (0, 255), (150, 250))

if image:
    #### individual image
    # img = loadRGBImage("d:/temp/vlcsnap-2017-05-05-18h48m43s520.jpg")
    img = loadRGBImage("d:/temp/vlcsnap-2017-05-05-18h48m47s856.jpg")
    # img = loadRGBImage("d:/temp/vlcsnap-2017-05-05-18h48m30s335.jpg")
    # img = loadRGBImage("d:/temp/vlcsnap-2017-05-05-18h48m27s201.jpg")
    # img = loadRGBImage("d:/temp/vlcsnap-2017-05-05-18h48m20s988.jpg")
    # img = loadRGBImage("d:/temp/vlcsnap-2017-05-05-18h47m53s850.jpg")
    fig = plt.figure()
    a = fig.add_subplot(2, 1, 1)
    plt.imshow(img)
    plt.title("Original image")
    fig.add_subplot(2, 1, 2)
    plt.imshow(pipeline(img, params))
    plt.title("Processed image")
    plt.show()

else:
    #### video
    def pipeline2(img):
        return pipeline(img, params)


    video = VideoFileClip("project_video.mp4")
    video_clip = video.fl_image(pipeline2)
    video_clip.write_videofile("./output_images/project_video.mp4", audio=False)
