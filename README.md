# **Advanced Line Finding** 

## Writeup

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort.png "Undistorted"
[image2]: ./output_images/corners_found4.jpg "Detected corners"
[image3]: ./output_images/original_undistorted.png "Undistorted image"
[image4]: ./output_images/sobelx.png "Sobel X thresholding"
[image5]: ./output_images/luminosity.png "Luminosity (L) thresholding"
[image6]: ./output_images/saturation.png "Saturation (S) thresholding"
[image7]: ./output_images/V.png "V channel (from YUV) thresholding"
[image8]: ./output_images/threshold_combined.png "combined thresholding"
[image9]: ./output_images/area_for_perspective.png "Area for perspective change"
[image9b]: ./output_images/warped.png "Perspective changed"
[image10]: ./output_images/straighten.png "Thresholded and straighten"
[image11]: ./output_images/histogram.png "Threshold histogram"
[image12]: ./output_images/sliding_windows.png "Sliding windows"
[image13]: ./output_images/polyfit.png "Polynomial fitting"
[image14]: ./output_images/curv_ratio.png "Curvature ratio"
[image15]: ./output_images/processed_complete.png "Processed image"
[imgOriginal]: ./output_images/vlcsnap-2017-05-05-18h48m47s856.jpg "Original image"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

This is it :)

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function `calibrate_camera()`.

I start by preparing a list of "object points", which will be the coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time the code successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

The function includes code to save the images showing the corners, like this:

![alt text][image2]

After running this corner detection process on all calibration images, I used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. These coefficients are then saved to a pickle file. Later on, we will load this calibration data and just apply them to remove the distortion from images.

I applied this distortion correction to a test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

This code is encapsulated in the function `undistort_image()`.


---

## Pipeline

The process applied to each image is divided here in several steps, executed one after the other conforming a pipeline. To process an entire video, we apply the pipeline to each frame.
To demonstrate the pipeline, I will apply and document each step on the following image:

![alt text][imgOriginal]


#### 1. Distortion correction

The distortion can be corrected once we have calibrated the camera and obtained the distortion parameters. For this project, the distortion parameters are computed previously (in the code, set `calibration = True` to run the calibration code and generate the parameters) and then loaded before starting the pipeline.
We call the `undistort_image` function with the original image and get an undistorted image. This is the result on our test image:

![alt text][image3]


#### 2. Thresholding

Thresholding is a technique used to make the lines 
I used a combination of color and gradient thresholds to generate a binary image with the lane lines resalted. The code is in function `thresholding_img` which takes several parameters to apply different transformacions:

`img`: the image to transform
`sobel`: a (boolean, boolean, int) tuple. First boolean indicates whether to apply Sobel filter on X direction, second one is for Sobel in Y direction, and the integer is the size of the kernel used in the transformation in pixels (3 by default).
If any Sobel transformations is applied, the input image is converted to grayscale internally.
`HLS`: a tuple of three booleans, to select the channel to apply to an HLS version of the input image.
`YUV`: a tuple of three booleans, to select the channel to apply from a YUV version of the input image.
`LAB`: another tuple of three booleas to apply channels of a LAB conversion of the input image.
After applying each selected transform, the resulting pixels are compared with a low and high thresholds to decide if the pixel will be 'activated' in the resulting binary image. All thresholds are passed to the function as tuples of pairs (minimum threshold, maximum threshold). For example the argument `thresYUV` is interpreted as `((Ymin, Ymax), (Umin, Umax), (Vmin, Vmax))`.
The resulting binary image starts as an array of 0 in the same shape of the input image. When a filter is applied and the resulting pixel is between the corresponding thresholds, it is set to 1 in the binary image. This is done for all activated filters over the same image, so it is an aggregation of different filters.
Following is our test image after threshold with this parameters:

```python
    sobel = (True, False, 3)
    HLS = (False, True, True)
    YUV = (False, False, True)
    LAB = (False, False, False)
    thresSobel = ((40, 150), (70, 150))
    thresHLS = ((0, 255), (220, 240), (170, 220))
    thresYUV = ((0, 255), (0, 255), (150, 250))
    thresLAB = ((100, 250), (100, 200), (130, 240))
```

So the thresholding is a combination of Sobel X with a kernel of 3 pixels, 

![alt text][image4]

channels L and S of the HLS color space, 

![alt text][image5]![alt text][image6]

and V channel of the YUV transformation

![alt text][image7]

The thresholds applied were selected through an 'informed process' of trial and error:

![alt text][image8]


#### 3. Perspective transform

Now we have to take the thresholded image and apply a perspective transform that will convert the image as if we were looking at it from above ('birdseye' perspective). This way, the lines of the lane will become parallels -or at least close to it. This way is much easier to filter outlier pixels and fit a polygon on the remaining ones.
The code to transform the perspective is in the function `perspective_transform()` which accepts an image and returns the direct and inverse transform matrices to go from an image to a birdseye perspective and viceversa.
The transformation turns an arbitrary polygon with the approximate shape of the lane

![alt text][image9]

to a straight rectangle, interpolating the rest of the image.

![alt text][image9b]

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 282, 686      | 282, 728      | 
| 589, 465      | 282, 10       |
| 717, 465      | 1126, 10      |
| 1126, 686     | 1126, 728     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image10]


#### 4. Fit polinomials to each line

Now that we have the lane lines as seen from above, we look for the actual pixels and fit a polinomial to those. The first step is to find the bottom of the lines to follow them up. We do this creating a histogram of the pixels in the bottom of the image:

![alt text][image11]

this plot shows the major concentration of '1' pixels at the bottom of the lines. We start from there and take small windows to look for the pixels that continue the lines. This process is known as 'sliding windows':

![alt text][image12]

While going up with the sliding windows, we gather all points inside their. Those points are then used to fit a 2nd order polynomial which will define the lines:

![alt text][image13]

The code is in `pipeline()` function, in region 4.

Once we found the lines in the first frame, next frames are processed differently: the lines are continuous, so we can assume that the pixels for the lines in the next frame will be close to the lines in present frame. In code, we look for the activated pixels that are in a vicinity of the lines found. 
This allows faster processing, smoother results, and some noise rejection.


#### 5. Calculate the curvature radio and distance from the center of the lane

The polynomials fitted to the points define two neat lines in _pixel space_. We want to take this to the real world, so we scale the pixels based on the meters per pixel scale factors

```python
    ym_per_pix = 10.0 / 738  # meters from top to bottom of the image
    xm_per_pix = 3.7 / 760  # 4 meters between the lanes, covered by 760 pixels approx
```

The curvature ratio is found by applying the formula 

![alt text][image14]

A and B are the coefficients of a new set of polynomials fitted to the scaled points.
The average curvature ratio is computed and drawed on the original image.

We also can calculate an approximate position of the car inside the lane by assuming the camera is seated in the center of the car, and so the lane lines should be also centered in the image. We can calculate the geometrical center of the lines, then comparing with the center of the image we obtain the offset of the car from the center of the line. This is drawed too on the original image.


#### 6. Superimpose the identified lane over the original image for visualization

Now that we have found the lane lines and calculated the curvature and the distance of the car from the center of the lane, we can draw this data over the original image. Previously we have to go back to the original point of view, so we do a new change of perspective using the inverse matrix calculated in step 3.
For better results, the inside of the polygon fitted to the lane is filled with color.


![alt text][image15]


---

### Pipeline (video)

This pipeline can be applicated to a video by splitting it in a series of images (frames) and passing each one in order to the pipeline function. We used videopy to do this processing and end up with a new video showing the result of the process.

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This process allows us to get results that look pretty nice. There are some problems to tackle to make it more general:

* The discovery of the lines depends on the thresholding process. This process is critical to end up with a binary image where the lines are clearly distinguishable from the rest. Here is where the different color modes can be of help. Also different transformations could be applied to the image previously to the thresholding: sobel uses grayscaling, but there are other processes that for example get rid of the shadows or smooth the image to eliminate single stranded pixels.
In fact, the threshold used for the project video does not produce a good result in the challenge video. This is due to the visual difference in the images: the challenge video has darker lines almost parallel to the real lane lines in some places that confuse Sobel and other approaches.
The biggest problem I see with this technique is that the threshold have to be calculated manually for each case (as far as I know). This would limit the practical application on real cases.
* I think the process could be made more robust introducing some extra calculation to compute the _expected_ position of a line given the other, for example using the norms for the lane width on the country. Then when one of the lines is lost (due to reflections, shadows, bad painting, etc) it could be _estimated_ using the other. This would make the process slower but the results could be more robust.
* Another 'smoothing' process that can be applied is the averaging of several consecutive frames to fit the lines. This would take care of places where one of both lines are lost or not clear, and avoid jittering.

Even with the problems I see, the technique is wonderful and I think it is a good starting point to find a general process to do line finding in the real world.
