# Highway Lane Tracker  [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[![Peter Moran's Blog badge](https://img.shields.io/badge/Peter%20Moran's%20Blog-Robust%20Lane%20Tracking-blue.svg?style=social)](http://petermoran.org/robust-lane-tracking/)



![project_video_clip](./data/documentation_imgs/project_video_clip.gif)

![challenge_video_clip](./data/documentation_imgs/challenge_video_clip.gif)



# The Project

The goals of this project were to:

* Calibrate cameras and apply distortion correction to raw images.
* Apply a perspective transform to get an overhead perspective of the road.
* Detect lane pixels.
* Filter out the noise and determine a stable mathematical fit to the lane.
* Determine the vehicle position in the lane and the lane curvature.

# How it Works

In simplest form, the major steps in the lane finding pipeline are:

1. **Get the current frame** from the dashboard camera's video.

2. **Warp the perspective** of the dashboard image to an overhead view of the road.

3. **Select for potential lane pixels** by locally normalizing the image, thresholding the color channels in various color spaces, and combining all the selected pixels into a single "score" image. Specifically, the LAB B, HSV value, and HLS lightness color channels were used.
   ![Single channel thresholding example](https://i1.wp.com/petermoran.org/wp-content/uploads/2017/09/hsv_v_thresh.png?w=600)

4. **Perform a sliding window search** over the selected lane pixels to find the center of each lane line at different points along the y-axis. Each window has it's own independent Kalman filter that uses a decaying velocity model (i.e. we expect the velocity of the windows to decrease over time).

   For each window, we:

   * Scan the window across the image (keeping its y-position fixed) and find the x-position where that window covers the most pixels (according to a gaussian kernel).

   * Using the Kalman filter and the signal-to-noise ratio, determine if the measurement is unreliable or outlier. If it is, do not update the filtered window position.

     * If the last window position update was too long ago, drop the window entirely until a new, reliable measurement is made and we run an update.

   * Search with the next window, but constrain it's search to a region centered on this window.

     ![Window selection example](https://i2.wp.com/petermoran.org/wp-content/uploads/2017/09/raw_vs_filtered.gif?w=1100)

5. **Apply a polynomial fit** along the filtered windows that have *not* been dropped for each lane line.

6. **Calculate the position** of the car in the lane using the x-offsets, **find the radius of curvature** from the polynomial coefficients, and **draw the lane line onto the image**, making sure not to extrapolate past the furthest point on the shorter line. With this, we have all our desired outputs and are done.

For a more in depth description of the process and the code (including plenty of pictures and videos) **check out my full [writeup at my blog](http://petermoran.org/robust-lane-tracking/)!**

---

# Installation

## This Repository

Download this repository by running:

```
git clone https://github.com/peter-moran/highway-lane-tracker.git
cd highway-lane-tracker
```

## Software Dependencies

This project utilizes the following, easy to obtain software:

* Python 3
* OpenCV 2
* Matplotlib
* Numpy
* Moviepy
* [Symfit](http://symfit.readthedocs.io/en/latest/)
* [Filterpy](https://filterpy.readthedocs.io/en/latest/)

An easy way to obtain these is with the [Udacity CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) and Anaconda. To do so, run the following (or see the [full instructions](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md)):

```
git clone https://github.com/udacity/CarND-Term1-Starter-Kit.git
cd CarND-Term1-Starter-Kit
conda env create -f environment.yml
activate carnd-term1
```

And then install the rest (while still in that environment) by running:

```
pip install symfit
pip install filterpy
```

---

# Usage

## Standard usage

To produce standard output, with the lane lines and detected windows all in one video (as shown in the clips above), simply run:

```
python find_lane.py ./path/to/video_file.mp4
```

This will produce a new file `video_file_presentation.mp4` in the `./output` folder.

For example, to run the standard test video, run:

```
python find_lane.py ./data/test_videos/project_video.mp4
```

You can find the output at `./output/project_video_presentation.mp4`.

More input video files can be found in `./data/test_videos`. However, new videos from a new camera would require calibration.

## Advanced Usage

### Pipeline visualizations

To view any part of the image pipeline, such as the binary images alone or just the filtered windows, you can pass a 2nd parameter.

```
python find_lane.py ./path/to/video_file.mp4 <pipeline_element>
```

Files will be saved according to the pipeline element you selected, such as `./output/video_<pipeline_element>.mp4`.

Valid pipeline elements include:

**Lane detections:**

- `presentation`, a combination of views as used for demonstration above (**default mode**).
- `windows_raw`
- `windows_filtered`
- `highlighted_lane`

**Image perspectives:**

* `dash_undistorted`
* `overhead`

**Lane color space and binary thresholds:**

*  `lab_b`, `lab_b_binary`
*  `lightness`, `lightness_binary`
*  `value`, `value_binary`


* `pixel_scores`

### Custom save location

The third parameter allows you to specify a location to save the file. Make sure to specify the directory as well as the file name with extension.

```
python find_lane.py <video_file> <pipeline_element> <save_file>
```

