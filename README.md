# Highway Lane Tracker  [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)



![project_video_clip](./data/documentation_imgs/project_video_clip.gif)



# The Project

The goals of this project was to:

* Calibrate cameras and apply distortion correction to raw images.
* Apply a perspective transform to get an overhead perspective of the road.
* Detect lane pixels.
* Filter out the noise and determine a stable mathematical fit to the lane.
* Determine the vehicle position in the lane and the lane curvature.

# Installation

## Dependencies

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
```

And then install the rest by running:

```
pip install symfit
pip install filterpy
```

## This Repository

Download this repository by running:

```
git clone FILL IN *****************************************************
cd FILL IN *****************************************************
```

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

