# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[original]: ./examples/original.jpg "Original"
[hsv]: ./examples/hsv.jpg "Hsv"
[hsl]: ./examples/hsl.jpg "Hsl"
[whiteyellow]: ./examples/white_yellow.jpg "WhiteYellow"
[gray]: ./examples/gray.jpg "Gray"
[gaussian]: ./examples/gaussian.jpg "Gaussian"
[canny]: ./examples/canny.jpg "Canny"
[mask]: ./examples/mask.jpg "Mask"
[hough]: ./examples/hough.jpg "Hough"
[average]: ./examples/average.jpg "Average"
[final]: ./examples/final.jpg "Combine"
[comparison1]: ./examples/comparison1.png "Comparison1"
[comparison2]: ./examples/comparison2.png "Comparison2"
[comparison3]: ./examples/comparison3.png "Comparison3"
[comparison4]: ./examples/comparison4.png "Comparison4"

---


# Reflection

## Table of Contents
1. [Description of Image Pipeline](#describtion)
   1. [HSL](#hsl)
   2. [Highlight Whilte or Yellow Color](#whiteyellow)
   3. [Gray](#gray)
   4. [Gaussian Blur](#gaussian)
   5. [Canny](#canny)
   6. [Region of Interest](#region)
   7. [Hough](#hough)
   8. [Draw Lines](#draw)
      1. [Separate Left and Right Lines](#separate)
      2. [Get Average Line Method 1: weighted average](#weightedaverage)
      3. [Get Average Line Method 2: linear regression](#regression)
      4. [Extrapolate](#extrapolate)
   9. [Combine](#combine)
2. [Description of Video Pipeline](#video)
3. [Shortcomings](#shortcoming)
4. [Suggestion](#suggestion)

## 1. Describe the pipeline <a name="describtion"></a>

In this project, I take advantage of the tools learned in the lesson to identify lane lines on the road and build up my own pipeline on a series of images and video streams. The pipeline is developed step by step:

### Convert original RGB image to HSL image <a name="hsl"></a>

This step is not taught in the lesson, but it is very useful to isolate whilte or yellow lines to improve Canny edge detection accuracy. Without this step, it is very difficult to detect lane lines under different light conditions for the challenging task.

Whether to convert the original RGB image to HSV image or HSL image, the following words from the [resource](http://codeitdown.com/hsl-hsb-hsv-color/) might be helpful to provide some insight:

> HSL is slightly different. Hue takes exactly the same numerical value as in HSB/HSV. However, S, which also stands for Saturation, is defined differently and requires conversion. L stands for Lightness, is not the same as Brightness/Value. Brightness is perceived as the "amount of light" which can be any color while Lightness is best understood as the amount of white. Saturation is different because in both models is scaled to fit the definition of brightness/lightness.

The comparison among RGB, HSV and HSL images below points me to the direction:

RGB Image | HSV Image | HSL Image
 :---:  | :---:  | :---:  
![alt text][original] | ![alt text][hsv] | ![alt text][hsl]

As shown in the above images, white lane lines are "blurred" in HSV image but are "highlighted" in HSL images. Hence it is straightforward to choose HSL image conversion.

```python
cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
```

### Highlight yellow and white color from HSL image <a name="whiteyellow"></a>

Since lane lines are in white or yellow color, we can isolate yellow or white lines from the HSL image and then combined with the original RGB image. 

RGB Image | HSL Image | White or Yellow Highlighted Image
 :---:  | :---:  | :---:  
![alt text][original] | ![alt text][hsl] | ![alt text][whiteyellow]

```python
# converge image from RGB to HSL    
cvt_image = convert_image(img, 'HSL')

# mask for yellow color
yellow_low_threshold = np.array([15, 38, 115], dtype=np.uint8)
yellow_high_threshold = np.array([35, 204, 255], dtype=np.uint8)
yellow_mask = cv2.inRange(cvt_image, yellow_low_threshold, yellow_high_threshold)

# mask for white collor
white_low_threshold = np.array([0, 200, 0], dtype=np.uint8)
white_high_threshold = np.array([255, 255, 255], dtype=np.uint8)
white_mask = cv2.inRange(cvt_image, white_low_threshold, white_high_threshold)

# combine mask
yellow_white_mask = cv2.bitwise_or(yellow_mask, white_mask)
cv2.bitwise_and(img, img, mask = yellow_white_mask)
```

### Convert highlighed image to gray image <a name="gray"></a>

To detect white or yellow lines out of the black road from the previously prepossed images, it is better to covert to gray images to increase the contrast.

RGB Image | Input (White Highlighted) Image | Output (Gray) Image
 :---:  | :---:  | :---:  
![alt text][original] | ![alt text][whiteyellow] | ![alt text][gray]

```python
cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
```

### Gaussian Blur <a name="gaussian"></a>

[Gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) (also referred to as Gaussian smoothing) is a pre-processing technique used to smoothen the edges of an image to suppress noise and spurious gradients. 

RGB Image | Input (Gray) Image | Output (Gaussian Blur) Image
 :---:  | :---:  | :---:  
![alt text][original] | ![alt text][gray] | ![alt text][gaussian]

```python
cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```

### Canny Edge Detection <a name="canny"></a>

Canny edge detection is then used to identify lines in the prepossed image. It detects strong edge (strong gradient) pixels above the high_threshold, and reject pixels below the low_threshold; pixels with values between the low_threshold and high_threshold will be included as long as they are connected to strong edges.

RGB Image | Input (Gaussian Blur) Image | Output (Canny) Image
 :---:  | :---:  | :---:  
![alt text][original] | ![alt text][gaussian] | ![alt text][canny]

```python
cv2.Canny(img, low_threshold, high_threshold)
```

### Region Of Interest <a name="region"></a>

Here, a region of interest (mask) is set to discard any lines outside of this region characterized by a polygon. Note that we make a strong assumption that the camera remains in a fixed position and thus the region of interest can keep the same for this task.

RGB Image | Input (Canny) Image | Output (Region of Interest) Image
 :---:  | :---:  | :---:  
![alt text][original] | ![alt text][canny] | ![alt text][mask]

```python
#defining a blank mask to start with
mask = np.zeros_like(img)   

#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
if len(img.shape) > 2:
  channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
  ignore_mask_color = (255,) * channel_count
else:
  ignore_mask_color = 255

#filling pixels inside the polygon defined by "vertices" with the fill color    
cv2.fillPoly(mask, vertices, ignore_mask_color)

#returning the image only where mask pixels are nonzero
masked_image = cv2.bitwise_and(img, mask)
```


### Hough Transform <a name="hough"></a>

Then, we apply [Hough Transform](https://en.wikipedia.org/wiki/Hough_transform) technique to extract lines from the masked edge image.

RGB Image | Input (Region of Interest) Image | Output (Hough) Image
 :---:  | :---:  | :---:  
![alt text][original] | ![alt text][mask] | ![alt text][hough]

```python
# rho: distance resolution in pixels of the Hough grid, typical value (1, 2, ...)
# theta: angular resolution in radians of the Hough grid, typical value np.pi/180 * (1, 2, ...)
# threshold: minimum number of votes (intersections in Hough grid cell)
# minLineLength: minimum number of pixels making up a line
# maxLineGap: maximum gap in pixels between connectable line segments
lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)
```

### Draw Lines <a name="draw"></a>
The Hough Transfrom technique returns us a set of lines, not an ideal left and right lines to define a lane. Hence, we need to obtain the average smooth left and right lines from the hough transform results.

#### Separating Left And Right lanes <a name="separate"></a>
First, we need to distinguish left from right lines based on slop of lines.

Obervation:
* left line: **y value decreases (i.e. height) as x value (i.e. width) increases, implying negative slope**
* right lane: **y value increases (i.e. height) as x value (i.e. width) increases, implying positive slope**

Thumb of rule: in oder to screen out some noisy lines, i.e. horizental lines or lines with small slope, we set a threshold as 0.5 in the code below.

```python
lines_left = list()
lines_right = list()

for line in lines:
  for x1, y1, x2, y2 in line:
      if x1 == x2: # vertical line
          continue
      else:
          m, b, d = line_xy2mb(x1, y1, x2, y2)
          if m < -0.5: # slope of right lines are negative, moreover, we set -0.5 to screen out some noisy lines
              lines_left.append(line)
          elif m > 0.5: # slope of right lines are positive, moreover, we set 0.5 to screen out some noisy lines
              lines_right.append(line)
```

####  Average Method 1: Weighted Average <a name="weightedaverage"></a>
Each detected line has its own slope (m) and intercept (b), corresponding to a point (m,b) in the Hough plane while mutiple lines can be viewed as multiple points in the Hough plane and we can obtain a mean point (m_0,b_0) to represent an average line. But lines detected by Hough Transform do not have the same length, we need to take length of lines into consideration when calculating the average line, as shown in the following code.

```python
def lines_weighted_average(lines):
    """
    objective:
        Get an average line based on a set of obtained lines
    Inputs:
        lines: a list of distinguished lines
    Outputs:
        line_mb: an average line parameterized by slope--m and intercept--b
    """
    
    if lines is None:
        return None
    
    list_lines_mb = list()
    list_lines_d = list()  
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            m, b, d = line_xy2mb(x1, y1, x2, y2)
            list_lines_mb.append((m,b))
            list_lines_d.append(d)
                        
    if len(list_lines_d) > 0 :
        line_mb = np.dot(list_lines_d, list_lines_mb) / np.sum(list_lines_d)
        return line_mb
    else:
        return None
```

#### Average Method 2: Linear Regression <a name="regression"></a>

Each line can be represented by a starting point and an end point in x-y plane and multiple lines are formed by various points. To find a line to "go through" these point is a typical linear regression problem, which can be realized by the function

```python
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
```
Considering the length factor, we can use linear interpolation to add more points to represent one line.

In complicated curve line situations, it is almost impossbile to use one smooth straight line to represent a curve lane. However, we can deliver roughly better results, if we put more weights on the detected lines that are near to the car (camera), that is to say, we can use log-space interpolation to add more points to lines with higher y values.

```python
def lines_weighted_linear_regression(lines, imshape):
    """
    objective:
        Get an average line based on a set of obtained lines via linear regression
    Inputs:
        lines: a list of distinguished lines
        imshape: image shape
    Outputs:
        line_mb: an average line parameterized by slope--m and intercept--b
    """
    
    if lines is None:
        return None
    
    list_lines_x = list()
    list_lines_y = list()  
    
    y_linspace = np.linspace(0, imshape[0], 1000)
    y_logspace = imshape[0] - np.logspace(np.log10(0.1), np.log10(0.4*imshape[0]), 1000)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            m, b, d = line_xy2mb(x1, y1, x2, y2)
            for yp in y_logspace:
                if ((y1 >= yp) and (yp >= y2)) or ((y1 <= yp) and (yp <= y2)):
                    xp = (yp - b) * 1.0 /m
                    list_lines_x.append(xp)
                    list_lines_y.append(yp)
                        
    if len(list_lines_x) > 0 :
        slope, intercept, r_value, p_value, std_err = stats.linregress(list_lines_x, list_lines_y)
        line_mb = (slope, intercept)
        return line_mb
    else:
        return None
```

####  Extrapolate <a name="extrapolate"></a>

After getting slope and intercept of the average line, we can extrapolate the line from y1 to y2.

RGB Image | Input (Hough) Image | Output (Draw Line) Image
 :---:  | :---:  | :---:  
![alt text][original] | ![alt text][hough] | ![alt text][average]

```python
def line_extrapolate(line_mb, y1, y2):
    """
    objective:
        Generate a line based on slope and intercept
    Inputs:
        line_mb: a line parameterized by slope and intercept
        y1: y value of the first point
        y2: y value of the second point
    Outputs:
        line_xy: a line parameterized by x1, y1 and x2, y2
    """
    if line_mb is None:
        return None
    
    m, b = line_mb
    if m == 0:
        return None
    else:
        x1 = int((y1 - b) * 1.0 / m)
        x2 = int((y2 - b) * 1.0 / m)
        y1 = int(y1)
        y2 = int(y2)
        line_xy = [[[x1, y1, x2, y2]]]
        return line_xy
```

### Combine <a name="combine"></a>
Lastly, we combine the detected lines with the original image by calling the function as shown in the following code.

To compare effects of the above two average method, I use blue line to represent the weighted average method and use red line to demonstrate the linear regression method.

RGB Image | Input (Draw Line)) Image | Output (Combine) Image
 :---:  | :---:  | :---:  
![alt text][original] | ![alt text][average] | ![alt text][final]

```python
cv2.addWeighted(initial_img, α, img, β, γ)
```

## 2. Describe the improved pipeline for videos <a name="video"></a>
In the above pipeline, we deal with each image as a single object. A video can be seens as combination of frames, but should not be treated as isolated frames. If we ignore information contained in adjacent frames and directly apply the image pipeline to the video without any revision, we might encounter some problems:
1. detected lines are not smooth from one frame to the next frame, i.e., slopes or intercept of lines might change too much among adjacent frames. (this should not happen in real cases)
2. for some frames, lane lines of the current frame might not be clear enough for hough detection, leading to wrong lane detection

Fortunately, these can be solved by combining information contained in the current frame with that in the past few freams. We can filter (use moving average filter, one kind of FIR filter; or Gaussian weighted filter) the detected coefficients, then we can get smoother result. Moreover, if the current coefficients differ from the past results so much, we can view the current result not correct and then replace the current coefficients with the past average ones. In this way, we can improve the effect of detected lines on videos, especially on the last chanllenging video. (Here, I need to appreciate the advice and tips from Eddie Forson)

Comparison1 | Comparison2 | Comparison3
 :---:  | :---:  | :---:  
![alt text][comparison1] | ![alt text][comparison2] | ![alt text][comparison3]

## 3. Identify potential shortcomings with your current pipeline <a name="shortcoming"></a>
There are some shorcoming for the current pipeline:
* For video case, I find that detected lines are not so smooth from one frame to another one.
* There are many parameters need to be tuned carefully. It is hard to find one set of parameters that are good and robust for different situations, especially for the challenging video. In reality, complicated situations, such as rainy day, dark night, rush hour, would bring more challenges to parameter tuning.
```python
dict_parameters = dict()

dict_parameters['gaussian_kernel_size'] = 7

dict_parameters['canny_threshold'] = (150, 250)

dict_parameters['region_bottom_x_ratio'] = (0.0, 0.5, 1.0)
dict_parameters['region_top_x_ratio'] = (0.46, 0.5, 0.54)
dict_parameters['region_bottom_y_ratio'] = 1.0
dict_parameters['region_top_y_ratio'] = 0.6

dict_parameters['hough_rho'] = 2
dict_parameters['hough_theta'] = np.pi/180
dict_parameters['hough_threshold'] = 15
dict_parameters['hough_min_line_len'] = 5
dict_parameters['hough_max_line_gap'] = 60
```
* Straight line detection cannot handle curve lanes well.


## 4. Suggest possible improvements to your pipeline <a name="suggestion"></a>
For video case, my further step should take the past frames into consideration, for example, adopt an moving average filter to deliver smoother results.

Moreover, I can take full advantage of deep learning tools to improve accuracy of lane detection.

The code is available on [Github](https://github.com/fuqiang07/SDCND/edit/master/CarND-LaneLines-P1-master).

References and Acknowledgements:
During this project, I get some great ideas, HSL conversion and linear regression, from the following websites or repositories:
* [Eddie Forson's github repository: CarND-LaneLines-P1](https://github.com/kenshiro-o/CarND-LaneLines-P1)
* [Kirill's github repository: p1_lane_lines_detection](https://github.com/Kidra521/carnd/tree/master/p1_lane_lines_detection)
