# Canny-Edge-Detection
Detection edge using canny edge algorithm

In 1986, John Canny defined a set of goals for an edge detector and described an optimal method for achieving
them. Canny specified three issues that an edge detector must address:
• Error rate: Desired edge detection filter should find all the edges, there should not be any missing edges,
and it should respond only to edge regions.
• Localization: Distance between detected edges and actual edges should be as small as possible.
• Response: The edge detector should not identify multiple edge pixels where only a single edge exists.
Remember from the lecture that in Canny edge detection, we will first smooth the images, then compute gradients,
magnitude, and orientation of the gradient. This procedure is followed by non-max suppression, and finally hysteresis
thresholding is applied to finalize the steps. Briefly, follow the steps below for practical implementation of Canny
Edge detector :
1. Read a gray scale image you can find from Berkeley Segmentation Dataset, Training images, store it as a
matrix named I.
2. Create a one-dimensional Gaussian mask G to convolve with I. The standard deviation(s) of this Gaussian is
a parameter to the edge detector (call it σ > 0).
3. Create a one-dimensional mask for the first derivative of the Gaussian in the x and y directions; call these G x
and G y . The same σ > 0 value is used as in step 2.
4. Convolve the image I with G along the rows to give the x component image (I x ), and down the columns to
give the y component image (I y ).
5. Convolve I x with G x to give I x 0 , the x component of I convolved with the derivative of the Gaussian, and
convolve I y with G y to give I y 0 , y component of I convolved with the derivative of the Gaussian.
6. Compute the magnitude of the edge response by combining
q the x and y components. The magnitude of the
result can be computed at each pixel (x, y) as: M(x, y) = I x 0 (x, y) 2 + I y 0 (x, y) 2 .
7. Implement non-maximum suppression algorithm that we discussed in the lecture. Pixels that are not local
maxima should be removed with this method. In other words, not all the pixels indicating strong magnitude
are edges in fact. We need to remove false-positive edge locations from the image.
8. Apply Hysteresis thresholding to obtain final edge-map.

Definition: Non-maximal suppression means that the center pixel, the one under consideration, must have
a larger gradient magnitude than its neighbors in the gradient direction. That is: from the center pixel,
travel in the direction of the gradient until another pixel is encountered; this is the first neighbor. Now,
again starting at the center pixel, travel in the direction opposite to that of the gradient until another
pixel is encountered; this is the second neighbor. Moving from one of these to the other passes though
the edge pixel in a direction that crosses the edge, so the gradient magnitude should be largest at the edge pixel.
Algorithmically, for each pixel p (at location x and y), you need to test whether a value M(p) is maximal in
the direction θ(p). For instance, if θ(p) = pi/2, i.e., the gradient direction at p = (x, y) is downward, then
M(x, y) is compared against M(x, y − 1) and M(x, y + 1), the values above and below of p. If M(p) is not
larger than the values at both of those adjacent pixels, then M(p) becomes 0. For estimation of the gradient
orientation, θ(p), you can simply use atan2(I y 0 , I x 0 ).
