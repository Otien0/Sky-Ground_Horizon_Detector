# Sky-Ground_Horizon_Detector
A Python implementation of Sky Region Detection in multiple images, using an application of a computer vision-based horizon detector(OpenCV).

This script takes in an input image and returns a resized output image which separates the exclusively sky part from the rest or ground region.

#    Explanation;
    
 Horizon detection or sky segmentation involves the approach of finding a boundary between sky and non sky regions, particularly the ground region in a given image.
 This can have many applications especially in navigation of UAV. 
 
 In most of the early attempts of this problem, there is an underlying assumption that the horizon boundary is linear.

 The basic algorithm in that the sky and ground are modeled as two different gaussian distributions in RGB space,
 and then horizon line is a line segment separating the two, which can found by maximizing an optimization criterion. 
  Thus sky and ground regions are represented as two set of points each distributed about a separate mean point in the RGB space.
  We then perform a search through potential set of lines (m,b), to find the line with highest likelihood of being the best fit horizon line.
  Now we just need to find the scalar term for the optimization criterion. 
  
 Intuitively, given the pixel groupings, we need to quantify the assumption that a sky pixel will look similar to other sky pixels and
   likewise for the ground ones. Thus we are definitely looking for a degree of variance in each distribution. 
 
# Links to further notes:
   
