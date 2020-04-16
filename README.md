# Sky-Ground_Horizon_Detector
A Python Script that implements the technique of Sky Region Detection in multiple input_images, using an application of a computer vision-based horizon detector(OpenCV).

This script takes in an input image and returns a resized output image which separates the exclusively sky part from the rest or ground region.

#    Explanation;
 First, after loading the images, we will need to downsample and resize the input_images using a "factor" to reduce the images resolution.
# Links to resizing images:
#                          https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
#                          https://medium.com/@manivannan_data/resize-image-using-opencv-python-d2cdbbc480f0
#                          https://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
#                          https://stackoverflow.com/questions/18666014/downsample-array-in-python
#                          https://pythonexamples.org/python-opencv-cv2-resize-image/
                     

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
#                        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5948826/
#                        http://www.cs.cmu.edu/afs/cs/project/viper/www/Papers/WACV96/node5.html
#                        https://mc.ai/practical-opencv-3-image-processing-with-python/
#                        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html#hough-tranform-in-
#                        https://stackoverflow.com/questions/44449871/fine-tuning-hough-line-function-parameters-opencv
#                        https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
#                        https://github.com/jaym096/Horizon-Detection
#                        http://au.tono.my/log/20130722-horizon-detection.html
#                        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
#                        https://github.com/k29/horizon_detection


# THINGS TO NOTE:
This script runs perfectly according to the differnt availabe operating systems;
In Linux, it has several issues including file handling of image directories, but when right steps are initiated it runs properly as required.
When using linux OS, use the terminal or Anaconda3 on jupyter notebook, VisualStudio code(VScodium), or Pycharm latest version.

(NOTE: WHEN USING TERMINAL/CMD/SHELL, MAKE SURE YOU'RE ON THE CURRENT WORKING DIRECTORY(CWD/PWD)/PATH,
 IN ORDER TO EXECUTE THIS PYTHON FILE).
Example: path = r'/home/net/MORYSO/sky-ground_detection/'
 
# >>>>>>On terminal, run the script as; 
     python3 detector.py /path 
     this is designed in order to avoid the first argument in "sys.argv[1]/[0]"
     
Then make sure you state the correct path to your initial_images folder for it to give the required output

#        FILE HANDLING BOTH IN WINDOWS AND UNIX/LINUX OPERATING SYSTEMS:
  1. Windows uses different data drives like C: D: E to stored files and folders. While;
     Unix/Linux uses a tree like a hierarchical file system.
  
  2. In windows, My Documents is default home directory. While;
  	 For every user /home/username directory is created which is called his home directory.
  	 
 3. In windows, there is back slash is used for Separating the directories.
     i.e; "C:\home\net\MORYSO\sky-ground_detection\initial_images\*.*
     The C:\ part of the path is the root folder, which contains all other folders. 
     On Windows, the root folder is named C:\ and is also called the C: drive.
     
     while;
    In Lunux, there is forward slash is used for Separating the directories.
     i.e; r'/home/net/MORYSO/sky-ground_detection/initial_images/*.*'
     
 4. Links to file handling : 
#                           https://automatetheboringstuff.com/chapter8/
#                           https://stackoverflow.com/questions/2953834/windows-path-in-python
#                           https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
         
Lastly, the output of the detected/resized images will depend on; image quality and type, the runtime of your machine including the processors speed.
It is better to input few images for faster output.
Here I've set the output images to be in ".jpg" -format, you can customize the output format according to your wish as to be either; .png, .jpeg or any other format.

Then whenever one wants to run this code, first, should ensure that he puts/place the required image(s) inside the initial_images folder,
 and again ensure that the output_images directory/folder is empty... 
 then run the python script successfully, either on the terminal as elaborated above while running it on terminal...
After that, one should wait depending on your PC's processor speed, then later the output image will be shown under the output_images folder.
   
This script takes a lot of time when running multiple images say, 10 images or more. It will also depend on the image quality.
