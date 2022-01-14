# Sift-Implementation

you can run the main.py file to find some test results
This repo was developed as a recreation of methods layed by a paper by David G. Lowe you can find [here]{https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf}.


1. involves taking sparse sift keypioints over the image 
2. Use a brute force feature matcher to find good matches by using knn taking M and comparing the distance of it too the second good match
            - This is done by comapring the distance of a first match to a second match by a distance of .75 
3.  this step involves taking a cluster hough transform from the identified.
4.  from the following data the algorithim writes abounding box of various size and orientation



[alt text] (https://github.com/torn8to/sift-based-OD/blob/main/Capture.PNG)
