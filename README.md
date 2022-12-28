# Sensor-Fusion-Udacity-Nanodegree

## 2D Feature Tracking Project

### Task 1: The Data Buffer
As adding new images for infinity doesn't a memory friendly, so we build fixed size buffer with a ring iterator.
In file "MidTermProject_Camera_Student.cpp"
```cpp
// added a ring buffer with fixed size
if (dataBuffer.size()> dataBufferSize){
    // point back to the begin
    dataBuffer.erase(dataBuffer.begin());
}

dataBuffer.push_back(frame);  
```

### Task 2 and 3: Keypoint Detection

1. HARRIS Implementation

In file "matching2D.hpp"
```cpp
// implement the harris detector algorithm
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false, bool NMS=false);
```

In file "MidTermProject_Camera_Student.cpp"
```cpp
// call the harris detector allgorithm
else if (detectorType.compare("HARRIS")==0){
     detKeypointsHarris(keypoints, imgGray, false, false);
}
```

In file "matching2D_Student.cpp"
```cpp
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, bool NMS)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    int apertureSize = 3;
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    // double minDistance = (1.0 - maxOverlap) * blockSize;
    //int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints
    // double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;
    int response_min = 100;

    // Apply corner detection
    double t = (double)cv::getTickCount();

    // Detect Harris corners and normalize output
    cv::Mat harmat;
    harmat = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, harmat, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(harmat, harmat, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());


    for (int j = 0; j < harmat.rows; j++)
    {
        for (int i = 0; i < harmat.cols; i++)
        {
            int response = (int)harmat.at<float>(j, i);
            if (response > response_min)
            {
                cv::KeyPoint newKeypoint;
                newKeypoint.pt = cv::Point2f(i, j);
                newKeypoint.size = 2 * apertureSize;
                newKeypoint.response = response;
                // if non-maximum suppression (NMS) is required to perform in the neighbors
                if (NMS)
                {

                    bool overlapped = false;
                    for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                    {
                        double overlap = cv::KeyPoint::overlap(newKeypoint, *it);
                        if (overlap > maxOverlap)
                        {
                            overlapped = true;
                            if (newKeypoint.response > (*it).response)
                            {
                                *it = newKeypoint; // replace the keypoint with a higher response one
                                break;
                            }
                        }
                    }
                    // add the new keypoint which isn't consider to have overlap with the keypoints already stored in the list
                    if (!overlapped){keypoints.push_back(newKeypoint);}
                }
                else
                { keypoints.push_back(newKeypoint);}
            }
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
```

2. FAST
3. BRISK
4. ORB
5. AKAZE 
6. SIFT