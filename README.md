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
In file 
```cpp
else if ((detectorType.compare("FAST") == 0) || (detectorType.compare("BRISK") == 0) || (detectorType.compare("ORB") == 0) || (detectorType.compare("AKAZE") == 0) || (detectorType.compare("SIFT") == 0))
{
detKeypointsModern(keypoints, imgGray, detectorType, false);
}
```
In file "matching2D_Student.cpp"
```cpp
cv::Ptr<cv::FeatureDetector> detector;

int threshold = 30;
bool NMS = true;

cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
detector = cv::FastFeatureDetector::create(threshold, bNMS, type);

```
3. BRISK

In file "matching2D_Student.cpp"
```cpp
cv::Ptr<cv::FeatureDetector> detector;

int threshold = 30;
int octaves = 3;
float patterScale = 1.0f;

detector = cv::BRISK::create(threshold, octaves, patterScale);
    
```
4. ORB

In file "matching2D_Student.cpp"
```cpp
cv::Ptr<cv::FeatureDetector> detector;

int nFeatures = 500;
int nLevels = 8;
int firstLevel = 0;
float scaleFactor = 1.2f;
int edgeThreshold = 31;
int WTA_K = 2;
int patchSize = 31;
int fastThreshold = 20;

cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
detector = cv::ORB::create(nFeatures, scaleFactor, nLevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);

```
5. AKAZE

In file "matching2D_Student.cpp"
```cpp
cv::Ptr<cv::FeatureDetector> detector;

int descriptorSize = 0;
int descriptorChannels = 3;
float threshold = 0.001f;
int nOctaves = 4;
int nOctaveLayers = 4;

cv::AKAZE::DescriptorType descriptorType = cv::AKAZE::DESCRIPTOR_MLDB;
cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;
detector = cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels, threshold, nOctaves, nOctaveLayers, diffusivity);

```
6. SIFT

In file "matching2D_Student.cpp"
```cpp
cv::Ptr<cv::FeatureDetector> detector;

int nFeatures = 0;
int nOctaveLayers = 3;
double contrastThreshold = 0.04;
double edgeThreshold = 10.0;
double sigma = 1.6;

detector = cv::xfeatures2d::SIFT::create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

```
## Task 3: Keypoint Removal
In file MidTermProject_Camera_Student.cpp

```cpp
vector<cv::KeyPoint> keypointstemp;
for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
{
    if (vehicleRect.contains(it->pt))
    {keypointstemp.push_back(*it);}
}
```
- The result after applying the ROI has less keypoints.

## Task 4: Keypoint Descriptors
Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.

## Task 5: Descriptor Matching
Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function.

## Task 6: Descriptor Distance Ratio
Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.

## Task 7: Performance Evaluation 1
Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.

## Task 8: Performance Evaluation 2
Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

## Task 9: Performance Evaluation 3
Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.