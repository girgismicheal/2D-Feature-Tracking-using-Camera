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
In file matching2D_Student.cpp

```cpp
// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
  	else if (descriptorType.compare("BRIEF") == 0)
    {
        int bytes = 32;
        bool bOrientation = false;

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, bOrientation);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        int nFeatures = 500;
        float scaleFactor = 1.2f;
        int nLevels = 8;
        int edgeThreshold = 31;
        int firstLevel = 0;
        int WTA_K = 2;
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31;
        int fastThreshold = 20;

        extractor = cv::ORB::create(nFeatures, scaleFactor, nLevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        bool orientationNormalized = true;
        bool scaleNormalized = true;
        float patternScale = 22.0f;
        int nOctaves = 4;
        const std::vector<int> selectedPairs = std::vector<int>();

        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves, selectedPairs);
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        cv::AKAZE::DescriptorType descriptorType = cv::AKAZE::DESCRIPTOR_MLDB;
        int descriptorSize = 0;
        int descriptorChannels = 3;
        float threshold = 0.001f;
        int nOctaves = 4;
        int nOctaveLayers = 4;
        cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;

        extractor = cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels, threshold, nOctaves, nOctaveLayers, diffusivity);
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        int nFeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10.0;
        double sigma = 1.6;

        extractor = cv::xfeatures2d::SIFT::create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }
    else
    {
        
    }
    
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}
```

## Task 5: Descriptor Matching
Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function.
In file matching2D_Student.cpp
```cpp
// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
  
	double t;
    if (matcherType.compare("MAT_BF") == 0)
    {
//         int normType = cv::NORM_HAMMING;
      	int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
      cout << "MAT_BF match (" << descriptorType << ") with cross-check=" << crossCheck;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher =cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout << "MAT_FLANN match";
    }
  	else
    {
        cerr << "#4 : Wrong matcherType - " << matcherType << endl;
        exit(-1);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
		
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
vector<vector<cv::DMatch>> knnMatches;
        t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knnMatches, 2); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knnMatches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else
    {
        cerr << "\n#4 :  Wrong selectorType - " << selectorType << endl;
        exit(-1);
    }
}
```

## Task 6: Descriptor Distance Ratio
Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.
In file matching2D_Student.cpp
```cpp
else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        int k = 2;
        vector<vector<cv::DMatch>> knnMatches;
        t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knnMatches, k);// Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knnMatches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        // use descriptor distance ratio test to remove bad keypoint matches
        const double ratioThreshold = 0.8f;
        for (int i = 0; i < knnMatches.size(); i++)
        {
            if (knnMatches[i][0].distance<ratioThreshold*knnMatches[i][1].distance)
            {matches.push_back(knnMatches[i][0]);}
        }
        cout << "distance ratio test to remove: " << knnMatches.size() - matches.size() << "keypoints"<< endl;
    }
```

## Task 7: Performance Evaluation 1
Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.

## Task 8: Performance Evaluation 2
Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

## Task 9: Performance Evaluation 3
Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.