# 2D Feature Tracking Project

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
I have implemented all the keypoint detector in the matching2D_sturdent.cpp as mention in the lecture
Also for task 3 have implemented the keypoints removel which are located outside the rectangle bounding box.

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


I have implemented all the required keypoint descriptors.
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

I did the required FLANN and the KNN as shown.
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

## Task 7: Performance Evaluation 1:
Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.

To validate this part run the following commands on the terminal
```cpp
./2D_feature_tracking SHITOMASI BRISK MAT_BF DES_BINARY SEL_NN
./2D_feature_tracking HARRIS BRISK MAT_BF DES_BINARY SEL_NN
./2D_feature_tracking FAST BRISK MAT_BF DES_BINARY SEL_NN
./2D_feature_tracking BRISK BRISK MAT_BF DES_BINARY SEL_NN
./2D_feature_tracking ORB BRISK MAT_BF DES_BINARY SEL_NN
./2D_feature_tracking AKAZE BRISK MAT_BF DES_BINARY SEL_NN
./2D_feature_tracking SIFT BRISK MAT_BF DES_BINARY SEL_NN
```

ROI KeyPoints:

|         | SHITOMASI | HARRIS | FAST  | BRISK | ORB   | AKAZE | SIFT  |
| ------- | --------- | ------ | ----- | ----- | ----- | ----- | ----- |
| image1  | 125       | 17     | 149   | 264   | 92    | 166   | 138   |
| image2  | 118       | 14     | 152   | 282   | 102   | 157   | 132   |
| image3  | 123       | 18     | 150   | 282   | 106   | 161   | 124   |
| image4  | 120       | 21     | 155   | 277   | 113   | 155   | 137   |
| image5  | 120       | 26     | 149   | 297   | 109   | 163   | 134   |
| image6  | 113       | 43     | 149   | 279   | 125   | 164   | 140   |
| image7  | 114       | 18     | 156   | 289   | 130   | 173   | 137   |
| image8  | 123       | 31     | 150   | 272   | 129   | 175   | 148   |
| image9  | 111       | 26     | 138   | 266   | 127   | 177   | 159   |
| image10 | 112       | 34     | 143   | 254   | 128   | 179   | 137   |
| AVG     | 117.9     | 24.8   | 149.1 | 276.2 | 116.1 | 167   | 138.6 |
| Sum     | 1179      | 248    | 1491  | 2762  | 1161  | 1670  | 1386  |

Total KeyPoints:

|         | SHITOMASI | HARRIS | FAST   | BRISK | ORB   | AKAZE | SIFT   |
| ------- | --------- | ------ | ------ | ----- | ----- | ----- | ------ |
| image1  | 1370      | 115    | 1824   | 264   | 500   | 1351  | 1438   |
| image2  | 1301      | 98     | 1832   | 282   | 500   | 1327  | 1371   |
| image3  | 1361      | 113    | 1810   | 282   | 500   | 1311  | 1380   |
| image4  | 1358      | 121    | 1817   | 277   | 500   | 1351  | 1335   |
| image5  | 1333      | 160    | 1793   | 297   | 500   | 1360  | 1305   |
| image6  | 1284      | 383    | 1796   | 279   | 500   | 1347  | 1369   |
| image7  | 1322      | 85     | 1788   | 289   | 500   | 1363  | 1396   |
| image8  | 1366      | 210    | 1695   | 272   | 500   | 1331  | 1382   |
| image9  | 1389      | 171    | 1749   | 266   | 500   | 1358  | 1463   |
| image10 | 1339      | 281    | 1770   | 254   | 128   | 1331  | 1422   |
| AVG     | 1342.3    | 173.7  | 1787.4 | 276.2 | 462.8 | 1343  | 1386.1 |
| Sum     | 13423     | 1737   | 17874  | 2762  | 4628  | 13430 | 13861  |

Image Distribution: ([Link](https://drive.google.com/drive/folders/1mQV9FlBOVldrU1RrQVQcbkFjRo4c5CKl?fbclid=IwAR2S3zlRDBfmKs6qnaQg_MnVgrxIqJ03pRY-8XejznfylshBK2mfu69biZs))

## Task 8: Performance Evaluation 2
Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

To validate this part run the following commands on the terminal
```cpp
./2D_feature_tracking SHITOMASI BRISK MAT_BF DES_BINARY SEL_NN > ../results/SHITOMASI_BRISK-logs.log
./2D_feature_tracking HARRIS BRISK MAT_BF DES_BINARY SEL_NN > ../results/HARRIS_BRISK-logs.log
./2D_feature_tracking FAST BRISK MAT_BF DES_BINARY SEL_NN > ../results/FAST_BRISK-logs.log
./2D_feature_tracking BRISK BRISK MAT_BF DES_BINARY SEL_NN > ../results/BRISK_BRISK-logs.log
./2D_feature_tracking ORB BRISK MAT_BF DES_BINARY SEL_NN > ../results/ORB_BRISK-logs.log
./2D_feature_tracking AKAZE BRISK MAT_BF DES_BINARY SEL_NN > ../results/AKAZE_BRISK-logs.log
./2D_feature_tracking SIFT BRISK MAT_BF DES_BINARY SEL_NN > ../results/SIFT_BRISK-logs.log
./2D_feature_tracking SHITOMASI BRIEF MAT_BF DES_BINARY SEL_NN > ../results/SHITOMASI_BRIEF-logs.log
./2D_feature_tracking HARRIS BRIEF MAT_BF DES_BINARY SEL_NN > ../results/HARRIS_BRIEF-logs.log
./2D_feature_tracking FAST BRIEF MAT_BF DES_BINARY SEL_NN > ../results/FAST_BRIEF-logs.log
./2D_feature_tracking BRISK BRIEF MAT_BF DES_BINARY SEL_NN > ../results/BRISK_BRIEF-logs.log
./2D_feature_tracking ORB BRIEF MAT_BF DES_BINARY SEL_NN > ../results/ORB_BRIEF-logs.log
./2D_feature_tracking AKAZE BRIEF MAT_BF DES_BINARY SEL_NN > ../results/AKAZE_BRIEF-logs.log
./2D_feature_tracking SIFT BRIEF MAT_BF DES_BINARY SEL_NN > ../results/SIFT_BRIEF-logs.log
./2D_feature_tracking SHITOMASI ORB MAT_BF DES_BINARY SEL_NN > ../results/SHITOMASI_ORB-logs.log
./2D_feature_tracking HARRIS ORB MAT_BF DES_BINARY SEL_NN > ../results/HARRIS_ORB-logs.log
./2D_feature_tracking FAST ORB MAT_BF DES_BINARY SEL_NN > ../results/FAST_ORB-logs.log
./2D_feature_tracking BRISK ORB MAT_BF DES_BINARY SEL_NN > ../results/BRISK_ORB-logs.log
./2D_feature_tracking ORB ORB MAT_BF DES_BINARY SEL_NN > ../results/ORB_ORB-logs.log
./2D_feature_tracking AKAZE ORB MAT_BF DES_BINARY SEL_KNN > ../results/AKAZE_ORB-logs.log
./2D_feature_tracking SHITOMASI FREAK MAT_BF DES_BINARY SEL_KNN > ../results/SHITOMASI_FREAK-logs.log
./2D_feature_tracking HARRIS FREAK MAT_BF DES_BINARY SEL_KNN > ../results/HARRIS_FREAK-logs.log
./2D_feature_tracking FAST FREAK MAT_BF DES_BINARY SEL_KNN > ../results/FAST_FREAK-logs.log
./2D_feature_tracking BRISK FREAK MAT_BF DES_BINARY SEL_KNN > ../results/BRISK_FREAK-logs.log
./2D_feature_tracking ORB FREAK MAT_BF DES_BINARY SEL_KNN > ../results/ORB_FREAK-logs.log
./2D_feature_tracking AKAZE FREAK MAT_BF DES_BINARY SEL_KNN > ../results/AKAZE_FREAK-logs.log
./2D_feature_tracking SIFT FREAK MAT_BF DES_BINARY SEL_KNN > ../results/SIFT_FREAK-logs.log
./2D_feature_tracking AKAZE AKAZE MAT_BF DES_BINARY SEL_KNN > ../results/AKAZE_AKAZE-logs.log
./2D_feature_tracking SHITOMASI SIFT MAT_BF DES_HOG SEL_KNN > ../results/SHITOMASI_SIFT-logs.log
./2D_feature_tracking HARRIS SIFT MAT_BF DES_HOG SEL_KNN > ../results/HARRIS_SIFT-logs.log
./2D_feature_tracking FAST SIFT MAT_BF DES_HOG SEL_KNN > ../results/FAST_SIFT-logs.log
./2D_feature_tracking BRISK SIFT MAT_BF DES_HOG SEL_KNN > ../results/BRISK_SIFT-logs.log
./2D_feature_tracking ORB SIFT MAT_BF DES_HOG SEL_KNN > ../results/ORB_SIFT-logs.log
./2D_feature_tracking AKAZE SIFT MAT_BF DES_HOG SEL_KNN > ../results/AKAZE_SIFT-logs.log
./2D_feature_tracking SIFT SIFT MAT_BF DES_HOG SEL_KNN > ../results/SIFT_SIFT-logs.log
```

Average of keypionts matched counts:

| Detector vs Descriptors | BRISK | SIFT | ORB  | AKAZE | BRIEF | FREAK |
| ----------------------- | ----- | ---- | ---- | ----- | ----- | ----- |
| SHITOMASI               | 106   | 92   | 107  | none  | 106   | 77    |
| HARRIS                  | 89    | 89   | 89   | none  | 89    | 42    |
| FAST                    | 134   | 104  | 134  | none  | 134   | 88    |
| BRISK                   | 250   | 164  | 250  | none  | 250   | 152   |
| ORB                     | 95    | 73   | 104  | none  | 103   | 42    |
| AKAZE                   | 149   | 127  | 118  | 125   | 149   | 119   |
| SIFT                    | 124   | 80   | none | none  | 124   | 59    |

## Task 9: Performance Evaluation 3
Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.

keypoints detection time:

| Detector vs Descriptors | BRISK | SIFT    | ORB   | AKAZE | BRIEF | FREAK   |
| ----------------------- | ----- | ------- | ----- | ----- | ----- | ------- |
| SHITOMASI               | 20.9  | 13.9    | 17.1  | none  | 17.3  | 13.6    |
| HARRIS                  | 16.9  | 169.6   | 16.8  | none  | 17.2  | 14.0266 |
| FAST                    | 10.4  | 1.056   | 0.99  | none  | 1.034 | 1.01    |
| BRISK                   | 416.3 | 413.2   | 414.1 | none  | 415.1 | 419.9   |
| ORB                     | 10.2  | 8.18    | 9.02  | none  | 8.6   | 8.824   |
| AKAZE                   | 108.6 | 105.09  | 110.6 | 108.5 | 1094  | 107.6   |
| SIFT                    | 159.8 | 136.292 | none  | none  | 164.1 | 162     |


DESCRIPTORS detection time:

| Detector vs Descriptors | BRISK | SIFT    | ORB   | AKAZE | BRIEF | FREAK |
| ----------------------- | ----- | ------- | ----- | ----- | ----- | ----- |
| SHITOMASI               | 2.7   | 18.9    | 1.06  | none  | 1.31  | 45.7  |
| HARRIS                  | 2.19  | 21.9547 | 1.1   | none  | 1.24  | 44.8  |
| FAST                    | 22.37 | 33.6    | 1.217 | none  | 1.443 | 47.1  |
| BRISK                   | 3.3   | 68.85   | 5.49  | none  | 1.3   | 47.4  |
| ORB                     | 1.6   | 78.7    | 5.56  | none  | 7.2   | 46.2  |
| AKAZE                   | 2.3   | 33.6    | 3.7   | 92.8  | 1.5   | 45.8  |
| SIFT                    | 1.81  | 96.0904 | none  | none  | 8.17  | 46.32 |

As per the above table we noticed that the fast algorithm has very low excution times.
So, I can recommend FAST-ORB then FAST-BRIEF and lastly the FAST-BRISK as they compromise both times.
