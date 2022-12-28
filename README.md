# Sensor-Fusion-Udacity-Nanodegree

## 2D Feature Tracking Project

### Task 1: The Data Buffer
As adding new images for infinity doesn't a memory friendly, so we build fixed size buffer with a ring iterator.
as shown in file "MidTermProject_Camera_Student.cpp"
```cpp
// added a ring buffer with fixed size
if (dataBuffer.size()> dataBufferSize){
    // point back to the begin
    dataBuffer.erase(dataBuffer.begin());
}

dataBuffer.push_back(frame);  
```

