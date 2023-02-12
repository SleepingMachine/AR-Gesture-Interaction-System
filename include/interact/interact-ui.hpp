//
// Created by sleepingmachine on 23-2-12.
//

#ifndef AR_GESTURE_INTERACTION_SYSTEM_INTERACT_UI_HPP
#define AR_GESTURE_INTERACTION_SYSTEM_INTERACT_UI_HPP
#include <iostream>
#include <opencv2/opencv.hpp>
#include <atomic>

const int kModesNum = 4;

class InteractClass{
private:
    static cv::Mat menu_;
    static int rect_width;
    static void MouseClick(int event, int x, int y, int flags, void* param) ;
public:
    static void MenuDrawing(cv::Mat & image);

    InteractClass ();
    ~InteractClass(){};
};
#endif //AR_GESTURE_INTERACTION_SYSTEM_INTERACT_UI_HPP
