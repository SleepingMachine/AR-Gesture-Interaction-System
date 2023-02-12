//
// Created by sleepingmachine on 23-2-12.
//
#include "interact/interact-ui.hpp"

extern std::atomic_int state_flag;

InteractClass::InteractClass() {}

int InteractClass::rect_width = 0;

cv::Mat InteractClass::menu_ (480, 640, CV_8UC3);
void InteractClass::MenuDrawing(cv::Mat & image) {
    rect_width = image.size().width/kModesNum;

    rectangle(image, cv::Point2i(0,0),              cv::Point2i(rect_width, 480),   cv::Scalar(0, 255, 255),-1);    /*
    rectangle(image, cv::Point2i(rect_width+1,0),   cv::Point2i(2*rect_width, 480), cv::Scalar(0, 200, 255),-1);
    rectangle(image, cv::Point2i(2*rect_width+1,0), cv::Point2i(3*rect_width, 480), cv::Scalar(0, 150, 255),-1);
    rectangle(image, cv::Point2i(3*rect_width+1,0), cv::Point2i(4*rect_width, 480), cv::Scalar(0, 100, 255),-1);
    */
    rectangle(image, cv::Point2i(0,450), cv::Point2i(640, 480), cv::Scalar(0,0,255),-1);

    cv::setMouseCallback("MainMenu", MouseClick);

    imshow("MainMenu", image);

}

void InteractClass::MouseClick(int event, int x, int y, int flags, void *param) {
    {
        if (event == cv::EVENT_LBUTTONDOWN)//鼠标移动将会触发此事件，CV_EVENT_MOUSEMOVE和0等效
            if (y >= 450){
                state_flag = -1;
            }
            else if(x< rect_width){
                state_flag = 1;
            }

            else if(x< 2*rect_width){
                state_flag = 2;
            }
            else if(x< 3*rect_width){
                state_flag = 3;
            }
            else if(x< 4*rect_width){
                state_flag = 4;
            }
    }
}
