//
// Created by sleepingmachine on 23-2-12.
//

#include "fsm/fsm.hpp"
#include "interact/interact-ui.hpp"

// custom states (gerunds) and actions (infinitives)

cv::Mat image;
cv::VideoCapture capture;

std::atomic_int state_flag = 0;

enum {
    initial_status  = 'init',
    identify_status = 'identify',
    tick = 'tick'
};

struct StateControl {
    fsm::stack fsm;

    StateControl(){
        fsm.on(initial_status, 'init') = [&]( const fsm::args &args ) {//绑定init函数（初始化）
            std::cout << "init_init" << std::endl;
            cv::destroyAllWindows ();
        };
        fsm.on(initial_status, 'quit') = [&]( const fsm::args &args ) {//绑定quit函数（状态结束时清理）
            std::cout << "init_quit" << std::endl;
            exit(0);
        };
        fsm.on(initial_status, tick) = [&]( const fsm::args &args ) {//定义一个tick动作，用于输出一些信息
            InteractClass::MenuDrawing(image);

        };


        fsm.on(identify_status, 'init') = [&]( const fsm::args &args ) {//绑定init函数（初始化）
            std::cout << "identify_init" << std::endl;
            cv::destroyAllWindows ();
        };
        fsm.on(identify_status, 'quit') = [&]( const fsm::args &args ) {//绑定quit函数（状态结束时清理）
            std::cout << "identify_quit" << std::endl;
        };
        fsm.on(identify_status, tick) = [&]( const fsm::args &args ) {//定义一个tick动作，用于输出一些信息
            imshow("Mode1", image);
        };


        fsm.set( initial_status);
    }

};

int main() {
    capture.set(cv::CAP_PROP_FRAME_WIDTH,  640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.open(0);
    if(capture.isOpened())
    {
        std::cout << "Capture is opened" << std::endl;
        StateControl state_controller;
        while(true){

            capture >> image;
            if(image.empty())
                break;
            //imshow("Sample", image);
            if(cv::waitKey(5) >= 0)
                break;

            state_controller.fsm.command(tick);//刷新每一帧，输出一些信息

            switch (state_flag) {
                case -1: state_controller.fsm.pop();
                    break;
                case 1:  state_controller.fsm.push(identify_status);
                    break;
            }
            std::cout << state_flag;
        }
    }
    else
    {
        std::cout << "No capture" << std::endl;
        exit(0);
    }

    return 0;
}
