//
// Created by zer0like on 2018/7/12.
//
#include "utils.h"

cv::Mat getPatch(const cv::Mat & image,const cv::Point& p,int RADIUS){
    assert(RADIUS <= p.x && p.x <image.cols-RADIUS && RADIUS <= p.y && p.y < image.rows-RADIUS);

    return image(cv::Range(p.y-RADIUS,p.y+RADIUS+1),cv::Range(p.x-RADIUS,p.x+RADIUS+1));

}

std::vector<cv::Point> FindPoint(cv::Point &p,int width, int height,int range,int RADIUS){
    //give a point to calculate the around point of the Radius
    std::vector<cv::Point> pointList ;

    assert(range%2 !=0 );

    int startX =  p.x - ((range-1)/2)*RADIUS;
    int stratY = p.y - ((range-1)/2)*RADIUS;

    int startX_end = startX + RADIUS*(range-1);
    int startY_end = stratY + RADIUS*(range-1);


    for(;startX<=startX_end;startX+=RADIUS){
        for(stratY = p.y - ((range-1)/2)*RADIUS;stratY<=startY_end;stratY+=RADIUS){
            if(startX > width -RADIUS || stratY > height -RADIUS) continue;
            if(startX == p.x && stratY ==p.y) continue;
            if(startX < RADIUS || stratY <RADIUS)continue;
            pointList.push_back(cv::Point(startX,stratY));
        }
    }

    return pointList;
}

