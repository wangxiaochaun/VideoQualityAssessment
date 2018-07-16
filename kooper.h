//
// Created by zer0like on 2018/7/12.
//

#ifndef HOLE_FILLING_kooper_H
#define HOLE_FILLING_kooper_H

#include "hole_filling.h"


class kooper {
public:
	kooper(std::string t_video_path, std::string d_video_path, int distance);
    cv::Mat A6Porcess(cv::Mat &I_ref,cv::Mat &D_ref);
    int depthfill(cv::Mat &D_syn,cv::Mat &D_out);
    void spriteUpdate(cv::Mat &D_syn,cv::Mat &I_syn,cv::Mat &D_sprite,cv::Mat &I_sprite,int cmin_median);
    void ImageUpdate(cv::Mat &I_syn,cv::Mat &D_syn,cv::Mat &I_sprite,cv::Mat &D_sprite,double beta,cv::Vec3b hole_value);
    void initialFill(cv::Mat &I_syn,cv::Mat &I_fill);
    void textrureRefinement(cv::Mat& I_syn,cv::Mat &I_fill,cv::Mat &I_refine,int radius);
    void generateSprite(std::string t_video_path,std::string d_video_path,std::string out_t_path,std::string out_d_path,int count);
    void extra_process(cv::Mat &I_syn,cv::Mat &I_refine,int radius);

private:
    std::vector<cv::Point> findHoleList(cv::Mat &I_syn,cv::Vec3b hole_value);
    void setGradientMap(cv::Mat &I_syn,cv::Mat &Src_Map);
    void LossFunction(cv::Mat& I_syn,cv::Mat &I_fill,cv::Mat &I_map,cv::Mat &I_fill_map,cv::Mat &I_refine,int radius,cv::Point HoleLoc);
    void deleteEdge(cv::Mat& D_sprite,cv::Mat& I_sprite,int size);
	cv::Mat getPatch(const cv::Mat & image, const cv::Point& p, int radius);
	std::vector<cv::Point> FindPoint(cv::Point &p, int with, int height, int range, int radius);

    cv::Vec3b hole_value;
    std::vector<cv::Point> HoleList;
    cv::Mat I_sprite;
    cv::Mat D_sprite;

	int distance;

};


#endif //HOLE_FILLING_kooper_H
