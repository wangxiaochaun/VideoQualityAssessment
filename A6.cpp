//
// Created by zer0like on 2018/7/12.
//

#include <opencv2/core/mat.hpp>
#include "A6.h"
#include "utils.h"
#include "hole_filling.h"

A6::A6(std::string t_video_path,std::string d_video_path){
    this->generateSprite(t_video_path,d_video_path,"../","../",10);
}

cv::Mat A6::A6Porcess(cv::Mat &I_ref,cv::Mat &D_ref){
    cv::Mat I_syn = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8UC3,cv::Scalar(0,0,0));
    cv::Mat D_syn = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8U,cv::Scalar(0));

    cv::Mat I_fill = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8UC3,cv::Scalar(255,255,255));
    cv::Mat I_refine = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8UC3,cv::Scalar(255,255,255));
    cv::Mat I_extra = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8UC3,cv::Scalar(0,0,0));

    cv::Mat I_final = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8UC3,cv::Scalar(255,255,255));
    cv::Mat I_mask = cv::Mat(I_fill.size().height, I_fill.size().width, CV_8U,cv::Scalar(0));

    algorithm test;

    cv::Vec3b holevlaue = (0,0,0);
    clock_t start = clock();

    test.warping_1d(I_ref,D_ref,I_syn,D_syn);

    int median = this->depthfill(D_syn,I_syn);

    this->ImageUpdate(I_syn,D_syn,this->I_sprite,this->D_sprite,15,cv::Vec3b(0,0,0));

    this->initialFill(I_syn,I_fill);

    this->textrureRefinement(I_syn,I_fill,I_refine,2);

    this->extra_process(I_fill,I_extra,6);


    for (int row = 0;row<I_fill.rows;row++){
        for(int col = 0 ;col < I_fill.cols;col++){
            if(I_fill.at<cv::Vec3b>(row,col)[0] == holevlaue[0]){
                I_mask.at<uchar>(row,col) = 255;
            }
        }
    }

    cv::inpaint(I_fill,I_mask,I_final,3, cv::INPAINT_NS);

    clock_t ends = clock();

    std::cout << "Running time of A6 porcess (ms): " << static_cast<double>(ends - start) / CLOCKS_PER_SEC * 1000 << std::endl;

    return I_final;
}
std::vector<cv::Point> A6::findHoleList(cv::Mat &I_syn,cv::Vec3b hole_value){
    std::vector<cv::Point> Holelist;
    for(int row =0 ;row<I_syn.rows;row++){
        for(int col = 0 ;col<I_syn.cols;col++){
            if(I_syn.at<cv::Vec3b>(row,col) == hole_value){
                //if(Patch.at<uchar>(row,col) > center_min ){
                //attention Point.x means cols Point.y means row,so in this palce row means the y is the point.y
                Holelist.push_back(cv::Point(col,row));
                //}
            }
        }

    }
    return Holelist;
}
/*
 * read a image and then fill the region of this map
 * */
int A6::depthfill(cv::Mat &D_syn,cv::Mat &D_out) {

	element a;
    int  D_width = D_syn.cols;
    int  D_height = D_syn.rows;
    int  radius = 4;
    cv::Point p(0,0);
    int cmin_sum =0;
    int count = 0;
    for(int x=radius; x < D_width-radius; x+=radius*2+1){
        for(int y=radius; y < D_height-radius; y+=radius*2+1){
            count+=1;
            p.x = x;
            p.y = y;
            //std::cout<<"x="<<x<<"  y="<<y<<std::endl;
            //get the patch
            cv::Mat Patch = a.getPatch(D_syn,p,radius);
            int sampleCount = Patch.cols * Patch.rows;

//            cv::Mat_<uchar> ::iterator begin = Patch.begin<uchar>();
//            cv::Mat_<uchar> ::iterator end = Patch.end<uchar>();
//
//            while(begin != end){
//                if (*begin == 0 ) sampleCount-=1;
//                begin++;
//            }

            for(int row =0 ;row<Patch.rows;row++){
                for(int col = 0 ;col<Patch.cols;col++){
                    if(Patch.at<uchar>(row,col)==0) sampleCount-=1;
                    //std::cout<<points.at<float>(index)<<" ";
                }

            }

            //calculate the k_means
            if (sampleCount <= 2 ) continue;
            int clusterCount = 2;
            cv::Mat points(sampleCount,1,CV_32F,cv::Scalar(0));

            cv::Mat labels;
            cv::Mat centers(clusterCount,1,points.type());
            int index =0;
            for(int row =0 ;row<Patch.rows;row++){
                for(int col = 0 ;col<Patch.cols;col++){
                    //index = row* Patch.cols + col;
                    int value = Patch.at<uchar>(row,col);
                    if (value ==0) continue;
                    else points.at<float>(index++) = value;
                    //std::cout<<points.at<float>(index-1)<<" ";
                }

            }
            //std::cout<<std::endl;

            cv::TermCriteria criteria = cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,10,1.0);

            cv::kmeans(points,clusterCount,labels,criteria,1,cv::KMEANS_RANDOM_CENTERS,centers);
            //std::cout<<centers.cols<<"  "<<centers.rows << std::endl;

            //std::cout<<centers.at<float>(0,0)<<std::endl;
            //std::cout<<centers.at<float>(0,1)<<std::endl;

            int center_min = centers.at<float>(0,0)>centers.at<float>(0,1)?centers.at<float>(0,1):centers.at<float>(0,0);
            cmin_sum += center_min;
            //std::cout<<"cmin="<<center_min<<std::endl;

            // compare all patch and to change the value if
            for(int row =0 ;row<Patch.rows;row++){
                for(int col = 0 ;col<Patch.cols;col++){
                    if(Patch.at<uchar>(row,col) == 0){
                        //if(Patch.at<uchar>(row,col) > center_min ){
                            Patch.at<uchar>(row,col) = center_min;
                        //}
                    }
                }

            }
        }
    }

    //std::cout<<D_syn.size<<std::endl;

//    cv::namedWindow("test");
//    cv::imshow("test",D_syn);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
    //std::cout<<cmin_sum*1.0f/count<<std::endl;
    return cmin_sum*1.0f/count;
}

void A6::spriteUpdate(cv::Mat &D_syn,cv::Mat &I_syn,cv::Mat &D_sprite,cv::Mat &I_sprite,int cmin_median){
    //using the cmin_median to update the picture if the pixel is less than cmin_mdian update it
    for(int row =0;row<D_syn.rows;row++)
    {
        for(int col =0 ;col<D_syn.cols;col++)
        {
            int value = D_syn.at<uchar>(row,col);
            if (value < cmin_median){
                D_sprite.at<uchar>(row,col) = value;
                I_sprite.at<cv::Vec3b>(row,col) = I_syn.at<cv::Vec3b>(row,col);
            }
        }
    }
    this->deleteEdge(D_sprite,I_sprite,1);
}

void A6::deleteEdge(cv::Mat& D_sprite,cv::Mat& I_sprite,int size){
    for(int row =0 ;row<D_sprite.rows-1;row++){
        for(int col =0; col<D_sprite.cols-1;col++){
            uchar current = D_sprite.at<uchar>(row,col);
            uchar next = D_sprite.at<uchar>(row,col+1);


            if (current == 0 && next !=0){
                //this is the left to right edge;
                D_sprite.at<uchar>(row,col) = 0;

                D_sprite.at<uchar>(row,col+1) = 0;
                I_sprite.at<cv::Vec3b>(row,col) = 0;
                I_sprite.at<cv::Vec3b>(row,col+1) = 0;
                col+=1;
            }
            else if(current != 0 && next ==0){
                //this is the right to left edge;
                D_sprite.at<uchar>(row,col) = 0;
                D_sprite.at<uchar>(row,col+1) = 0;
                I_sprite.at<cv::Vec3b>(row,col) = 0;
                I_sprite.at<cv::Vec3b>(row,col+1) = 0;
                col+=1;
            }
            else{
                //not the edge
            }

        }
    }

    for(int col = 0;col<D_sprite.cols;col++){
        for (int row = 0; row<D_sprite.rows-1;row++){
            uchar current = D_sprite.at<uchar>(row,col);
            uchar down = D_sprite.at<uchar>(row+1,col);

            if(current==0&&down !=0){
                D_sprite.at<uchar>(row,col) = 0;
                D_sprite.at<uchar>(row+1,col) = 0;
                I_sprite.at<cv::Vec3b>(row,col) = 0;
                I_sprite.at<cv::Vec3b>(row+1,col) = 0;
                row+=1;
            }else if(current!=0&&down ==0){
                D_sprite.at<uchar>(row,col) = 0;
                D_sprite.at<uchar>(row+1,col) = 0;
                I_sprite.at<cv::Vec3b>(row,col) = 0;
                I_sprite.at<cv::Vec3b>(row+1,col) = 0;
                row+=1;
            }else{

            }
        }
    }
}

void A6::ImageUpdate(cv::Mat &I_syn,cv::Mat &D_syn,cv::Mat &I_sprite,cv::Mat &D_sprite,double beta,cv::Vec3b hole_value){
    std::vector<cv::Point> HoleList = this->findHoleList(I_syn,hole_value);


    for (auto hole:HoleList){
        if(D_sprite.at<uchar>(hole) == 0) continue;

        int D = D_syn.at<uchar>(hole);
        int G = D_sprite.at<uchar>(hole);
        if(D < G+beta) I_syn.at<cv::Vec3b>(hole) = I_sprite.at<cv::Vec3b>(hole);
        else I_syn.at<cv::Vec3b>(hole)=0;
    }
}

void A6::initialFill(cv::Mat &I_syn,cv::Mat &I_fill) {
    I_fill = I_syn.clone();
    cv::Vec3b hole_value = (0, 0, 0);
    //std::cout<<"size = "<<I_syn.size << " cols = " <<I_syn.cols;
    for (int row = 1; row < I_syn.rows-2; row++) {
        for (int col = 1; col < I_syn.cols-2; col++) {
            if(I_fill.at<cv::Vec3b>(row,col) == hole_value) {
                assert(row-1 >= 0 && col -1 >=0 && row+1 < I_syn.rows -1 && col+1 <I_syn.cols -1);
                int all_b = I_fill.at<cv::Vec3b>(row-1,col)[0]+I_fill.at<cv::Vec3b>(row,col-1)[0]+I_fill.at<cv::Vec3b>(row+1,col)[0]+I_fill.at<cv::Vec3b>(row,col+1)[0];
                int all_g = I_fill.at<cv::Vec3b>(row-1,col)[1]+I_fill.at<cv::Vec3b>(row,col-1)[1]+I_fill.at<cv::Vec3b>(row+1,col)[1]+I_fill.at<cv::Vec3b>(row,col+1)[1];
                int all_r = I_fill.at<cv::Vec3b>(row-1,col)[2]+I_fill.at<cv::Vec3b>(row,col-1)[2]+I_fill.at<cv::Vec3b>(row+1,col)[2]+I_fill.at<cv::Vec3b>(row,col+1)[2];
                //cv::Vec3b all =I_fill.at<cv::Vec3b>(row-1,col)+I_fill.at<cv::Vec3b>(row,col-1)+I_fill.at<cv::Vec3b>(row+1,col)+I_fill.at<cv::Vec3b>(row,col+1);
                I_fill.at<cv::Vec3b>(row,col)[0] = all_b/4;
                I_fill.at<cv::Vec3b>(row,col)[1] = all_g/4;
                I_fill.at<cv::Vec3b>(row,col)[2] = all_r/4;
            }
        }
    }
//    for (int row = 1; row < I_syn.rows-2; row++) {
//        for (int col = 1; col < I_syn.cols-2; col++) {
//            if(I_syn.at<cv::Vec3b>(row,col) == hole_value) {
//                I_syn.at<cv::Vec3b>(row,col) = (100,100,100);
//            }
//        }
//    }

}

void A6::setGradientMap(cv::Mat &I_syn,cv::Mat &Src_Map){
    cv::Mat tempImage;
    cv::Mat soblex;
    cv::Mat soblex_pow;
    cv::Mat sobley;
    cv::Mat sobley_pow;

    cv::cvtColor(I_syn,tempImage,CV_BGR2GRAY);
    cv::Sobel(tempImage,soblex,CV_32F,1,0,3,1,1,cv::BORDER_DEFAULT);
    cv::pow(soblex,2,soblex_pow);

    cv::Sobel(tempImage,sobley,CV_32F,0,1,3,1,1,cv::BORDER_DEFAULT);
    cv::pow(sobley,2,sobley_pow);
    cv::sqrt(soblex_pow+sobley_pow,Src_Map);
}

void A6::LossFunction(cv::Mat& I_syn,cv::Mat &I_fill,cv::Mat &I_map,cv::Mat &I_fill_map,cv::Mat &I_refine,int radius,cv::Point HoleLoc){
	element a;
    cv::Mat I_syn_region = a.getPatch(I_syn,HoleLoc,radius);
    //std::cout<<I_syn_region.size<<std::endl;
    cv::Mat I_fill_region = a.getPatch(I_fill,HoleLoc,radius);
    cv::Mat I_map_region = a.getPatch(I_map,HoleLoc,radius);
    cv::Mat I_fill_map_region = a.getPatch(I_fill_map,HoleLoc,radius);

    std::vector<cv::Point> loc = a.FindPoint(HoleLoc,I_syn.cols,I_syn.rows,5,radius);
//    std::vector<cv::Point> loc = FindPoint(HoleLoc,I_fill.cols,I_fill.rows,5,radius);

    int E = 100000;
    int findLoc = 0;
    for (int i=0;i<loc.size();i++){
        //std::cout<<loc[i].x <<"&"<<loc[i].y<<std::endl;
        if (radius > loc[i].x || loc[i].x >=I_syn.cols-radius || radius > loc[i].y || loc[i].y >= I_syn.rows-radius) continue;
        //if (radius > loc[i].x || loc[i].x >=I_map.cols-radius || radius > loc[i].y || loc[i].y >= I_map.rows-radius) continue;
        cv::Mat selected_I_syn_region = a.getPatch(I_syn,loc[i],radius);
        //std::cout<<selected_I_syn_region.size<<std::endl;
        cv::Mat selected_I_fill_region = a.getPatch(I_fill,loc[i],radius);
        cv::Mat selected_I_map_region = a.getPatch(I_map,loc[i],radius);
        cv::Mat selected_I_fill_map_region = a.getPatch(I_fill_map,loc[i],radius);


        cv::Mat syn_diff;
        cv::absdiff(I_syn_region,selected_I_syn_region,syn_diff);
        cv::Mat fill_diff;
        cv::absdiff(I_fill_region,selected_I_fill_region,fill_diff);
        cv::Mat map_diff;
        cv::absdiff(I_map_region , selected_I_map_region,map_diff);
        cv::Mat fillmap_diff ;
        cv::absdiff(I_fill_map_region,selected_I_fill_map_region,fillmap_diff);

        //part 1
        double P1 =0;
        cv::Mat syn_diff_pow;
        cv::pow(syn_diff,2,syn_diff_pow);
        P1 = cv::sum(syn_diff_pow)[0];

        //part 2
        double P2 = 0;
        cv::Mat fill_diff_pow;
        cv::pow(fill_diff,2,fill_diff_pow);
        P2 = cv::sum(fill_diff_pow)[0];

        //part 3
        double P3 = 0;
        cv::Mat map_diff_pow;
        cv::pow(map_diff,2,map_diff_pow);
        P3 = cv::sum(map_diff_pow)[0];

        //part 4
        double P4 = 0;
        cv::Mat fillmap_diff_pow;
        cv::pow(fillmap_diff,2,fillmap_diff_pow);
        P4 = cv::sum(fillmap_diff_pow)[0];


        int result = P1 +0.6*P2+0.6*P3+0.6*0.6*P4;

        if(result<E){
            findLoc=i;
            E = result;
        }
    }
    //std::cout<<loc[findLoc].x<<" "<<loc[findLoc].y<<std::endl;
    cv::Mat I_refine_region = a.getPatch(I_refine,HoleLoc,radius);
    cv::Mat selected_I_syn_region = a.getPatch(I_syn,loc[findLoc],radius);
    I_refine.at<cv::Vec3b>(HoleLoc) = I_fill.at<cv::Vec3b>(loc[findLoc]);
    I_fill.at<cv::Vec3b>(HoleLoc) = I_fill.at<cv::Vec3b>(loc[findLoc]);
    //I_refine_region = selected_I_syn_region.clone();
//    for (int row = 0; row < I_refine_region.rows; row++) {
//        for (int col = 0; col < I_refine_region.cols; col++) {
//            I_refine_region.at<cv::Vec3b>(row,col) = selected_I_syn_region.at<cv::Vec3b>(row,col);
//            I_syn_region.at<cv::Vec3b>(row,col) = selected_I_syn_region.at<cv::Vec3b>(row,col);
//
//        }
//    }
    //I_refine_region = selected_I_syn_region;
}

void A6::textrureRefinement(cv::Mat& I_syn,cv::Mat &I_fill,cv::Mat &I_refine,int radius){
    cv::Vec3b hole_value = (0,0,0);
    std::vector<cv::Point> HoleList = this->findHoleList(I_syn,hole_value);
    this->HoleList = HoleList;



    cv::Mat I_map;
    cv::Mat I_fill_map;

    this->setGradientMap(I_syn,I_map);
    this->setGradientMap(I_fill,I_fill_map);
    //std::cout<<"I_map:"<<I_map.size<<std::endl;
    //std::cout<<"I_fill_map:"<<I_fill_map.size<<std::endl;
    //std::cout<<"I_refine:"<<I_refine.size<<std::endl;

    for(auto hole : HoleList){
        if (radius > hole.x || hole.x >=I_syn.cols-radius || radius > hole.y || hole.y >= I_syn.rows-radius) continue;
        this->LossFunction(I_syn,I_fill,I_map,I_fill_map,I_refine,radius,hole);
        //std::cout<<"processed!"<<hole.x<<"  "<<hole.y << std::endl;
    }
}

void A6::generateSprite(std::string t_video_path,std::string d_video_path,std::string out_t_path,std::string out_d_path,int count){

    cv::VideoCapture cap_texture;
    cv::VideoCapture cap_depth;

    cap_texture.open(t_video_path);
    cap_depth.open(d_video_path);

    //read every frame and then change warping and get the D and I sprite
    int frame_num = cap_texture.get(cv::CAP_PROP_FRAME_COUNT);
    int height = cap_texture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int width = cap_texture.get(cv::CAP_PROP_FRAME_WIDTH);

    cv::Mat t_frame;
    cv::Mat d_frame;

    cv::Mat I_syn = cv::Mat(height, width, CV_8UC3,cv::Scalar(0,0,0));
    cv::Mat D_syn = cv::Mat(height, width, CV_8U,cv::Scalar(0));
    cv::Mat I_sprite = cv::Mat(height, width,CV_8UC3,cv::Scalar(0,0,0));
    cv::Mat D_sprite = cv::Mat(height, width,CV_8U,cv::Scalar(0));

    algorithm test;

    for(int i=0;i<count;i++){

        cap_texture>>t_frame;
        cap_depth>>d_frame;
        cv::cvtColor(d_frame, d_frame, CV_RGB2GRAY);


        test.warping_1d(t_frame,d_frame,I_syn,D_syn);

        int median = this->depthfill(D_syn,I_syn);

        this->spriteUpdate(D_syn,I_syn,D_sprite,I_sprite,median);
    }

    cap_texture.release();
    cap_depth.release();

    this->I_sprite = I_sprite;
    this->D_sprite = D_sprite;

    cv::imwrite(out_t_path+"t_sprite.bmp",I_sprite);
    cv::imwrite(out_d_path+"d_sprite.bmp",D_sprite);


}

void A6::extra_process(cv::Mat &I_syn,cv::Mat &I_refine,int radius){
	element a;
    for(auto hole : this->HoleList){
        if (radius > hole.x || hole.x >=I_syn.cols-radius || radius > hole.y || hole.y >= I_syn.rows-radius) continue;
        cv::Mat hole_region = a.getPatch(I_syn,hole,radius);

        int sum_b = 0;
        int sum_g = 0;
        int sum_r = 0;

        int count = 0;
        for(int row = 0;row < hole_region.rows -1;row++){
            for(int col = 0;col < hole_region.cols-1;col++){
                if (hole_region.at<cv::Vec3b>(row,col) == I_syn.at<cv::Vec3b>(row,col)) continue;
                sum_b += hole_region.at<cv::Vec3b>(row,col)[0];
                sum_g += hole_region.at<cv::Vec3b>(row,col)[1];
                sum_r += hole_region.at<cv::Vec3b>(row,col)[2];
                count++;
            }
        }

        //std::cout<<sum_b/count<<std::endl;

        I_syn.at<cv::Vec3b>(hole)[0] = sum_b / count;
        I_syn.at<cv::Vec3b>(hole)[1] = sum_g / count;
        I_syn.at<cv::Vec3b>(hole)[2] = sum_r / count;
        I_refine.at<cv::Vec3b>(hole)[0] = sum_b / count;
        I_refine.at<cv::Vec3b>(hole)[1] = sum_g / count;
        I_refine.at<cv::Vec3b>(hole)[2] = sum_r / count;
    }
}
