#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <dirent.h>

using namespace cv;
using namespace std;

char*sample_path;
float rate=1;
int rect_width=112,rect_height=24,X,Y;
int manual_sample_count,machine_sample_count;

Mat sample_image;
string file_index;
cv::Mat org,dst,img,tmp;

HOGDescriptor initialize_descriptor_by_file(ifstream &fin){

    HOGDescriptor myHOG(Size(112,24),Size(16,16),Size(16,8),Size(8,8),9);//HOG检测器，设置HOGDescriptor的检测子,用来计算HOG描述子的

    float val = 0.0f;
    vector<float> myDetector;
    while(!fin.eof())
    {
        fin>>val;
        myDetector.push_back(val);
    }
    fin.close();
    myDetector.pop_back();

    myHOG.setSVMDetector(myDetector);
    //myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    return myHOG;
}

Rect redraw_img(Point left_up,Point right_down){
    org.copyTo(img);
    //rectangle(InputOutputArray _img,Point pt1,Point pt2,const Scalar& color,int thickness,int lineType,int shift)
    rectangle(img,left_up,right_down,Scalar(0,255,0),1,8,0);
    imshow("img",img);
}

void on_mouse(int event,int x,int y,int flags,void *ustc)//event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号
{
    X=x;
    Y=y;

    int val_width=(int)((rate*rect_width)/2),val_height=(int)((rate*rect_height)/2);
    Point left_up=Point(x-val_width,y-val_height),right_down=Point(x+val_width,y+val_height);
    redraw_img(left_up,right_down);

    if (event == CV_EVENT_LBUTTONUP)//左键松开，将在图像上划矩形
    {
        sample_image=org(Rect(left_up.x,left_up.y,right_down.x-left_up.x,right_down.y-left_up.y));

        resize(sample_image,sample_image,Size(112,24),0,0,CV_INTER_AREA);

        snprintf(sample_path,250,"/Users/lan/Desktop/TarReg/svm/crop_samples/pos_samples/manhards148_0504pm_0510pm/manual0508pm_%s_%d.bmp",file_index.c_str(),manual_sample_count);
        //snprintf(sample_path,250,"/Users/lan/Desktop/TarReg/svm/crop_samples/pos_samples/%s.bmp",file_names[i]);
        //snprintf(sample_path,250,"/Users/lan/Desktop/TarReg/svm/crop_samples/pos_samples/neg/manneg_%s_%d.bmp",file_index.c_str(),manual_sample_count);
        cout<<sample_path<<endl;
        manual_sample_count++;
        imwrite(sample_path,sample_image);
    }
}

void getFiles( string path, vector<string>& files )
{
    DIR  *dir;
    struct dirent  *ptr;
    dir = opendir(path.c_str());
    string pathName;

    while((ptr = readdir(dir)) != NULL){
        if(ptr->d_name[0]!='.'&&ptr->d_name[strlen(ptr->d_name)-4]=='.'){
            files.push_back(pathName.assign(path).append("/").append(string(ptr->d_name)));
        }
    }
}

string get_file_index(string file_name){
    int pre_index=file_name.rfind("_"),post_index=file_name.rfind(".");
    //cout<<pre_index<<" "<<post_index<<endl;
    return file_name.substr(pre_index+1,post_index-pre_index-1);
}

Mat machine_cropped(Mat &src,HOGDescriptor myHOG){
    Mat drawed_img;
    src.copyTo(drawed_img);
    char *saveName=(char*)malloc(300*sizeof(char));//set memories to save the file name of hard examples

    vector<Rect> found;// vector array of foundlocation

    myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(0,0), 1.05, 2); //1.05->36 levels, 1.1->17 levels, 1.12->14 levels

    //go through all of the detected targets, get the hard examples
    int len = found.size();
    for(int i=0; i < len; i++)
    {
        cout << "found.size() : " << len << endl;
        Rect r = found[i];
/***** guarantee the targeted view in the picture, resize the window inside the image, because sometime the window out of the image
//        if(r.x < 0)
//            r.x = 0;
//        if(r.y < 0)
//            r.y = 0;
//        if(r.x + r.width > src.cols)
//            r.width = src.cols - r.x;
//        if(r.y + r.height > src.rows)
//            r.height = src.rows - r.y;
******/
        if (len == 1)
            rectangle(drawed_img, r.tl(), r.br(), Scalar(0,255,255), 1);
        else
        {
            float area_inter, area_r=(float)r.area();
            for( int j = 0; j < len; j++)
            {
                area_inter = (float) (Rect(r & found[j]).area());
                if(j > i && area_inter > area_r*0.5) //j!=i, exception itself
                {
                    Point mintl, maxbr;
                    mintl.x = min(found[i].tl().x,found[j].tl().x);
                    mintl.y = min(found[i].tl().y,found[j].tl().y);
                    maxbr.x = max(found[i].br().x,found[j].br().x);
                    maxbr.y = max(found[i].br().y,found[j].br().y);
                    rectangle(drawed_img, mintl, maxbr, Scalar(0,255,255), 1);
                }
            }
        }

        snprintf(saveName,250,"/Users/lan/Desktop/TarReg/svm/crop_samples/pos_samples/autohards_148_0504pm_0510pm/auto0510pm_%s_%d.bmp",file_index.c_str(),machine_sample_count);

        //将矩形框保存为图片，就是Hard Example
        //Mat hardExampleImg = src(r);//从原图上截取矩形框大小的图片
        //resize(hardExampleImg,hardExampleImg,Size(112,24),0,0,CV_INTER_AREA);//将剪裁出来的图片缩放为112*24大小
        //imwrite(saveName, hardExampleImg);//保存文件
        machine_sample_count++;
        //hardExampleImg.release();
    }

    return drawed_img;
}

int main()
{
    ifstream fin("/Users/lan/Desktop/TarReg/svm/svmrobot/training/HOGDetector0502robot.txt", ios::in);
    HOGDescriptor myHOG=initialize_descriptor_by_file(fin);

    float rate_increment=1.03;
    manual_sample_count=0;
    machine_sample_count=0;
    sample_path=(char*)malloc(300*sizeof(char));
    vector<string> file_names;
    //49_0502_800_600/30_0503_800_600/148_0504pm/
    getFiles("/Users/lan/Desktop/TarReg/svm/crop_samples/tobecroped/49_0502_800_600/",file_names);

    namedWindow("img");// define a image window
    for(int i=0;i<file_names.size();i++){
        cout<<file_names[i]<<endl;

        org = imread(file_names[i]);
        file_index.assign(get_file_index(file_names[i]));

        imshow("detected",machine_cropped(org,myHOG));

//        org.copyTo(img);
//        setMouseCallback("img",on_mouse,0);//调用回调函数
//        imshow("img",img);
//        cout<<"---------3------------"<<endl;

        while(true){
            int k=cvWaitKey(10);

            if(k=='w'){
                rate*=rate_increment;
                int val_width=(int)((rate*rect_width)/2),val_height=(int)((rate*rect_height)/2);
                Point left_up=Point(X-val_width,Y-val_height),right_down=Point(X+val_width,Y+val_height);
                redraw_img(left_up,right_down);
            }
            if(k=='s'){
                rate/=rate_increment;
                rate=rate<1?1:rate;
                int val_width=(int)((rate*rect_width)/2),val_height=(int)((rate*rect_height)/2);
                Point left_up=Point(X-val_width,Y-val_height),right_down=Point(X+val_width,Y+val_height);
                redraw_img(left_up,right_down);
            }
            if(k=='d'){
                break;
            }
            if(k=='a'){
                i=(i-2)<0?-1:(i-2);
                break;
            }
        }
    }

    return 0;
}