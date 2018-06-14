#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"
#define OPENCV
extern "C"{
#include "darknet.h"
}

using namespace cv;
void ipl_into_image(Mat src, image im)
{
    unsigned char *data = (unsigned char *)src.data;
    int h = src.rows;
    int w = src.cols;
    int c = src.channels();
    int step = src.step;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
}

image ipl_to_image(Mat src)
{
    int h = src.rows;
    int w = src.cols;
    int c = src.channels();
    image out = make_image(w, h, c);
    ipl_into_image(src, out);
    return out;
}
static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
void show_image_cv(image p, const char *name, IplImage *disp)
{
    int x,y,k;
    if(p.c == 3) rgbgr_image(p);
    //normalize_image(copy);

    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s", name);

    int step = disp->widthStep;
    cvNamedWindow(buff, CV_WINDOW_NORMAL);
    //cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
    int windows = 0;
    for(y = 0; y < p.h; ++y){
        for(x = 0; x < p.w; ++x){
            for(k= 0; k < p.c; ++k){
                disp->imageData[y*step + x*p.c + abs(k-2)] = (unsigned char)(get_pixel(p,x,y,k)*255);
            }
        }
    }
    if(0){
        int w = 448;
        int h = w*p.h/p.w;
        if(h > 1000){
            h = 1000;
            w = h*p.w/p.h;
        }
        IplImage *buffer = disp;
        disp = cvCreateImage(cvSize(w, h), buffer->depth, buffer->nChannels);
        cvResize(buffer, disp, CV_INTER_LINEAR);
        cvReleaseImage(&buffer);
    }
    cvShowImage(buff, disp);
}

float t_colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * t_colors[i][c] + ratio*t_colors[j][c];
    //printf("%f\n", r);
    return r;
}

void t_draw_detections(Mat im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes)
{
    int i,j;

    for(i = 0; i < num; ++i){
        char labelstr[4096] = {0};
        int v_class = -1;
        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j] > thresh){
                if (v_class < 0) {
                    strcat(labelstr, names[j]);
                    v_class = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
            }
        }
        if(v_class >= 0){
            int width = im.rows * .006;

            /*
               if(0){
               width = pow(prob, 1./2.)*10+1;
               alphabet = 0;
               }
             */

            //printf("%d %s: %.0f%%\n", i, names[v_class], prob*100);
            int offset = v_class*123457 % classes;
            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];

            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = dets[i].bbox;
            //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

            int left  = (b.x-b.w/2.)*im.cols;
            int right = (b.x+b.w/2.)*im.cols;
            int top   = (b.y-b.h/2.)*im.rows;
            int bot   = (b.y+b.h/2.)*im.rows;

            if(left < 0) left = 0;
            if(right > im.cols-1) right = im.cols-1;
            if(top < 0) top = 0;
            if(bot > im.rows-1) bot = im.rows-1;

            rectangle(im, Point(left, top), Point(right, bot), Scalar(red, green, blue),width);
            // if (alphabet) {
            //     image label = get_label(alphabet, labelstr, (im.rows*.03));
            //     draw_label(im, top + width, left, label, rgb);
            //     free_image(label);
            // }

        }
    }
}

int main()
{
  list *options = read_data_cfg("cfg/coco.data");
  char *name_list = option_find_str(options, "names", "data/coco.names");
  char **names = get_labels(name_list);
  image **alphabet = load_alphabet();
  network *net = load_network((char *)"cfg/yolov3.cfg",(char *)"cfg/yolov3.weights",0);
  layer l = net->layers[net->n-1];
  set_batch_network(net, 1);
  float nms=.4;
  // VideoCapture feed("vtest.avi");
  VideoCapture feed("rtsp://192.168.0.204/11");
  Mat input_image;
  Mat resized_image;

  float thresh = 0.5;
  printf("Size of float %ld\n",sizeof(float));
  while(true)
  {
    feed.read(input_image);
    if(input_image.rows*input_image.cols)
    {
      image dark_image = ipl_to_image(input_image);
      image sized = resize_image(dark_image, net->w, net->h);
      network_predict(net,sized.data);
      int nboxes = 0;
      detection *dets = get_network_boxes(net, 1, 1, thresh, 0, 0, 0, &nboxes);
      if (nms) do_nms_sort(dets, l.side*l.side*l.n, l.classes, nms);
      // int num = l.side*l.side;
      // printf("%d-num\n",num);
      // int classes = 20;
      // for(int i = 0; i< num; ++i)
      // {
      //   for(int j = 0; j < classes; ++j)
      //   {
      //     printf("%f\n",dets[i].prob[j]);
      //     if (dets[i].prob[j] > thresh)
      //     {
      //
      //         printf("%s: %.0f%%\n", voc_names[j], dets[i].prob[j]*100);
      //     }
      //   }
      // }
      t_draw_detections(input_image, dets, nboxes, thresh, names, alphabet, 20);
      imshow("output",input_image);
      waitKey(1);
      free_image(dark_image);
      free_image(sized);
      input_image.release();
    }
  }
  destroyAllWindows();
}
