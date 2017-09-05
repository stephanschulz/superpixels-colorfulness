#pragma once

#include "slic.h"

#include "ofxCv.h"


#include "ofxGui.h"
#include "ofMain.h"

#define INIT_N_SUPERPIXELS 100
#define INIT_N_WEIGHT_FACTOR 100
#define IMG_RESIZE_SIZE 800

class ofApp : public ofBaseApp{

public:
    void setup();
    void update();
    void draw();

    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void mouseEntered(int x, int y);
    void mouseExited(int x, int y);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
    
    void checkGui();
    
    ofVideoGrabber cam;
    ofImage camImg;
    int camWidth, camHeight;
    
       Slic slic;
    void generateSuperpixels(ofPixels pix, int nr_superpixels, int nc);
    
    ofxPanel gui_main;
    
    ofParameter<int> nSuperpixels;
    int old_nSuperpixels;
    ofParameter<int> nWeightFactors;
    int old_nWeightFactors;
    
//    int nSuperpixels;
//    int nWeightFactors;
    
    ofImage img;

    cv::Mat image;
    cv::Mat lab_image;
};
