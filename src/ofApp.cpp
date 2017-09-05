#include "ofApp.h"

using namespace ofxCv;
using namespace cv;


void ofApp::setup(){
    
    camWidth = 320; //640;
    camHeight = 240; //480;
    
    cam.listDevices();
    cam.setDeviceID(0);
//    cam.setup(1920, 1080);
 cam.setup(camWidth, camHeight);
    /* setup gui */
    
    gui_main.setup("splitting");
    gui_main.setPosition(20,20);
    gui_main.setDefaultHeaderBackgroundColor(ofColor(255,0,0));
    gui_main.add(nSuperpixels.set("superpixels",INIT_N_SUPERPIXELS,0,500));
    gui_main.add(nWeightFactors.set("weightFactor",INIT_N_SUPERPIXELS,0,500));
    gui_main.loadFromFile("gui_main.xml");

    old_nSuperpixels = nSuperpixels;
    old_nWeightFactors = nWeightFactors;
    
//    nSuperpixels = INIT_N_SUPERPIXELS;
//    nWeightFactors = INIT_N_WEIGHT_FACTOR;
    
    /* generate superpixels on example image */
    
//    img.load("dog.jpg");
//    generateSuperpixels(img, 100, 100);
    
}

void ofApp::update(){
    ofSetWindowTitle(ofToString(ofGetFrameRate()));
    
    checkGui();
    cam.update();
    if(cam.isFrameNew()){
        camImg.setFromPixels(cam.getPixels());
        if(ofGetMousePressed() == false) generateSuperpixels(camImg,  nSuperpixels, nWeightFactors);
    }
}

void ofApp::draw(){
    
    
    
//    dogCvColorImg.draw(ofGetWidth()/2 - img.getWidth()/2,
//                       ofGetHeight()/2 - img.getHeight()/2);
    
      ofSetColor(255);
    camImg.draw(0,0);
//    img.draw(0, 0);
    drawMat(image, camWidth, 0);
    
    drawMat(lab_image, camWidth, camHeight);
    
    ofPushMatrix();
    ofTranslate(0, camHeight);
    slic.display_contours(ofColor(255,1,1));
    ofPopMatrix();
    
    ofPushMatrix();
    ofTranslate(camWidth, 0);
    slic.drawPixelGroups();
     ofPopMatrix();
    
    ofPushMatrix();
    ofTranslate(camWidth*2, 0);
    slic.drawPixelGroupsImage(4);
    ofPopMatrix();
    
//    cam.draw(0,0);
    gui_main.draw();
    
    
}

void ofApp::keyPressed(int key){

}

void ofApp::keyReleased(int key){
    if(key == 'g')  gui_main.saveToFile("gui_main.xml");
}

void ofApp::mouseMoved(int x, int y ){

}

void ofApp::mouseDragged(int x, int y, int button){

}

void ofApp::mousePressed(int x, int y, int button){

}

void ofApp::mouseReleased(int x, int y, int button){
    
}

void ofApp::mouseEntered(int x, int y){

}

void ofApp::mouseExited(int x, int y){

}

void ofApp::windowResized(int w, int h){

}

void ofApp::gotMessage(ofMessage msg){

}

void ofApp::checkGui(){
    if(old_nSuperpixels != nSuperpixels || old_nWeightFactors != nWeightFactors){
        old_nSuperpixels = nSuperpixels;
        old_nWeightFactors = nWeightFactors;
        
//            img.load("dog.jpg");
        
//            ofVec2f wh = ofVec2f(img.getWidth(),
//                                 img.getHeight());
//            wh.normalize();
//            img.resize(wh.x*IMG_RESIZE_SIZE,
//                       wh.y*IMG_RESIZE_SIZE);

        
        generateSuperpixels(img, nSuperpixels, nWeightFactors);

    }
}

void ofApp::dragEvent(ofDragInfo info) {
    
//    img.load(info.files.at(0));
//    
//    ofVec2f wh = ofVec2f(img.getWidth(),
//                         img.getHeight());
//    wh.normalize();
//    img.resize(wh.x*IMG_RESIZE_SIZE,
//               wh.y*IMG_RESIZE_SIZE);
//    
//    generateSuperpixels(img, nSuperpixels, nWeightFactors);
    
}


/* Written by: Pascal Mettes.
 *
 * This file creates an over-segmentation of a provided image based on the SLIC
 * superpixel algorithm, as implemented in slic.h and slic.cpp. */
void ofApp::generateSuperpixels(ofPixels pix, int nr_superpixels, int nc) {
  
  
    /* Load the image and convert to Lab colour space. */
//    dogCvColorImg.setFromPixels(img.getPixels()); //, img.getWidth(), img.getHeight());
    //    IplImage *image = pix.getCvImage();
    //    IplImage *lab_image = cvCloneImage(image);
    //     image = toCv(pix);
    copy(pix,image);
    ofLog()<<"pix w "<<pix.getWidth()<<" h "<<pix.getHeight();
    if(pix.getWidth() > 0 && pix.getHeight() > 0){
        cvtColor(image, lab_image, CV_BGR2Lab);
        
        int w = image.cols, h = image.rows;
        //    ofLog()<<"w "<<w<<" h "<<h;
        
        double step = sqrt((w * h) / (double) nr_superpixels);
        //    ofLog()<<"step "<<step;
        
        /* Perform the SLIC superpixel algorithm. */
        
        slic.generate_superpixels(lab_image, step, nc);
        
        slic.generate_contours(w, h);
        
        //don't really need this
        //        slic.create_connectivity(lab_image);
        
        //    ofSetColor(255);
        slic.image_colorfulness(pix);
        
        
        /* Display the contours and show the result. */
        //    slic.display_contours(&image, cv::Vec3b(0,0,255));
        //    slic.display_center_grid(&image, cv::Vec3b(255,0,255));
        
        slic.colour_with_cluster_means(&image);
    }
    
}

