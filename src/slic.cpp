#include "slic.h"

/*
 * Constructor. Nothing is done here.
 */
Slic::Slic() {

}

/*
 * Destructor. Clear any present data.
 */
Slic::~Slic() {
    clear_data();
}

/*
 * Clear the data as saved by the algorithm.
 *
 * Input : -
 * Output: -
 */
void Slic::clear_data() {
    clusters.clear();
    distances.clear();
    centers.clear();
    center_counts.clear();
}

/*
 * Initialize the cluster centers and initial values of the pixel-wise cluster
 * assignment and distance values.
 *
 * Input : The image (IplImage*).
 * Output: -
 */
void Slic::init_data(cv::Mat &image) {
//    cout<<"---init_data---------------------- "<<endl;
    /* Initialize the cluster and distance matrices. */
    //    cv::Mat cr_mat = cv::Mat::zeros(image->height, image->width, CV_8UC1);
    
    for (int i = 0; i < image.cols; i++) {
        vector<int> cr;
        vector<double> dr;
        for (int j = 0; j < image.rows; j++) {
            cr.push_back(-1);
            dr.push_back(FLT_MAX);
        }
        clusters.push_back(cr);
        distances.push_back(dr);
    }
    
//    cout<<"step "<<step;
    
    /* Initialize the centers and counters. */
    for (int i = step; i < image.cols - step/2; i += step) {
        for (int j = step; j < image.rows - step/2; j += step) {
            vector<double> center;
            /* Find the local minimum (gradient-wise). */
//            cout<<"cv::Point(i,j) "<<i<<" x "<<j<<endl;
            cv::Point nc = find_local_minimum(image, cv::Point(i,j));
            cv::Vec3b colour = image.at<cv::Vec3b>(nc.y, nc.x);
         
            /* Generate the center vector. */
            center.push_back(colour.val[0]);
            center.push_back(colour.val[1]);
            center.push_back(colour.val[2]);
            center.push_back(nc.x);
            center.push_back(nc.y);
            
            /* Append to vector of centers. */
            centers.push_back(center);
            center_counts.push_back(0);
        }
    }
}

/*
 * Compute the distance between a cluster center and an individual pixel.
 *
 * Input : The cluster index (int), the pixel (CvPoint), and the Lab values of
 *         the pixel (CvScalar).
 * Output: The distance (double).
 */
double Slic::compute_dist(int ci, CvPoint pixel, CvScalar colour) {
    double dc = sqrt(pow(centers[ci][0] - colour.val[0], 2) + pow(centers[ci][1]
            - colour.val[1], 2) + pow(centers[ci][2] - colour.val[2], 2));
    double ds = sqrt(pow(centers[ci][3] - pixel.x, 2) + pow(centers[ci][4] - pixel.y, 2));
    
    return sqrt(pow(dc / nc, 2) + pow(ds / ns, 2));
    
    //double w = 1.0 / (pow(ns / nc, 2));
    //return sqrt(dc) + sqrt(ds * w);
}

/*
 * Find a local gradient minimum of a pixel in a 3x3 neighbourhood. This
 * method is called upon initialization of the cluster centers.
 *
 * Input : The image (IplImage*) and the pixel center (CvPoint).
 * Output: The local gradient minimum (CvPoint).
 */

cv::Point Slic::find_local_minimum(cv::Mat &image, cv::Point center) {
    double min_grad = FLT_MAX;
    cv::Point loc_min(center.x, center.y);
    
//    cout<<"image "<<image->cols<<" x "<<image->rows<<endl;
    for (int i = center.x-1; i < center.x+2; i++) {
        for (int j = center.y-1; j < center.y+2; j++) {
            cv::Vec3b c1 = image.at<cv::Vec3b>(j+1, i);
//            cout<<c1<<" i "<<i<<" j+1 "<<(j+1)<<endl;
            
            cv::Vec3b c2 = image.at<cv::Vec3b>(j, i+1);
//            cout<<c2<<" i+1 "<<(i+1)<<" j "<<(j)<<endl;
            
            cv::Vec3b c3 = image.at<cv::Vec3b>(j, i);
//            cout<<c3<<" i "<<i<<" j "<<(j)<<endl;
            
            /* Convert colour values to grayscale values. */
            double i1 = c1.val[0];
            double i2 = c2.val[0];
            double i3 = c3.val[0];
            /*double i1 = c1.val[0] * 0.11 + c1.val[1] * 0.59 + c1.val[2] * 0.3;
             double i2 = c2.val[0] * 0.11 + c2.val[1] * 0.59 + c2.val[2] * 0.3;
             double i3 = c3.val[0] * 0.11 + c3.val[1] * 0.59 + c3.val[2] * 0.3;*/
            
            /* Compute horizontal and vertical gradients and keep track of the
             minimum. */
            if (sqrt(pow(i1 - i3, 2)) + sqrt(pow(i2 - i3,2)) < min_grad) {
                min_grad = fabs(i1 - i3) + fabs(i2 - i3);
                loc_min.x = i;
                loc_min.y = j;
            }
        }
    }
    
    return loc_min;
}

/*
 * Compute the over-segmentation based on the step-size and relative weighting
 * of the pixel and colour values.
 *
 * Input : The Lab image (IplImage*), the stepsize (int), and the weight (int).
 * Output: -
 
 the result is a 2d array the size of the original image, that contains a index number IDing to which cluster the pixel belongs
 */
void Slic::generate_superpixels(cv::Mat &image, int step, int nc) {
    this->step = step;
    this->nc = nc;
    this->ns = step;
    
    /* Clear previous data (if any), and re-initialize it. */
    clear_data();
    init_data(image);
    
    /* Run EM for 10 iterations (as prescribed by the algorithm). */
    for (int i = 0; i < NR_ITERATIONS; i++) {
        /* Reset distance values. */
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0;k < image.rows; k++) {
                distances[j][k] = FLT_MAX;
            }
        }
        
        for (int j = 0; j < (int) centers.size(); j++) {
            /* Only compare to pixels in a 2 x step by 2 x step region. */
            for (int k = centers[j][3] - step; k < centers[j][3] + step; k++) {
                for (int l = centers[j][4] - step; l < centers[j][4] + step; l++) {
                    
                    if (k >= 0 && k < image.cols && l >= 0 && l < image.rows) {
                        cv::Vec3b colour = image.at<cv::Vec3b>(l, k);
//                        double d = compute_dist(j, cvPoint(k,l), colour);
                        double d = compute_dist(j, cv::Point(k,l), colour);
                        /* Update cluster allocation if the cluster minimizes the
                         distance. */
                        if (d < distances[k][l]) {
                            distances[k][l] = d;
                            clusters[k][l] = j;
                            //                            clusters_mats[k][l] = j;
                        }
                    }
                }
            }
        }
        
        /* Clear the center values. */
        for (int j = 0; j < (int) centers.size(); j++) {
            centers[j][0] = centers[j][1] = centers[j][2] = centers[j][3] = centers[j][4] = 0;
            center_counts[j] = 0;
        }
        
        /* Compute the new cluster centers. */
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < image.rows; k++) {
                int c_id = clusters[j][k];
                
                if (c_id != -1) {
//                    cout<<c_id<<" k "<<k<<" j "<<j<<endl;
                    cv::Vec3b colour = image.at<cv::Vec3b>(k, j);
//                    cout<<"  colour "<<colour<<endl;
                    centers[c_id][0] += colour.val[0];
                    centers[c_id][1] += colour.val[1];
                    centers[c_id][2] += colour.val[2];
                    centers[c_id][3] += j;
                    centers[c_id][4] += k;
                    
                    center_counts[c_id] += 1;
                }
            }
        }
        
        /* Normalize the clusters. */
        for (int j = 0; j < (int) centers.size(); j++) {
//            cout<<"centers[i][0] "<<centers[j][0]<<" center_counts[i] "<<center_counts[j]<<endl;

            centers[j][0] /= center_counts[j];
            centers[j][1] /= center_counts[j];
            centers[j][2] /= center_counts[j];
            centers[j][3] /= center_counts[j];
            centers[j][4] /= center_counts[j];
        }
    }
}

/*
 * Enforce connectivity of the superpixels. This part is not actively discussed
 * in the paper, but forms an active part of the implementation of the authors
 * of the paper.
 *
 * Input : The image (IplImage*).
 * Output: -
 */
//https://groups.google.com/forum/#!topic/scikit-image/Ry7jCoT8Uys
void Slic::create_connectivity(IplImage *image) {
    int label = 0, adjlabel = 0;
    const int lims = (image->width * image->height) / ((int)centers.size());
    
    const int dx4[4] = {-1,  0,  1,  0};
	const int dy4[4] = { 0, -1,  0,  1};
    
    /* Initialize the new cluster matrix. */
    vec2di new_clusters;
    for (int i = 0; i < image->width; i++) { 
        vector<int> nc;
        for (int j = 0; j < image->height; j++) {
            nc.push_back(-1);
        }
        new_clusters.push_back(nc);
    }

    for (int i = 0; i < image->width; i++) {
        for (int j = 0; j < image->height; j++) {
            if (new_clusters[i][j] == -1) {
                vector<CvPoint> elements;
                elements.push_back(cvPoint(i, j));
            
                /* Find an adjacent label, for possible use later. */
                for (int k = 0; k < 4; k++) {
                    int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];
                    
                    if (x >= 0 && x < image->width && y >= 0 && y < image->height) {
                        if (new_clusters[x][y] >= 0) {
                            adjlabel = new_clusters[x][y];
                        }
                    }
                }
                
                int count = 1;
                for (int c = 0; c < count; c++) {
                    for (int k = 0; k < 4; k++) {
                        int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];
                        
                        if (x >= 0 && x < image->width && y >= 0 && y < image->height) {
                            if (new_clusters[x][y] == -1 && clusters[i][j] == clusters[x][y]) {
                                elements.push_back(cvPoint(x, y));
                                new_clusters[x][y] = label;
                                count += 1;
                            }
                        }
                    }
                }
                
                /* Use the earlier found adjacent label if a segment size is
                   smaller than a limit. */
                if (count <= lims >> 2) {
                    for (int c = 0; c < count; c++) {
                        new_clusters[elements[c].x][elements[c].y] = adjlabel;
                    }
                    label -= 1;
                }
                label += 1;
            }
        }
    }
}

/*
 * Display the cluster centers.
 *
 * Input : The image to display upon (IplImage*) and the colour (CvScalar).
 * Output: -
 */
void Slic::display_center_grid(cv::Mat *image, cv::Vec3b colour) {
    for (int i = 0; i < (int) centers.size(); i++) {
//        cvCircle(image, cvPoint(centers[i][3], centers[i][4]), 2, colour, 2);
        cv::circle(*image, cvPoint(centers[i][3], centers[i][4]), 2, cv::Scalar(colour.val[0],colour.val[1],colour.val[2]), 2);
    }
}

/*
 * Display a single pixel wide contour around the clusters.
 *
 * Input : The target image (IplImage*) and contour colour (CvScalar).
 * Output: -
 */

void Slic::display_contours(cv::Mat *image, cv::Vec3b colour) { //cv::Scalar colour){ //
    const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
    
    /* Initialize the contour vector and the matrix detailing whether a pixel
     * is already taken to be a contour. */
    vector<cv::Point> contours;
    vec2db istaken;
    for (int i = 0; i < image->cols; i++) {
        vector<bool> nb;
        for (int j = 0; j < image->rows; j++) {
            nb.push_back(false);
        }
        istaken.push_back(nb);
    }
    
    /* Go through all the pixels. */
    for (int i = 0; i < image->cols; i++) {
        for (int j = 0; j < image->rows; j++) {
            int nr_p = 0;
            
            /* Compare the pixel to its 8 neighbours. */
            for (int k = 0; k < 8; k++) {
                int x = i + dx8[k], y = j + dy8[k];
                
                if (x >= 0 && x < image->cols && y >= 0 && y < image->rows) {
                    if (istaken[x][y] == false && clusters[i][j] != clusters[x][y]) {
                        nr_p += 1;
//                        cout<<"nr_p "<<nr_p;
                    }
                }
            }
            
            /* Add the pixel to the contour list if desired. */
            if (nr_p >= 2) {
                contours.push_back(cv::Point(i,j));
                istaken[i][j] = true;
            }
        }
    }
  
    /* Draw the contour pixels. */
    for (int i = 0; i < (int)contours.size(); i++) {
        cv::Vec3b colour1(255,1,1);
        image->at<cv::Vec3b>(contours[i]) = colour1; //colour;
    }
    
}


void Slic::generate_contours(int width, int height) { //cv::Scalar colour){ //
    const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
    
    /* Initialize the contour vector and the matrix detailing whether a pixel
     * is already taken to be a contour. */
   
    contours2.clear();
    
    vec2db istaken;
    for (int i = 0; i < width; i++) {
        vector<bool> nb;
        for (int j = 0; j < height; j++) {
            nb.push_back(false);
        }
        istaken.push_back(nb);
    }
    
    /* Go through all the pixels. */
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int nr_p = 0;
            
            /* Compare the pixel to its 8 neighbours. */
            for (int k = 0; k < 8; k++) {
                int x = i + dx8[k], y = j + dy8[k];
                
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    if (istaken[x][y] == false && clusters[i][j] != clusters[x][y]) {
                        nr_p += 1;
                        //                        cout<<"nr_p "<<nr_p;
                    }
                }
            }
            
            /* Add the pixel to the contour list if desired. */
            if (nr_p >= 2) {
                contours2.push_back(ofPoint(i,j));
                istaken[i][j] = true;
            }
        }
    }
  
}
void Slic::display_contours(ofColor colour) { //cv::Scalar colour){ //

    /* Draw the contour pixels. */
    ofSetColor(colour);
    for (int i = 0; i < (int)contours2.size(); i++) {
        ofDrawCircle(contours2[i], 1);
//        image->at<cv::Vec3b>(contours[i]) = colour1; //colour;
    }
    
}


/*
 * Give the pixels of each cluster the same colour values. The specified colour
 * is the mean RGB colour per cluster.
 *
 * Input : The target image (IplImage*).
 * Output: -
 */

void Slic::colour_with_cluster_means(cv::Mat *image) {
//    vector<cv::Vec3b> colours(centers.size());
    vector<cv::Scalar> colours(centers.size());
//    cout<<"centers.size() "<<centers.size();
    
    /* Gather the colour values per cluster. */
    for (int i = 0; i < image->cols; i++) {
        for (int j = 0; j < image->rows; j++) {
            int index = clusters[i][j];
        
            //            cout<<"index "<<index;
//            CvScalar colour = cvGet2D(image, j, i);
            cv::Vec3b colour = image->at<cv::Vec3b>(j,i);
//            cv::Scalar = image->at
//            cout<<"colour "<<colour<<endl;
            colours[index].val[0] += colour.val[0];
            colours[index].val[1] += colour.val[1];
            colours[index].val[2] += colour.val[2];
        }
    }
   
    /* Divide by the number of pixels per cluster to get the mean colour. */
    for (int i = 0; i < (int)colours.size(); i++) {
//        cout<<"colours[i].val[0] "<<colours[i]<<" center_counts[i] "<<center_counts[i]<<endl;
        colours[i].val[0] /= center_counts[i];
        colours[i].val[1] /= center_counts[i];
        colours[i].val[2] /= center_counts[i];
    }
    
    /* Fill in. */
    for (int i = 0; i < image->cols; i++) {
        for (int j = 0; j < image->rows; j++) {
//            cv::Vec3b ncolour = colours[clusters[i][j]];
              cv::Scalar ncolour = colours[clusters[i][j]];
//            cout<<"ncolour "<<ncolour;
//            cvSet2D(image, j, i, ncolour);
//            image->at<cv::Vec3b>(i,j) = ncolour;
             image->at<cv::Vec3b>(j,i)[0] = ncolour.val[0];
              image->at<cv::Vec3b>(j,i)[1] = ncolour.val[1];
              image->at<cv::Vec3b>(j,i)[2] = ncolour.val[2];
        }
    }
}

//compare all clusters to their colorfulness
//original code from here:
//http://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
//ofPoint Slic::image_colorfulness(cv::Mat *image){
ofPoint Slic::image_colorfulness(ofPixels &pix){
    
    ofLog()<<"-----image_colorfulness------";
    
    allPixelGroups.clear();
    allPixelGroups.resize(centers.size());
    
    ofLog()<<"allPixelGroups.size() "<<allPixelGroups.size();
    
//     ofColor temp_color;
//    cv::Vec3b colour;
    ofColor colour;
    
    //create one object per pixel group/cluster
    //collect all RGB values for this group
    for (int i = 0; i < pix.getWidth(); i++) {
        for (int j = 0; j < pix.getHeight(); j++) {
            int index = clusters[i][j];
            
//             colour = image->at<cv::Vec3b>(j,i);
            colour = pix.getColor(i,j);
//            ofLog()<<"colour.val[0] "<<float(colour.r);
//              temp_color = src.getColor(i);
            allPixelGroups[index].R.push_back(colour.r);
            allPixelGroups[index].G.push_back(colour.g);
            allPixelGroups[index].B.push_back(colour.b);
            allPixelGroups[index].points.push_back(ofPoint(i,j));
        }
    }
    
   
    for (int i = 0; i < (int) centers.size(); i++) {
        allPixelGroups[i].group_index = i;
        allPixelGroups[i].centerPoint = ofPoint(centers[i][3],centers[i][4]);
        allPixelGroups[i].centerColor = ofColor(centers[i][0],centers[i][1],centers[i][2]);
    }
    
//    vector<float> rg;
//    vector<float> yb;
    
    for (auto group = allPixelGroups.begin(); group != allPixelGroups.end(); ++group)
    {

        (*group).calcSize();
        
//        ofLog()<<(*group).group_index<<" (*group).arraySize "<<(*group).arraySize;
        
        //rg -> Red-Green opponent
        //rg = np.absolute(R - G)
         vector<float> rg;
//        rg.clear();
        for(int i=0; i<(*group).arraySize; i++){
            float value = (*group).R[i] - (*group).G[i];
//             ofLog()<<"value "<<value;
            value = ABS(value);
            rg.push_back(value);
        }
        
//        ofLog()<<"rg.size "<<rg.size();
        
        //Yellow-Blue opponent
        //yb = np.absolute(0.5 * (R + G) - B)
        vector<float> yb;
//        yb.clear();
        for(int i=0; i<(*group).arraySize; i++){
            float value = 0.5* ((*group).R[i] + (*group).G[i]) - (*group).B[i];
            value = ABS(value);
            yb.push_back(value);
        }
        
//        ofLog()<<"yb.size "<<yb.size();
        
        //compute the mean and standard deviation of both `rg` and `yb`
        //(rbMean, rbStd) = (np.mean(rg), np.std(rg))
        //(ybMean, ybStd) = (np.mean(yb), np.std(yb))
        //https://www.programiz.com/cpp-programming/examples/standard-deviation
        
        float rg_sum = 0;
        float yb_sum = 0;
        float rg_mean = 0;
        float yb_mean = 0;
        float rg_std = 0;
        float yb_std = 0;
        
        for(int i=0; i<(*group).arraySize; i++){
            rg_sum += rg[i];
            yb_sum += yb[i];
        }
        
//        ofLog()<<"\t \t rg_sum "<<rg_sum<<" , yb_mean "<<yb_mean;

        
        rg_mean = rg_sum / (*group).arraySize;
        yb_mean = yb_sum / (*group).arraySize;
        
//        ofLog()<<"\t \t rg_sum "<<rg_sum<<" , yb_mean "<<yb_mean;
        
        for(int i=0; i<(*group).arraySize; i++){
            rg_std += pow(rg[i] - rg_mean, 2);
            yb_std += pow(yb[i] - yb_mean, 2);
        }
        
        rg_std = sqrt(rg_std / (*group).arraySize);
        yb_std = sqrt(yb_std / (*group).arraySize);
        
        
        //# combine the mean and standard deviations
        //    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        //    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        
        float stdRoot = sqrt(pow(rg_std,2) + pow(yb_std,2));
        float meanRoot = sqrt(pow(rg_mean,2) + pow(yb_mean,2));
        
        //# derive the "colorfulness" metric and return it
        (*group).colorfulness = stdRoot + (0.3 * meanRoot);
    }
    
    /*
     //find group with max colorfulness
     float temp_max = -1;
     int temp_idx = -1;
     for (auto group = allPixelGroups.begin(); group != allPixelGroups.end(); ++group)
     {
     if((*group).colorfulness > temp_max){
     temp_max = (*group).colorfulness;
     temp_idx = (*group).group_index;
     }
     }
     
     allPixelGroups[temp_idx].mostColorfull = true;
     return allPixelGroups[temp_idx].centerPoint;
     */
    
    //sort array from most colorfull to least
    //https://stackoverflow.com/questions/1380463/sorting-a-vector-of-custom-objects
    sort(allPixelGroups.begin(),allPixelGroups.end(),greater<onePixelGroup>());
    
    allPixelGroups[0].mostColorfull = true;
    return allPixelGroups[0].centerPoint;
}

void Slic::drawPixelGroupsImage(int _amt){
    
    ofPushStyle();
    
    if(_amt == 0 || _amt > allPixelGroups.size()){
        _amt = allPixelGroups.size();
    }
//    for (auto group = allPixelGroups.begin(); group != allPixelGroups.end(); ++group)
    for (auto group = allPixelGroups.begin(); group != allPixelGroups.begin()+_amt; ++group)
    {
        for(int i=0; i<(*group).points.size();i++){
            ofSetColor((*group).R[i], (*group).G[i], (*group).B[i]);
            ofDrawCircle((*group).points[i], 1);
        }
        
    }
    
    ofPopStyle();
}

void Slic::drawPixelGroups(){
    
    ofPushStyle();
    
    
    for (auto group = allPixelGroups.begin(); group != allPixelGroups.end(); ++group)
    {
        if((*group).mostColorfull){
            ofFill();
            ofSetColor(255);
            ofDrawCircle((*group).centerPoint, 7);
        }
        
        ofNoFill();
        ofSetColor(255, 0, 0);
        ofDrawCircle((*group).centerPoint, 3);
        
         ofSetColor(255);
        ofDrawBitmapString(ofToString((*group).colorfulness,0), (*group).centerPoint);
    }
    ofPopStyle();
}