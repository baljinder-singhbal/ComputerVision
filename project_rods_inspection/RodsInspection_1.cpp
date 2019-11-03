//============================================================================
// Name        : RodsInspection_1.cpp
// Author      : Baljinder Singh Bal
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
//OpenCV
#include <opencv2/opencv.hpp>

using namespace cv;
//C++
#include <iostream>
using namespace std;



//MORPHOLOGICAL OPERATIONS
Mat erosion(Mat inputImage, int strucElement_type, int structElement_size);
Mat dilation(Mat inputImage, int strucElement_type, int structElement_size);
Mat closing(Mat inputImage, int strucElement_type, int structElement_size);
Mat opening(Mat inputImage, int strucElement_type, int structElement_size);

//GENERAL FUNCTIONS

//THRESHOLD THE IMAGE
Mat Thresh(Mat image);

//CLASSIFY THE RODS
struct features extractFeutures(Mat inputImage,vector <vector<Point2i> > contours, vector<Vec4i> hierarchy);

//SHOW THE RESULT
Mat showContours( Mat colorImage,vector< vector<Point2i> > contours,vector<Vec4i> hierarchy);

//COMPUTE THE ORIENTATION
vector<double> computeOrientation(vector <vector<Point2i> > contours,vector<int> rodA);

//COMPUTE BARYCENTERS
vector<Point2f> computeBarycenters(vector <vector<Point2i> > contours,vector<int> rod);

//SHOW AXIS OF THE FOUND RODS
Mat showAxis(Mat inputImage, vector<int> rod,vector<vector<Point2i> > contours);

//COMPUTE THE EXACT WIDTH AT THE BARYCENTER
struct WidthAndLenght_AtBarycenter hotellingTrans(Mat inputImage, vector<vector<Point2i> > contours, vector<int> rod);

void DisplayTableResults(Mat inputImage,vector< vector<Point2i> > contours,vector<Vec4i> hierarchy);
//TWO ALTERNATIVE FUNCTIONS TO DEAL WITH THE TOUCHING RODS CASE
struct touchingRods_contours touchingRods(Mat inputImage);
struct touchingRods_contours separateUsingCorners(Mat inputImage);

//THREE MORE FUNCTIONS THAT DOESN'T WORK
Mat hugh(Mat inputImage);
Mat convexityD(Mat inputImage);
Mat findRightContours(Mat inputImage);//TENTATIVE UDE OF THE WATERSHED ALGORITHM

//random number needed for displaying the contours as random colors
RNG rng(12345);

//STRUCTS FOR PASSING THE FOUND INFORMATIONS BETWEEN FUNCTIONS
struct touchingRods_contours {
		//essential points of the touching contours
		vector<vector<Point2i > > touchingRodsA_contours;
		vector<vector<Point2i > > touchingRodsB_contours;
};
struct features{

		//all the contours classified as holes with their center and radius
		vector<int> holes_contours;
		vector<Point2f> holesCenters;
		vector<float> holesRadius;
		//all contours classified as rod type A and type B with their min enclosing rectangle
		vector<int> rodA_contours;
		vector<int> rodB_contours;
		vector<RotatedRect> minRectRodA;
		vector<RotatedRect> minRectRodB;
		vector<vector<Point > > touchingRodsA_contours;
    	vector<vector<Point > > touchingRodsB_contours;
};

struct WidthAndLenght_AtBarycenter{
	vector<double> exactWidth;
	vector<double> exactLenght;
};

/*
 * MAIN FUNCTION
 */
int main() {

		//IN THIS FIRST PART OF THE CODE WE CAN CHOOSE WITCH IMAGE TO LOAD
		cout<<"chose the image to load:"<<endl;
		cout<<"press 1 for the first task images\npress 2 for the second task images"<<endl;
		int task;
		cin>>task;

		char filename[150]="/home/.../RODS_IMAGES/"; //insert here the right path to the folder with the images
		char imageName[11];
		if(task==1){
				cout<<"\nFirst task Image list: "<<endl;
				cout<<"\nTesi33.bmp\nTESI00.BMP\nTESI01.BMP\nTESI12.BMP\nTESI21.BMP\nTESI31.BMP";
				cout<<"\nType the image name: \n"<<endl;
				cin>>imageName;
				char task_folder[20]="first task/";
				strcat(filename,task_folder);
				strcat(filename,imageName);

		}
		if(task==2){
				cout<<"\nSecond task Image list:";
				cout<<"\nPRESENCE OF DISTRACTORS:\nTESI44.BMP\nTESI47.BMP\nTESI48.BMP\nTESI49.BMP\n\n"
						"TOUCHING RODS:\nTESI50.BMP\nTESI51.BMP\n\n"
						"IRON POWDER IN THE INSPECTION AREA:\nTESI90.BMP\nTESI92.BMP\nTESI98.BMP\n\n";
				cout<<"Type the image name:\n"<<endl;
				cin>>imageName;
				char task_folder[20]="second task/";
				strcat(filename,task_folder);
				strcat(filename,imageName);

		}
		//LOAD THE IMAGE
		Mat image=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);

		if (image.empty()){
				cout<<"Could not open the image"<<endl;
				return -1;
		}

		//apply  a median filter to cancel out some of the dust
		Mat median;
		medianBlur(image, median,3);

		//threshold the image
		Mat binary=Thresh(median);


		//the color image is passed to show contours with different colors
		Mat color;
		cvtColor(image,color,CV_GRAY2BGR);

		//after the binarization process, find the contours of the connected components
		vector< vector<Point2i> > contours;
		vector<Vec4i> hierarchy;
		findContours(binary, contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

		//perform the classification and show the result as a table
		DisplayTableResults( image, contours, hierarchy);


		//show and classify the contours and show them in a window
		Mat  contours1=showContours(color,contours,hierarchy);
        imshow("Otsu", contours1);

        //uncomment to save the result image
        /*
        vector<int> compression_params;
       	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
       	compression_params.push_back(9);
       	imwrite("/home/baljinder/University/computer vision and image processing/project:rods inspection/binary.png",binary,compression_params);
		*/

        //try with watershed algorithm
		//Mat right=findRightContours(color);

		waitKey(0);
		return 0;
}


/*
 * THRESHOLD THE IMAGE
 */
Mat Thresh(Mat image){


		int thresholdType = THRESH_BINARY;//TYPE OF VISUALIZATION
		double maxThresh = 255;
		double thresh = 117;
		int blockSize = 7;//SIZE OF THE NEIGHBORHOOD FOR ADAPTIVE THRESHOLDING
		double C = 3;//CONSTANT TO BE SUCTRACTED TO THE ADAPTIVE THRESHOLD
		//adaptive thresholds : gaussian and mean
		int type = 2;
		//UNCOMMENT TO CHOOSE FROM THE COMMAND LINE THE TYPE OF THE THRESHOLD
		/*
		cout<< "Press:0 for Adaptive threshold with gaussian\n";
		cout<< "      1 for Adaptive threshold with mean\n";
		cout<< "      2 for Threshold with Otsu\n";
		cout<< "      3 for Threshold with Triangle methods\n";

		cin >> type;
	    */

		Mat binary; Mat labels;
		if(type==0){
				adaptiveThreshold(image,binary,maxThresh,ADAPTIVE_THRESH_GAUSSIAN_C,thresholdType, blockSize, C);

				//finding the connected components
				int n_obj=connectedComponents(binary, labels,8,CV_32S);
				//cout<<"number of detected object with adaptive gaussina\n"<<n_obj<<endl;
		}
		if (type==1){
				adaptiveThreshold(image,binary,maxThresh,ADAPTIVE_THRESH_MEAN_C,thresholdType, blockSize, C);

				//finding the connected components
				int n_obj=connectedComponents(binary, labels,8,CV_32S);
				//cout<<"number of detected object with adaptive mean: "<<n_obj <<endl;
		}
		if (type==2){
				int t=threshold(image,binary,thresh,maxThresh, THRESH_OTSU);

				//finding the connected components
				int n_obj=connectedComponents(binary, labels,8,CV_32S);
				//cout<<"number of detected object with otsu: "<<n_obj <<endl;
				//cout<<"threshold value with otsu: "<<t <<endl;
		}

		if (type==3){
				threshold(image,binary,thresh,maxThresh,THRESH_TRIANGLE);

				//finding the connected components
				//int n_obj=connectedComponents(binary, labels,8,CV_32S);
				//cout<<"number of detected object with otsu: "<<n_obj <<endl;
		}


		return binary;
}


/*
 * SHOW RESULT OF THE INSPECTION
 */
Mat showContours( Mat colorImage,vector< vector<Point2i> > contours,vector<Vec4i> hierarchy ){


		if (contours.empty()){
				cout<<"contour vector empty"<<endl;
				return Mat::zeros(colorImage.size(),CV_8UC1);;
		}


		//draw the contours with different random colors
		Mat temp =colorImage.clone();

		//extract the features
		features F=extractFeutures(colorImage,contours,hierarchy);

		//LOOK INSIDE F TO SEE HOW MANY HOLES, RODS AND OTHER FEAUTURES HAVE BEEN DETECTED
		/*
    	cout<<"table "<<endl;
 		cout<<" H :centers,radius,contours"<<endl;
 		cout<<F.holesCenters.size()<<endl;
 		cout<<F.holesRadius.size()<<endl;
		cout<<F.holes_contours.size()<<endl;
		cout<<" R :rectA,rectB,contoursA, contoursB"<<endl;
		cout<<F.minRectRodA.size()<<endl;
		cout<<F.minRectRodB.size()<<endl;
		cout<<F.rodA_contours.size()<<endl;
		cout<<F.rodB_contours.size()<<endl;
		cout<<" R: touching A,B"<<endl;
		cout<<F.touchingRodsA_contours.size()<<endl;
		cout<<F.touchingRodsB_contours.size()<<endl;
 	 	 */

		Mat A,B,C,output;



		if(F.rodA_contours.empty()==false || F.rodB_contours.empty()==false){

				//show the approximated major and minor axis of the found rods
				A= showAxis(colorImage, F.rodA_contours, contours);
				B= showAxis(A,F.rodB_contours,contours);


				//show the exact lenght axis
				//imshow(" axis A and B",output);
		}

		if(F.touchingRodsB_contours.empty()==false || F.touchingRodsB_contours.empty()==false){

				vector<int> touching_rodsA(F.touchingRodsA_contours.size());
				for(int u=0;u<F.touchingRodsA_contours.size();u++){
							touching_rodsA[u]=u;
				}
				if(B.empty()==true){
					output=colorImage;
				}
				if(B.empty()==false){
					output=B;
				}
				C=showAxis( output, touching_rodsA,  F.touchingRodsA_contours );

				vector<int> touching_rodsB(F.touchingRodsB_contours.size());
				for(int c=0;c<F.touchingRodsB_contours.size();c++){
							touching_rodsB[c]=c;
				}
			    output=showAxis( C,touching_rodsB , F.touchingRodsB_contours );
		}

		if(output.empty()==true){
			output=B;
		}


		//show ALL the contours with random colors
		for( int i = 0; i< contours.size(); i++ ){

				//uncomment to look at the hierarchy vector
		        //cout<<"hierarchy[]: "<<hierarchy[i][0]<<" "<<hierarchy[i][1]<<" "<<hierarchy[i][2]<<" "<<hierarchy[i][3]<<endl;

				Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
				drawContours( output, contours, i, color, 2 , 8);
		}




		//now it's time to draw the bounding boxes and circles
		//first we draw the A type rods enclosed in minimum area rectangles
		Point2f vertices[4];
		for(int d=0;d<F.rodA_contours.size();d++){
				F.minRectRodA[d].points(vertices);
				putText(output,"ROD A",F.minRectRodA[d].center,CV_FONT_HERSHEY_PLAIN,1.0,Scalar(250,0,0),1,8,false);
				for (int w=0;w<4;w++){
						Scalar color= Scalar( 250,0,0);
						line(output,vertices[w], vertices[(w+1)%4], color, 1);
				}
		}
		//now the B type rods enclosed in minimum area rectangles
		for(int q=0;q<F.rodB_contours.size();q++){
				F.minRectRodB[q].points(vertices);
				putText(output,"ROD B",F.minRectRodB[q].center,CV_FONT_HERSHEY_PLAIN,1.0,Scalar(0,250,0),1,8,false);
				for (int z=0;z<4;z++){
						Scalar color= Scalar(0,250,0);// black rectangle
						line(output,vertices[z], vertices[(z+1)%4], color, 1);
				}
		}
		//and now hole in the min area enclosing circles
		for(int s=0;s<F.holes_contours.size();s++){
				Scalar color= Scalar( 0,0,0);
				circle(output, (Point_<int>)F.holesCenters[s], (int)F.holesRadius[s], color,1);
		}

		//Check if any touching roda have been detected and eventually draw  minimum area enclosing rectangles
		if(F.touchingRodsA_contours.empty()==false | F.touchingRodsB_contours.empty()==false){

				vector<vector<Point> > erodedRodsA_contours(F.touchingRodsA_contours.size());
				vector<vector<Point> > erodedRodsB_contours(F.touchingRodsB_contours.size());

				for(int d=0;d<F.touchingRodsA_contours.size();d++){

						RotatedRect rect=minAreaRect(F.touchingRodsA_contours[d]);
						rect.points(vertices);
						putText(output,"ROD A",rect.center,CV_FONT_HERSHEY_PLAIN,1.0,Scalar(125,0,0),1,8,false);
						for(int g=0;g<4;g++){
								line(output,vertices[g],vertices[(g+1)%4],Scalar(125.0,0),2);
						}
				}
				for(int s=0;s<F.touchingRodsB_contours.size();s++){

						RotatedRect rect=minAreaRect(F.touchingRodsB_contours[s]);
						rect.points(vertices);
						putText(output,"ROD B",rect.center,CV_FONT_HERSHEY_PLAIN,1.0,Scalar(0,125,0),1,8,false);
						for(int h=0;h<4;h++){
								line(output,vertices[h],vertices[(h+1)%4],Scalar(0,125,0),2);
						}
				}
		}
		//check if it's all right up to now
		if (output.empty()==true){
				cout << "error:showContour output empty"<<endl;
				return Mat::zeros(colorImage.size(),CV_8UC1);
		}
		return output;
}


/*
 * CLASSIFY THE RODS
 */
struct features extractFeutures(Mat inputImage,vector <vector<Point2i> > contours, vector<Vec4i> hierarchy){

		if(contours.empty()==true){
				cout<<"ERROR: the contours vector supplied to extracrFeutures is empty"<<endl;
		}
		//find and store the rod's holes
		Moments contour_moments;
		vector<int> holes;
		for (int i=0;i<contours.size();i++){

				//check if it has no hole and if its parent is not the first element in the hierarchy vector(to exclude from analisys the objects that have no hole(that are the screw)
				if(hierarchy[i][2]==-1  && hierarchy[i][3]!=0 ){
						//check the area of the hole
						contour_moments=moments(contours[i],true);
						if(contour_moments.m00>50){
								//check the rectangularity of its father(outer contour)
								float rect_holes;
								RotatedRect r=minAreaRect(contours[hierarchy[i][3]]);
								if ((r.size.height/r.size.width)<=1){
										rect_holes=r.size.height/r.size.width;
								}
								else{
										rect_holes=r.size.width/r.size.height;
								}
								// check
								//cout<<"rectangularity "<<rect_holes;
								if(rect_holes<0.75){
										holes.push_back(i);
								}
						}
				}

		}
		//find and store all the rods
		//now  i use the fact that each parent(external contour) has at least one child(hole) and the fact that rodA has two holes and rodB has one to classify the two objects
		vector<int> rodA;
		vector<int> rodB;
		struct touchingRods_contours touchingContours;
		for(int j=0;j<contours.size();j++){
				//if it has at least one child
				if(hierarchy[j][2]!=-1){

						//we can check the area of the contour to see if it's not to small(dust)
						contour_moments=moments(contours[j], true);
						//cout<<"contour "<<j<<" area "<<contour_moments.m00<<endl;
						if(contour_moments.m00>800 && contour_moments.m00<7000 ){

								//check if it's not the outer contours(so it must have some same level contour)
								if(hierarchy[j][0]!=-1 || hierarchy[j][1]!=-1 ){

										//check the contours rectangularity to cancel out the non rod objects with holes
										RotatedRect s=minAreaRect(contours[j]);
										float rect_ext;
										if ((s.size.height/s.size.width)<=1){
												rect_ext=s.size.height/s.size.width;
										}
										else{
												rect_ext=s.size.width/s.size.height;
										}
										// check
										//cout<<"rectangularity "<<rect_ext;
										if(rect_ext<0.75){
												//if its child has a sibling(contour of the same level) NOTE: here it is needed to check only the next contour on the same level
												if(hierarchy[hierarchy[j][2]][0]!=-1){
														//store the contour as a rod of type A
														rodA.push_back(j);

												}
												//if its child has no siblings
												else{
														//put it in the list of rod B type objects
													rodB.push_back(j);
												}
										}
								}
						}
						//for touching contours scenario
						else if(contour_moments.m00>7000 && contour_moments.m00<40000 && hierarchy[hierarchy[j][2]][0]!=-1 ){
								//cout<<"found touching contours "<<contour_moments.m00<<endl;

								Mat BigContour=Mat::zeros(inputImage.size(),CV_8UC3);
								drawContours(BigContour,contours,j,Scalar(125,125,125), CV_FILLED,8,hierarchy);
								//uncomment to see the touching rods contour
								//imshow("bigContour",BigContour);

								/*separate the touching rods by using erosion and distance transform:
							  	  the result it's less precise
							  	  than the Corner option but could be more robust
							 	 (unfortunatly i don't have enough touching rods images
							 	 to check different scenarios)
								 */

								//uncomment if you want to erode the big contour objects to give
								//an approximated position and type of rods
								//touchingContours=touchingRods(BigContour);

								//alternatevally we can use harris corners to have a better separation
								//of the touching rods
								touchingContours=separateUsingCorners(BigContour);


						}

				}
		}

		vector<RotatedRect> rectRodA;
		vector<RotatedRect> rectRodB;
		vector<RotatedRect> rect_touchingRodsA;
		vector<RotatedRect> rect_touchingRodsB;
		//bounding boxes for the holes
		vector<Point2f> holeCenters(holes.size());
		vector<float> holeRadius(holes.size());
		for (int k=0;k<holes.size();k++){
				minEnclosingCircle(contours[holes[k]], holeCenters[k],holeRadius[k]);
		}
		//bounding boxes for the rods
		//rodA
		for (int s=0;s<rodA.size();s++){
				rectRodA.push_back(minAreaRect(contours[rodA[s]]));
		}


		//for touching rodsA
		if(touchingContours.touchingRodsA_contours.empty()==false){

				for (int a=0;a<touchingContours.touchingRodsA_contours.size();a++){
						rect_touchingRodsA.push_back(minAreaRect(touchingContours.touchingRodsA_contours[a]));
				}
		}
		//rodB
		for (int t=0;t<rodB.size();t++){
				rectRodB.push_back(minAreaRect(contours[rodB[t]]));
		}

		//for touching rodsB

		if(touchingContours.touchingRodsB_contours.empty()==false){
				for (int v=0;v<touchingContours.touchingRodsB_contours.size();v++){
				rect_touchingRodsB.push_back(minAreaRect(touchingContours.touchingRodsB_contours[v]));
				}
		}
		//cout<<" rect touching B "<<rect_touchingRodsB.size()<<endl;




		//return the found feutures
		features f;
		f= {holes,holeCenters,holeRadius,rodA,rodB,rectRodA,rectRodB, touchingContours.touchingRodsA_contours,touchingContours.touchingRodsB_contours};



		return f ;

}


/*
 * MORPHOLOGICAL OPERATIONS
 */
Mat opening(Mat inputImage, int strucElement_type, int structElement_size){
		Mat output,temp;
		int elemType;
		// choose witch structuring element to apply
		if(strucElement_type==0)
				elemType=MORPH_RECT;

		if(strucElement_type==1)
				elemType=MORPH_CROSS;

		if(strucElement_type==2)
				elemType=MORPH_ELLIPSE;

		Mat element=getStructuringElement(elemType,Size(2*structElement_size+1,2*structElement_size+1), Point(structElement_size,structElement_size));			cout<<"element:\n"<<element<<endl;
		cout<<"size\n"<<Size(2*structElement_size+1,2*structElement_size+1)<<endl;
		erode(inputImage,temp,element);
		dilate(temp,output,element);
		return output;
}

Mat dilation(Mat inputImage, int strucElement_type, int structElement_size){
		Mat output;
		int elemType;
		//to choose witch structuring element to apply
		if(strucElement_type==0)
			elemType=MORPH_RECT;

		if(strucElement_type==1)
			elemType=MORPH_CROSS;

		if(strucElement_type==2)
			elemType=MORPH_ELLIPSE;
		Mat element=getStructuringElement(elemType,Size(2*structElement_size+1,2*structElement_size+1), Point(structElement_size,structElement_size));
		dilate(inputImage,output,element,Point(-1,-1),1);
		cout<<"element:\n"<<element<<endl;
					cout<<"size\n"<<Size(2*structElement_size+1,2*structElement_size+1)<<endl;

		return output;
}

Mat erosion(Mat inputImage, int strucElement_type, int structElement_size){
		Mat output;
		int elemType;
		//to choose witch structuring element to apply
		if(strucElement_type==0)
				elemType=MORPH_RECT;

		if(strucElement_type==1)
				elemType=MORPH_CROSS;

		if(strucElement_type==2)
				elemType=MORPH_ELLIPSE;
		Mat element=getStructuringElement(elemType,Size(2*structElement_size+1,2*structElement_size+1), Point(structElement_size,structElement_size));
		erode(inputImage,output,element,Point(-1,-1), 1);
		return output;

}

Mat closing(Mat inputImage, int strucElement_type, int structElement_size){
		Mat output,temp;
		int elemType;
		//to choose witch structuring element to apply
		if(strucElement_type==0)
				elemType=MORPH_RECT;

		if(strucElement_type==1)
				elemType=MORPH_CROSS;

		if(strucElement_type==2)
				elemType=MORPH_ELLIPSE;

		Mat element=getStructuringElement(elemType,Size(2*structElement_size+1,2*structElement_size+1), Point(structElement_size,structElement_size));

		dilate(inputImage,temp,element);
		erode(temp,output,element);
		return output;
}


/*
 *  COMPUTE ORIANTATION USING COVARIANCE MATRIX("inertia matrix" of the object)
 */
vector<double>  computeOrientation(vector <vector<Point2i> > contours,vector<int> rodA){
		vector<Moments> mA(rodA.size());

		for(int i=0;i<rodA.size();i++){
				mA[i]=moments(contours[rodA[i]],true);

		}
		vector<double> thetaA(rodA.size());

		for(int j=0;j<rodA.size();j++){
				thetaA[j]=-0.5*atan((2*mA[j].mu11)/(mA[j].mu02-mA[j].mu20));

		}



		return thetaA;
}


/*
 * COMPUTE THE BARYCENTER USING THE CONTOURS MOMENTS
 */
vector<Point2f> computeBarycenters(vector <vector<Point2i> > contours,vector<int> rod){

		//compute the moments necessary to calculate the barycenters
		vector<Moments> m(rod.size());
		for(int i=0;i<rod.size();i++){
				m[i]=moments(contours[rod[i]],true);

			}

		vector<Point2f> barycenters(rod.size());
		//barycenters=(first order moment)/(area)
		for(int j=0;j<rod.size();j++){
				barycenters[j].x=(m[j].m10)/(m[j].m00);
				barycenters[j].y=(m[j].m01)/(m[j].m00);

			}
		//uncomment to see the barycenters
		//cout<<"barycenters: " << barycenters<<endl;
		return barycenters;
}
/*
 * APPROXIMATE THE WIDTH AND LENGHT AT THE BARYCENTER AS SQAURE ROOD OF THE APPROPRIATE EIGENVALUE
 */
Mat showAxis( Mat inputImage,vector<int> rod,vector<vector<Point2i> > contours){


	//compute the moments needed to build the covariance matrixes
		vector<Moments> mom(rod.size());

		for(int i=0;i<rod.size();i++){
				mom[i]=moments(contours[rod[i]],true);

		}

	//build the covar matrixes and calculate the eigenvectors and eigenvalues
		vector<Mat> eigen_val(rod.size()),eigen_vec(rod.size());
		vector<Mat> covar(rod.size());

		for(int y=0;y<rod.size();y++){

		//compute the covariance matrix
				double area=mom[y].m00;
				double c11=mom[y].mu20/area;
				double c12=mom[y].mu11/area;
				double c21=mom[y].mu11/area;
				double c22=mom[y].mu02/area;

		//compute the eigenvalues and eigenvectors of the covar matrix

				covar[y]=(Mat_<double>(2,2) <<c11,c12,c21,c22);
				eigen(covar[y],eigen_val[y],eigen_vec[y]);
		}

	//barycenters vector
		vector<Point2f> bar=computeBarycenters(contours,rod);


	//approximation of the lenght of the semi-axis
		vector<double> approx_lenght(rod.size()),approx_width(rod.size());
		for(int h=0;h<rod.size();h++){
				approx_lenght[h]=sqrt(eigen_val[h].at<double>(0,0));
				approx_width[h]=sqrt(eigen_val[h].at<double>(0,1));
		}

	//draw the major and minor axis
		for(int i=0;i<rod.size();i++){

				line(inputImage,bar[i], (Point_<int>)bar[i] + Point2i(eigen_vec[i].at<double>(0,0)*approx_lenght[i] ,eigen_vec[i].at<double>(0,1)*approx_lenght[i]) ,Scalar(255,255,0));
				line(inputImage,bar[i], (Point_<int>)bar[i] + Point2i(eigen_vec[i].at<double>(1,0)*approx_width[i] ,eigen_vec[i].at<double>(1,1)*approx_width[i]) ,Scalar(0,0,255));
				circle(inputImage,bar[i],2,Scalar(255,255,255));
		}
	return inputImage;
}


/*
 * USE AN EIGENVECTOR CENTERED SYSTEM COORDINATE AND FIND THE (EXACT) WIDTH OF THE RODS AT THE BARYCENTER
 */
struct WidthAndLenght_AtBarycenter hotellingTrans(Mat inputImage, vector<vector<Point2i> > contours, vector<int> rod){


		//compute the moments needed to build the covariance matrixes
		vector<Moments> mom(rod.size());

		for(int i=0;i<rod.size();i++){
				mom[i]=moments(contours[rod[i]],true);
		}

		//contruct the covar matrixes and calculate the eigenvectors and eigenvalues
		vector<Mat> eigen_val(rod.size()),eigen_vec(rod.size());
		vector<Mat> covar(rod.size());

		for(int y=0;y<rod.size();y++){

			//compute the covariance matrix
				double area=mom[y].m00;
				double c11=mom[y].mu20/area;
				double c12=mom[y].mu11/area;
				double c21=mom[y].mu11/area;
				double c22=mom[y].mu02/area;

			//compute the eigenvalues and eigenvectors of the covar matrix

				covar[y]=(Mat_<double>(2,2) <<c11,c12,c21,c22);
				eigen(covar[y],eigen_val[y],eigen_vec[y]);
		}
		//create a matrix double the size of the inputImage
		//this is done in order to  contain the rotated contour in the eventuality we want to show it
		Mat output((inputImage.size()+inputImage.size()),CV_8UC3);

		//barycenter vector
		vector<Point2f> bar=computeBarycenters(contours,rod);

		//hoteling tranform to get the exact lenght of the axis
		//find the intersections of the main axis with the contour(in the centered eigenvectors coordinate system)
		vector<vector<Point2i> > rotated(rod.size());
		vector<vector<Point2i> > candidates(rod.size());
		vector<vector<Point2i> > candidatesL(rod.size());
		vector<double> exact_width(rod.size());
		vector<double> exact_lenght(rod.size());
		for(int l=0;l<rod.size();l++){
				vector<Point2i> temp(contours[rod[l]].size());
				for(int j=0;j<contours[rod[l]].size();j++){

						vector<double> diff(contours[rod[l]].size());
						Point2i centered = (Point2i)contours[rod[l]][j]- (Point2i)bar[l];

						temp[j].x=  eigen_vec[l].at<double>(0,0)*centered.x+eigen_vec[l].at<double>(0,1)*centered.y;
						temp[j].y=  eigen_vec[l].at<double>(1,0)*centered.x+eigen_vec[l].at<double>(1,1)*centered.y;



				}
				//uncomment this piece of code if you want to display the rotated contour obtained using the hotelling transform
				/*
				for(int j=0;j<contours[rod[l]].size();j++){
						temp[j]=(Point2i)temp[j]+(Point2i)(bar[l])+(Point2i)(50,50);
				}
				rotated[l]=(temp);
				drawContours(output,rotated,l,Scalar(0,255,255), 2);
				imshow("rotated",output);
				waitKey(0);
				for(int j=0;j<contours[rod[l]].size();j++){
						temp[j]=(Point2i)temp[j]-(Point2i)(bar[l]);
				}
				 */



				rotated[l]=(temp);
				//after computing the rotated contour points we store those that belong to the rotated axes
				//axes = eigenvectors directions

				int k=0;
				for(int s=0;s<contours[rod[l]].size();s++){
						//CANDIATES POINTS ON THE MAJOR AXIS(USE TO COMPUTE THE EXACT WIDTH)
						if((int)temp[s].x==0){
								candidates[l].push_back(temp[s]);

						}
						//CANDIATES POINTS ON THE MINOR AXIS(USE TO COMPUTE THE EXACT LENGHT)
						if((int)temp[s].y==0){
								candidatesL[l].push_back(temp[s]);

						}

				}

				/*now it's time to compute the width at the baricenter as the distance between points
				that belongs to the rotated axis(axis along the eigenvectors directions)
				 */
				/*NOTE: DUE TO ROUNDING ERRORS WE GET MORE THAN JUST TWO POINTS PER AXIS.
      	  	  			NEAR POINTS GET TO BE STORED NEAR IN THE CANDIDATES VECTOR.
		        		SO I TAKE THE FIRST AND THE LAST THAT USUALLY CORRISPONDS TO OPPOSITE SIDES OF THE OBJECTS.
				 */
				//UNCOMMENT TO SHOW THE CANDIDATES VECTORS
				//cout<<"candidates for width rod"<<l<<" "<<candidates[l]<<endl;
				//cout<<"candidates for lenght rod"<<l<<" "<<candidatesL[l]<<endl;

				//COMPUTE THE EXACT LENGHT AND WIDTH
				exact_width[l]=abs(candidates[l][0].y)+abs(candidates[l][candidates[l].size()-1].y);
				exact_lenght[l]=abs(candidatesL[l][0].x)+abs(candidatesL[l][candidatesL[l].size()-1].x);

				//cout<<"exact lenght "<<exact_lenght[l]<<endl;
				//cout<<"exact width "<<exact_width[l]<<endl;

		}
		//we can show the axes having eigenvectors directions and lenght half of the exact leenght and width
		//UNCOMMENT TO DISPLAY THE EXACT LENGHT AXIS
		/*
		Mat C=inputImage.clone();
		for(int i=0;i<rod.size();i++){

				line(C,bar[i], (Point_<int>)bar[i] + Point2i(eigen_vec[i].at<double>(0,0)*(exact_lenght[i]/2) ,eigen_vec[i].at<double>(0,1)*(exact_lenght[i])/2) ,Scalar(252,20,0));
				line(C,bar[i], (Point_<int>)bar[i] + Point2i(eigen_vec[i].at<double>(1,0)*(exact_width[i]/2) ,eigen_vec[i].at<double>(1,1)*(exact_width[i])/2) ,Scalar(0,0,255));
				circle(C,bar[i],2,Scalar(255,255,255));
			}
		*/
		struct WidthAndLenght_AtBarycenter f={exact_width,exact_lenght};

		return f;
}


/*
 * USE WATER SHED ALGHORITHM TO TRY TO FIND THE RIGHT CONTOURS FOR THE TOUCHING RODS
 */
Mat findRightContours(Mat inputImage){

		//take the inputImage and convert it in grayscale image
		//this is done in order to binarize it
		Mat gray;
		cvtColor(inputImage,gray,CV_BGR2GRAY);
		Mat binarized=Thresh(gray);
		//invert the rappresentation(foreground==white)
		Mat inv;
		threshold(binarized,inv,5,255,THRESH_BINARY_INV);


		//performe distance transform on the image
		//(assigns each pixel the value of the distance from the nearest black(0) pixel)

		Mat dist(inputImage.size(),inputImage.type());
		distanceTransform(inv,dist,DIST_L2,5, CV_8U);
		//normalize to make it displayable
		normalize(dist, dist, 0, 1, NORM_MINMAX);
		//imshow("Distance Transform water thresholded", dist);

		//Threshold to obtain the peaks
		//These will be the markers for the foreground objects
	    threshold(dist, dist, .61, 1., CV_THRESH_BINARY);


	   // imshow("watershed dist",dist);


	    //find the markers as the contours
	    vector<vector<Point> > contours;
	    vector<Vec4i> hierarchy;
	    Mat dist_8U;
	    dist.convertTo(dist_8U,CV_8U);
	    findContours(dist_8U, contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	    //cout<<"c "<<contours.size()<<endl;

	    Mat markers = Mat::zeros(dist.size(), CV_32SC1);

	    //choose only the big contours to pass as markers
	    //this is done to avoid keeping small fragments that will result in different objects
	    for( int i = 0; i< contours.size(); i++ ){
		    	Moments cont_moments=moments(contours[i]);
		    	if(cont_moments.m00>20){
		    			drawContours( markers, contours, i, Scalar::all(i+200), 1 , 8);
		    	}
	     }

	    //mark point to(this is done in order to mark the usniform background region as a diffferent object
	    circle(markers, Point(2,2), 3, CV_RGB(120,120,120), -1);

	    //extract the holes from the image in order to consider them different regions
	    vector<vector<Point> > contours_temp;
	    vector<Vec4i> hierarchy_temp;
	    findContours(binarized, contours_temp,hierarchy_temp,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	    struct features G=extractFeutures(inputImage,contours_temp,hierarchy_temp);
		for(int s=0;s<G.holesCenters.size();s++){
				circle(markers, G.holesCenters[s],2,Scalar(125,125,125),-1);
		}

		//apply watershed alghorithm
		watershed(inputImage,markers);

		//to see what's inside markers up to now
		Mat markers_8U=Mat::zeros(markers.size(),CV_8U);
		markers.convertTo(markers_8U,CV_8UC1);
		bitwise_not(markers_8U,markers_8U);
		imshow("output watershed", markers_8U);


		//store the boundary points in a mask
		//watershed returns -1 at each pixel that are on the boundary pixels
		Mat mask(inputImage.size(), CV_8UC1);
		for(int y=0;y<markers.rows;y++){
				for(int u=0;u<markers.cols;u++){
	        			if(markers.at<int>(y,u)==-1){
	        					mask.at<uchar>(y,u)=255;
	        			}
	        	}
	    }

		//imshow("mask", mask);
		//return a black and white picture with boundaries
		Mat temp=binarized+mask;

		imshow("watershed final result", temp);
		waitKey(0);
		return temp;

}


/*
 *ERODE THE TOUUCHING ROD CONTOUR AND CLASSIFY THE RODS BASED ON THE LENGHT
 */
struct touchingRods_contours touchingRods(Mat inputImage){


		//binarize the image
		Mat gray(inputImage.size(),inputImage.type());
		cvtColor(inputImage,gray,CV_BGR2GRAY);
		Mat binarized=Thresh(gray);

		//imshow("binarized", binarized);

		//performe distance transform on the image(assigns each pixel the value of the distance from the nearest black(0) pixel)
		Mat dist(inputImage.size(),inputImage.type());
		distanceTransform(binarized,dist,DIST_L2,5, CV_8U);
		//normalize to visualize
		normalize(dist, dist, 0, 1, NORM_MINMAX);
		//imshow("Distance Transform Image", dist);

		// Threshold to obtain the peaks
		// This will be the image with the foreground objects
		threshold(dist, dist, .55, 1., CV_THRESH_BINARY);

		//dilate a bit
		dist=dilation(dist,0,1);
		//imshow("dist", dist);

		//find the objects on the eroded image
		vector<vector<Point> > eroded_contours;
		vector<Vec4i> hierarchy;
		Mat dist_8U;
		dist.convertTo(dist_8U,CV_8U);
		findContours(dist_8U, eroded_contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

		//draw the contours

		for( int i = 0; i< eroded_contours.size(); i++){
				Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
				drawContours( inputImage, eroded_contours, i, color, 1 , 8);
		}

		//classify the contours
		Point2f vertices[4];
		vector<int> rodsA;
		vector<int> rodsB;
		float height;
		//look at the image and classify the rods based on their lenght
		for(int l=0;l<eroded_contours.size();l++){
				RotatedRect rect=minAreaRect(eroded_contours[l]);
				cout<<"rect size "<<rect.size<<endl;

				if (rect.size.height>rect.size.width ){
						height=rect.size.height;
				}
				else{
						height=rect.size.width;
				}
				if( height<100 && height>30){
						rodsA.push_back(l);
				}
				if( height>100){
						rodsB.push_back(l);
				}

		}




		//imshow("Touching rods processed",inputImage);
		//waitKey(0);
		//put the found ros and return them
		vector<vector<Point> >erodedRodsA_contours(rodsA.size());
		vector<vector<Point> >erodedRodsB_contours(rodsB.size());
		//for touching rodsB
		for (int v=0;v<rodsA.size();v++){
				erodedRodsA_contours[v]=eroded_contours[rodsA[v]];
		}
		for (int h=0;h<rodsB.size();h++){
				erodedRodsB_contours[h]=eroded_contours[rodsB[h]];
		}
		struct touchingRods_contours t={erodedRodsA_contours,erodedRodsB_contours};
		return t;
}


/*
 * FIND CORNERS WITH HARRIS AND SEPARETE THE TOUCHING RODS CONTOUR
 */
struct touchingRods_contours separateUsingCorners(Mat inputImage){

		vector< vector<Point2i> > contours_updated;
		vector<Vec4i> hierarchy;

		//binarize the image
		Mat gray(inputImage.size(), CV_8UC1);
		cvtColor(inputImage,gray,CV_BGR2GRAY);
		Mat temp_binary(inputImage.size(),CV_8UC1);
		temp_binary=Thresh(gray);
		//bitwise_not(temp_binary,temp_binary);
		//imshow("temp bin", temp_binary);

		//initialize a counter that goes to zero when the contours on the image are not too big(area<6000)
		int count=1;
		while(count!=0){
				//find the contours
				findContours(temp_binary, contours_updated,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
				count--;
				//check all of them
				for(int a=0;a<contours_updated.size();a++){
						Moments cont_moments=moments(contours_updated[a]);
						//if they are too big continue
						if(cont_moments.m00>4600 && cont_moments.m00<40000 && hierarchy[hierarchy[a][2]][0]!=-1){

								Mat temp_color(inputImage.size(),CV_8UC3);
								drawContours(temp_color,contours_updated,a,Scalar(255,255,255),CV_FILLED,8);
								//imshow("temp", temp_binary);
								//look for strong corners in the image
								//NOTICE: WE LOOK FOR 4 OF THEM
								vector<Point2f> corners;
								goodFeaturesToTrack(temp_binary,corners,4,0.01,5,noArray(),5,false);
								//uncomment to see witch corners have been found
								//the darker the stronger they are
								/*
								for(int y=0;y<corners.size();y++){
										circle(temp_color,corners[y],3,Scalar(0,50+40*y,0),2,8);
								}
								imshow("corners", temp_color);
								*/



								//find the two nearest detected corners and draw a line
								double min=100;
								int index;
								for(int b=1;b<corners.size();b++){
										Point2f diff=corners[0]-corners[b];
										double di=sqrt(diff.x*diff.x+diff.y*diff.y);
										if(di<min){
											min=di;
											index=b;
										}
								}
								line(temp_binary,corners[0],corners[index],Scalar(0,0,0),2,8);
								//imshow("temp binary",temp_binary);
								//waitKey(0);
								findContours(temp_binary, contours_updated,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
								//check how many are there
								//cout<<"cont size "<<contours_updated.size()<<endl;
								for(int c=0;c<contours_updated.size();c++){
										Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
										drawContours(inputImage,contours_updated,c,color,1,8);
										Moments cont_mom=moments(contours_updated[c]);
										//cout<<"area "<<cont_mom.m00<<endl;
										if(cont_mom.m00>4600 && cont_mom.m00<40000 && count!=1){
												count++;
										}
								}

								//imshow("contours", inputImage);
						}
				}

		}
		//imshow("image processed using Harris corners", temp_binary);
		//categorize the contours using their area
		struct touchingRods_contours f;
		vector<vector<Point> > touchingRodsA;
		vector<vector<Point> > touchingRodsB;
		for(int d=0;d<contours_updated.size();d++){
				Moments cont_mom=moments(contours_updated[d]);
				//if the contours is between 2000 and 3000 square pixels then it's a type A rod
				if(cont_mom.m00>2000 && cont_mom.m00<3000){
						touchingRodsA.push_back(contours_updated[d]);
				}
				//if the contours is between 4000 and 4500 square pixels then it's a type B rod
				if(cont_mom.m00>4000 && cont_mom.m00<4600){
						touchingRodsB.push_back(contours_updated[d]);
    	    	}
		}

		f={touchingRodsA,touchingRodsB};
		return f;


}


void DisplayTableResults(Mat inputImage,vector< vector<Point2i> > contours,vector<Vec4i> hierarchy){

		struct features F=extractFeutures( inputImage, contours, hierarchy);

		//compute oriention
		vector<double>  thA=computeOrientation( contours, F.rodA_contours);
		vector<double>  thB=computeOrientation(contours, F.rodB_contours);
		vector<double>  th_touchingA;
		vector<double>  th_touchingB;

		vector<RotatedRect> rect_touchingRodsA;
		vector<RotatedRect> rect_touchingRodsB;



		struct WidthAndLenght_AtBarycenter barWidthRodA=hotellingTrans(inputImage, contours, F.rodA_contours);
		struct WidthAndLenght_AtBarycenter barWidthRodB=hotellingTrans(inputImage, contours, F.rodB_contours);

		struct WidthAndLenght_AtBarycenter barWidthTouchingRodA;
		struct WidthAndLenght_AtBarycenter barWidthTouchingRodB;


		if(F.touchingRodsA_contours.empty()==false){
				//for touching rodsA
				vector <int> touching_rodsA(F.touchingRodsA_contours.size());
				for(int g=0;g<F.touchingRodsA_contours.size();g++){
						touching_rodsA[g]=g;
				}
				for (int a=0;a<F.touchingRodsA_contours.size();a++){
						rect_touchingRodsA.push_back(minAreaRect(F.touchingRodsA_contours[a]));
				}
				 barWidthTouchingRodA=hotellingTrans(inputImage, F.touchingRodsA_contours, touching_rodsA);
				 th_touchingA=computeOrientation(F.touchingRodsA_contours, touching_rodsA);
		}

		if(F.touchingRodsB_contours.empty()==false){
				vector <int> touching_rodsB(F.touchingRodsB_contours.size());
				for(int c=0;c<F.touchingRodsB_contours.size();c++){
						touching_rodsB[c]=c;
				}

				for (int v=0;v<F.touchingRodsB_contours.size();v++){
						rect_touchingRodsB.push_back(minAreaRect(F.touchingRodsB_contours[v]));
				}
				 barWidthTouchingRodB=hotellingTrans(inputImage, F.touchingRodsB_contours, touching_rodsB);
				 th_touchingB=computeOrientation(F.touchingRodsB_contours, touching_rodsB);

		}

		// display the informations about each rod and hole
		cout<<"Total number of rod found: "<<F.minRectRodA.size()+F.minRectRodB.size()+F.touchingRodsA_contours.size()+F.touchingRodsB_contours.size() <<endl;
		cout<<"*****************************************\n";
		cout<<"          instance  position         (lenght x height)           orientation     mytheta     width at the barycenter \n";


		for(int r=0;r<F.minRectRodA.size();r++){

				cout << "\nRod A:        "<<r<<"    "<<(Point_<int>)F.minRectRodA[r].center<<"                "<<(Size_<int>)F.minRectRodA[r].size<<"               "<<(int)F.minRectRodA[r].angle <<"        "<< (int)  ((thA[r]*180)/(3.14))<<"              "<< (int)barWidthRodA.exactWidth[r]  <<endl ;
		}



		if(F.touchingRodsA_contours.empty()==false){
				for(int w=0;w<F.touchingRodsA_contours.size();w++){
						cout << "\nTouching Rod found:\nRod A:        "<<w<<"    "<<(Point_<int>)rect_touchingRodsA[w].center<<"                "<<(Size_<int>)rect_touchingRodsA[w].size<<"             "<<(int)rect_touchingRodsA[w].angle<<"            "<<(int)((th_touchingA[w]*180)/(3.14))<<"           "<<(int) barWidthTouchingRodA.exactWidth[w] <<endl;
				}
		}

		for(int e=0;e<F.minRectRodB.size();e++){
				cout << "\nRod B:        "<<e<<"    "<<(Point_<int>)F.minRectRodB[e].center<<"                "<<(Size_<int>)F.minRectRodB[e].size<<"             "<<(int)F.minRectRodB[e].angle<<"            "<<(int)((thB[e]*180)/(3.14))<< "         "<< (int)barWidthRodB.exactWidth[e]<<endl;
		}

		if(F.touchingRodsB_contours.empty()==false){
				for(int h=0;h<F.touchingRodsB_contours.size();h++){
						cout << "\nTouching Rod found:\nRod B:        "<<h<<"    "<<(Point_<int>)rect_touchingRodsB[h].center<<"                "<<(Size_<int>)rect_touchingRodsB[h].size<<"             "<<(int)rect_touchingRodsB[h].angle<<"            "<<(int)((th_touchingB[h]*180)/(3.14))<<"            "<<(int) barWidthTouchingRodB.exactWidth[h] <<endl;
				}
		}

		cout  << "\n\nTotal number of holes found: "<< F.holes_contours.size();
		cout  << "\n*******************************************";
		cout  << "\n          instance  position     diameter";
		for(int l=0;l<F.holes_contours.size();l++){
				cout << "\nHole :        "<<l<<"    "<<(Point_<int>)F.holesCenters[l]<<"       "<<(int)(2*F.holesRadius[l])<<endl;;
		}

}
/*
 * HOUGH TRANSFORM TO FIND CIRCLES
 */


Mat hugh(Mat inputImage){
	vector<Vec3f> circles;
	HoughCircles(inputImage, circles,CV_HOUGH_GRADIENT, 1,inputImage.cols/2,59,40,0,0);
	cout<<circles.size()<<endl;
	for( size_t i = 0; i < circles.size(); ++i ) {
	circle(inputImage,Point(cvRound(circles[i][0]), cvRound(circles[i][1])),cvRound(circles[i][2]),Scalar(255,255,255),2,CV_AA);
	}
	return inputImage;
}


/*
 * FIND THE CONVEX HULL OF THE TOUCHING ROD CONTOUR AND LOOK FOR THE DIFECTS(DOESN'T GIVE NICE RESULTS)
 */
Mat convexityD(Mat inputImage){

	vector<vector<Point> > Contours;
	vector<Vec4i> hierarchy;

	findContours(inputImage, Contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	Mat frame=Mat::zeros(inputImage.size(),CV_8UC3);
	Mat output=Mat::zeros(inputImage.size(),CV_8UC1);
	vector<vector<Point> >hull(Contours.size());
	vector<vector<int> > hullsI(Contours.size()); // Indices to contour points
	vector<vector<Vec4i> > defects(Contours.size());
	for (int k = 0; k < Contours.size(); k++)
	{
	    convexHull(Contours[k], hull[k], false);
	    convexHull(Contours[k], hullsI[k], false);
	    if(hullsI[k].size() > 3 ) // You need more than 3 indices
	    {
	        convexityDefects(Contours[k], hullsI[k], defects[k]);

	    }
	}

	/// Draw convexityDefects
	for (int i = 1; i < Contours.size(); ++i)
	{
		for(int y=1;y<hull[i].size();y++){
			line(output,hull[i][y], hull[i][y-1], Scalar(255,255,255),1 );
		}

		for(int j=0; j<defects.size(); ++j)
	    {

	    const Vec4i& v = defects[i][j];

	         //  filter defects by depth, e.g more than 10
	        {
	            int startidx = v[0]; Point ptStart(Contours[i][startidx]);
	            int endidx = v[1]; Point ptEnd(Contours[i][endidx]);
	            int faridx = v[2]; Point ptFar(Contours[i][faridx]);

	            line(frame, ptStart, ptEnd, Scalar(0, 255, 0), 1);
	            line(frame, ptStart, ptFar, Scalar(0, 255, 0), 1);
	            line(frame, ptEnd, ptFar, Scalar(0, 255, 0), 1);

	            circle(frame, ptFar, 2, Scalar(0, 255, 255), 2);
	        }
	    }
	}
	bitwise_not(inputImage,inputImage);
	imshow("o", output+inputImage);
	return frame;
}




