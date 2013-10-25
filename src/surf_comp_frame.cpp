#include <cv.h>
#include <highgui.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/nonfree/features2d.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <boost/program_options.hpp>
#include "3dms-func.h"

// パノラマ画像の大きさ
#define PANO_W 6000
#define PANO_H 3000

// 合成フレームのFPS
#define TARGET_VIDEO_FPS 30

using namespace std;
using namespace cv;
using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;
using boost::program_options::notify;

// Tilt :
int SetTiltRotationMatrix(Mat *tiltMatrix, double tilt_deg) {
	double tilt_angle;

	tilt_angle = tilt_deg / 180.0 * M_PI;
	(*tiltMatrix).at<double> (1, 1) = cos(tilt_angle);
	(*tiltMatrix).at<double> (1, 2) = -sin(tilt_angle);
	(*tiltMatrix).at<double> (2, 1) = sin(tilt_angle);
	(*tiltMatrix).at<double> (2, 2) = cos(tilt_angle);

	//	cvmSet(tiltMatrix, 1, 1, cos(tilt_angle));
	//	cvmSet(tiltMatrix, 1, 2, -sin(tilt_angle));
	//	cvmSet(tiltMatrix, 2, 1, sin(tilt_angle));
	//	cvmSet(tiltMatrix, 2, 2, cos(tilt_angle));
	return 0;
}

int SetPanRotationMatrix(Mat *panMatrix, double pan_deg) {
	double pan_angle;
	pan_angle = pan_deg / 180.0 * M_PI;

	(*panMatrix).at<double> (2, 2) = cos(pan_angle);
	(*panMatrix).at<double> (2, 0) = -sin(pan_angle);
	(*panMatrix).at<double> (0, 2) = sin(pan_angle);
	(*panMatrix).at<double> (0, 0) = cos(pan_angle);
	//	cvmSet(panMatrix, 2, 2, cos(pan_angle));
	//	cvmSet(panMatrix, 2, 0, -sin(pan_angle));
	//	cvmSet(panMatrix, 0, 2, sin(pan_angle));
	//	cvmSet(panMatrix, 0, 0, cos(pan_angle));
	return 0;
}

// Roll :
int SetRollRotationMatrix(Mat *rollMatrix, double roll_deg) {
	double roll_angle;

	roll_angle = roll_deg / 180.0 * M_PI;
	(*rollMatrix).at<double> (0, 0) = cos(roll_angle);
	(*rollMatrix).at<double> (0, 1) = -sin(roll_angle);
	(*rollMatrix).at<double> (1, 0) = sin(roll_angle);
	(*rollMatrix).at<double> (1, 1) = cos(roll_angle);
	//	cvmSet(rollMatrix, 0, 0, cos(roll_angle));
	//	cvmSet(rollMatrix, 0, 1, -sin(roll_angle));
	//	cvmSet(rollMatrix, 1, 0, sin(roll_angle));
	//	cvmSet(rollMatrix, 1, 1, cos(roll_angle));
	return 0;
}

// Pitch :
int SetPitchRotationMatrix(Mat *pitchMatrix, double pitch_deg) {
	double pitch_angle;

	pitch_angle = pitch_deg / 180.0 * M_PI;
	(*pitchMatrix).at<double> (1, 1) = cos(pitch_angle);
	(*pitchMatrix).at<double> (1, 2) = -sin(pitch_angle);
	(*pitchMatrix).at<double> (2, 1) = sin(pitch_angle);
	(*pitchMatrix).at<double> (2, 2) = cos(pitch_angle);
	//	cvmSet(pitchMatrix, 1, 1, cos(pitch_angle));
	//	cvmSet(pitchMatrix, 1, 2, -sin(pitch_angle));
	//	cvmSet(pitchMatrix, 2, 1, sin(pitch_angle));
	//	cvmSet(pitchMatrix, 2, 2, cos(pitch_angle));
	return 0;
}

// Yaw
int SetYawRotationMatrix(Mat *yawMatrix, double yaw_deg) {
	double yaw_angle;

	yaw_angle = yaw_deg / 180.0 * M_PI;
	(*yawMatrix).at<double> (2, 2) = cos(yaw_angle);
	(*yawMatrix).at<double> (2, 0) = -sin(yaw_angle);
	(*yawMatrix).at<double> (0, 2) = sin(yaw_angle);
	(*yawMatrix).at<double> (0, 0) = cos(yaw_angle);
	//	cvmSet(yawMatrix, 2, 2, cos(yaw_angle));
	//	cvmSet(yawMatrix, 2, 0, -sin(yaw_angle));
	//	cvmSet(yawMatrix, 0, 2, sin(yaw_angle));
	//	cvmSet(yawMatrix, 0, 0, cos(yaw_angle));
	return 0;
}

void setHomographyReset(Mat* homography) {
	cvZero(homography);
	(*homography).at<double> (0, 0) = 1;
	(*homography).at<double> (1, 1) = 1;
	(*homography).at<double> (2, 2) = 1;
	//cvmSet(homography, 0, 0, 1);
	//cvmSet(homography, 1, 1, 1);
	//cvmSet(homography, 2, 2, 1);
}

double compareSURFDescriptors(const float* d1, const float* d2, double best,
		int length) {
	double total_cost = 0;
	assert( length % 4 == 0 );
	for (int i = 0; i < length; i += 4) {
		double t0 = d1[i] - d2[i];
		double t1 = d1[i + 1] - d2[i + 1];
		double t2 = d1[i + 2] - d2[i + 2];
		double t3 = d1[i + 3] - d2[i + 3];
		total_cost += t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3;
		if (total_cost > best)
			break;
	}
	return total_cost;
}

void get_histimage(Mat image, Mat *hist_image) {
	MatND hist; // ヒストグラム
	Scalar mean, dev; // 平均と分散の格納先
	float hrange[] = { 0, 256 }; // ヒストグラムの輝度値レンジ
	const float* range[] = { hrange }; // チャネルごとのヒストグラムの輝度値レンジ（グレースケールなので要素数は１）
	int binNum = 256; // ヒストグラムの量子化の値
	int histSize[] = { binNum }; // チャネルごとのヒストグラムの量子化の値
	int channels[] = { 0 }; // ヒストグラムを求めるチャネル指定
	int dims = 1; // 求めるヒストグラムの数


	float max_dev = FLT_MIN, min_dev = FLT_MAX; // エッジ画像におけるヒストグラムの分散のmin max
	float max_mean = FLT_MIN, min_mean = FLT_MAX; // エッジ画像におけるヒストグラムの平均のmin max
	float sum_mean = 0.0;
	Rect roi_rect;
	Mat count(10, 10, CV_32F, cv::Scalar(0)); // エッジの数を格納するカウンタ

	cout << "making histgram" << endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			double max_value;
			int bin_w;
			Mat tmp_img(image, cv::Rect(j * 128, i * 72, 128, 72));
			calcHist(&tmp_img, 1, channels, Mat(), hist, dims, histSize, range,
					true, false);

			meanStdDev(hist, mean, dev);
			count.at<float> (i, j) = dev[0];

			if (dev[0] < min_dev)
				min_dev = dev[0];
			if (dev[0] > max_dev)
				max_dev = dev[0];

			if (mean[0] < min_mean)
				min_mean = mean[0];
			if (mean[0] > max_mean)
				max_mean = mean[0];

			sum_mean += mean[0];
			std::cout << "count : " << mean << std::endl;

			minMaxLoc(hist, NULL, &max_value, NULL, NULL);
			hist *= hist_image[i * 10 + j].rows / max_value;
			bin_w = cvRound((double) 260 / 256);

			for (int k = 0; k < 256; k++)
				rectangle(hist_image[i * 10 + j], Point(k * bin_w, hist_image[i
						* 10 + j].rows), cvPoint((k + 1) * bin_w, hist_image[i
						* 10 + j].rows - cvRound(hist.at<float> (k))),
						cvScalarAll(0), -1, 8, 0);
			roi_rect.width = tmp_img.cols;
			roi_rect.height = tmp_img.rows;
			roi_rect.x = 260;
			Mat roi(hist_image[i * 10 + j], roi_rect);
			tmp_img.copyTo(roi);
		}
	}
}
/*
 *  透視投影変換後の画像をパノラマ平面にマスクを用いて
 *  上書きせずに未投影の領域のみに投影する関数
 *
 * @Param  src パノラマ画像に投影したい画像
 * @Param  dst パノラマ画像
 * @Param mask 投影済みの領域を表したマスク画像
 * @Param  roi 投影したい画像の領域を表した画像
 *
 *  （＊maskは処理後に更新されて返される）
 */
void make_pano(Mat src, Mat dst, Mat mask, Mat roi) {

	//サイズの一致を確認
	if (src.cols == dst.cols && src.rows == dst.rows) {
		int h = src.rows;
		int w = src.cols;
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				if (mask.at<unsigned char> (j, i) == 0) {

					dst.at<Vec3b> (j, i) = src.at<Vec3b> (j, i);
					if (roi.at<unsigned char> (j, i) == 255)
						mask.at<unsigned char> (j, i) = roi.at<unsigned char> (
								j, i);
				}
			}
		}
	}
}
/*
 * @Param descriptors1 特徴量１
 * @Param descriptors2 特徴量２
 * @Param key1         特徴点１
 * @Param key2         特徴点２
 * @Param matches      良いマッチングの格納先
 * @Param pt1          良いマッチングの特徴点座標１
 * @Param pt2          良いマッチングの特徴点座標２
 */
void good_matcher(Mat descriptors1, Mat descriptors2, vector<KeyPoint> *key1,
		vector<KeyPoint> *key2, std::vector<cv::DMatch> *matches, vector<
				Point2f> *pt1, vector<Point2f> *pt2) {

	FlannBasedMatcher matcher;
	vector<std::vector<cv::DMatch> > matches12, matches21;
	std::vector<cv::DMatch> tmp_matches;
	int knn = 10;
	//BFMatcher matcher(cv::NORM_HAMMING, true);
	//matcher.match(objectDescriptors, imageDescriptors, matches);


	matcher.knnMatch(descriptors2, descriptors1, matches21, knn);
	matcher.knnMatch(descriptors1, descriptors2, matches12, knn);
	tmp_matches.clear();
	// KNN探索で，1->2と2->1が一致するものだけがマッチしたとみなされる
	for (size_t m = 0; m < matches12.size(); m++) {
		bool findCrossCheck = false;
		for (size_t fk = 0; fk < matches12[m].size(); fk++) {
			cv::DMatch forward = matches12[m][fk];
			for (size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++) {
				cv::DMatch backward = matches21[forward.trainIdx][bk];
				if (backward.trainIdx == forward.queryIdx) {
					tmp_matches.push_back(forward);
					findCrossCheck = true;
					break;
				}
			}
			if (findCrossCheck)
				break;
		}
	}

	cout << "matches : " << tmp_matches.size() << endl;
	double min_dist = DBL_MAX;
	for (int i = 0; i < (int) tmp_matches.size(); i++) {
		double dist = tmp_matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;
	}

	cout << "min dist :" << min_dist << endl;

	//  対応点間の移動距離による良いマッチングの取捨選択
	matches->clear();
	pt1->clear();
	pt2->clear();
	for (int i = 0; i < (int) tmp_matches.size(); i++) {
		if (round((*key1)[tmp_matches[i].queryIdx].class_id) == round(
				(*key2)[tmp_matches[i].trainIdx].class_id)) {
			if (tmp_matches[i].distance > 0 && tmp_matches[i].distance
					< (min_dist +0.1)) {
				//&&	fabs(((*key1)[tmp_matches[i].queryIdx]).pt.y - ((*key2)[tmp_matches[i].trainIdx]).pt.y) < 100){
				/// fabs((*key1)[tmp_matches[i].queryIdx].pt.x - 	(*key2)[tmp_matches[i].trainIdx].pt.x) < 0.1) {

				matches->push_back(tmp_matches[i]);
				pt1->push_back((*key1)[tmp_matches[i].queryIdx].pt);
				pt2->push_back((*key2)[tmp_matches[i].trainIdx].pt);
				//good_objectKeypoints.push_back(
				//		objectKeypoints[tmp_matches[i].queryIdx]);
				//good_imageKeypoints.push_back(
				//		imageKeypoints[tmp_matches[i].trainIdx]);
			}
		}
	}
}

int main(int argc, char** argv) {

	//ここからフレーム合成プログラム

	VideoCapture frame_cap; // target frame movie
	VideoCapture pano_cap; // background frame movie
	FileStorage cvfs("log.xml", CV_STORAGE_READ);

	Mat panorama, aim_frame, near_frame;
	Mat homography;
	unsigned long n_aim_frame = 1; // target frame number
	long offset_sec; // target video offset
	long s_pano; // panorama frame time
	long n_pano_start_frame;
	//	string str_pano, str_frame, str_video;
	string str_pano_video, str_pano_time, str_pano_ori;
	string str_aim_video, str_aim_time, str_aim_ori;
	string str_pano;
	Mat transform_image; // 画像単体での変換結果
	Mat transform_image2 = Mat(Size(PANO_W, PANO_H), CV_8UC3);

	Mat mask = Mat(Size(PANO_W, PANO_H), CV_8U, Scalar::all(0)); // パノラマ画像のマスク
	Mat pano_black = Mat(Size(PANO_W, PANO_H), CV_8U, Scalar::all(0)); // パノラマ画像と同じサイズの黒画像
	Mat white_img;//= Mat(Size(1280, 720), CV_8U, Scalar::all(255)); // フレームと同じサイズの白画像
	Mat gray_img1, gray_img2;
	Mat mask2;

	vector<Point2f> pt1, pt2;
	//	fstream fs(cam_data.txt);
	//	fs >> str_video;

	// 対応点の対の格納先
	std::vector<cv::DMatch> matches; // matcherにより求めたおおまかな対を格納

	// 特徴点の集合と特徴量
	std::vector<KeyPoint> objectKeypoints, imageKeypoints;
	Mat objectDescriptors, imageDescriptors;

	string algorithm_type("SURF");
	Ptr<Feature2D> feature;

	feature = Feature2D::create(algorithm_type);
	if (algorithm_type.compare("SURF") == 0) {
		feature->set("extended", 1);
		feature->set("hessianThreshold", 5);
		feature->set("nOctaveLayers", 4);
		feature->set("nOctaves", 3);
		feature->set("upright", 0);
	}
	if (argc != 4) {
		//	cout << "Usage : " << argv[0] << "target_frame_video"
		//			<< "target_frame_number" << "panorama_image_path"
		//			<< "panorama_src_video_path" << "offset_sec"
		//			<< "TIME_file_path" << "target_ORI_FILE_path"
		//			<< "panorama_frame_start_number" << "panorama_ORI_FILE_path"
		//			<< endl;
		cout << "Usage : " << argv[0] << "target_frame_image "
				<< "panorama_cam_data" << "panorama_background_image_path"
				<< "target_frame_number" << " offset_sec"
				<< "panorama_frame_start_number" << endl;
		return -1;
	}

	ifstream ifs_aimcam(argv[1]);
	ifstream ifs_panocam(argv[2]);

	if (!ifs_aimcam.is_open()) {
		cerr << "cannnot open target_cam_data : " << argv[1] << endl;
		return -1;
	}
	if (!ifs_panocam.is_open()) {
		cerr << "cannnot open panorama_cam_data : " << argv[2] << endl;
		return -1;
	}

	// パノラマ背景の元動画，TIMEファイル，ORIファイルのパスを取得
	ifs_panocam >> str_pano_video;
	ifs_panocam >> str_pano_time;
	ifs_panocam >> str_pano_ori;

	// はめ込むフレームを含む動画，TIMEファイル，ORIファイルのパスを取得
	ifs_aimcam >> str_aim_video;
	ifs_aimcam >> str_aim_time;
	ifs_aimcam >> str_aim_ori;

	// パノラマ背景画像のファイル名を取得
	str_pano = argv[3];

	// はめ込むフレーム画像の大まかなオフセット[sec]offset_secと，そこからの相対フレーム番号n_aim_frame
	//n_aim_frame = atoi(argv[4]);
	//offset_sec = atoi(argv[5]);

	// パノラマ背景合成時における初期フレーム番号
	//n_pano_start_frame = atoi(argv[6]);


	// パノラマ背景画像の読み込み
	panorama = imread(str_pano);
	if (panorama.empty()) {
		cerr << "cannot open panorama image" << endl;
		return -1;
	}
	cout << "load background image" << endl;

	// 合成フレームを含む動画ファイルをオープン
	/*frame_cap.open(str_aim_video);
	if (!frame_cap.isOpened()) {
		cerr << "cannnot open frame movie" << endl;
		return -1;
	}
	*/
	aim_frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);  // 直接合成フレームを指定する場合argv[1]に画像pathを指定する

	// パノラマ背景の元動画をオープン
	pano_cap.open(str_pano_video);
	if (!pano_cap.isOpened()) {
		cerr << "cannnot open panorama src movie" << endl;
		return -1;
	}

	//　パノラママスク画像の読み込み
	mask = imread("mask.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cout << "load background mask image" << endl;


	cout << "load target frame image" << endl;
/*
	// 合成フレームの位置時間を計算し，画像を取得
	double aim_frame_time= (((double)n_aim_frame / (double)TARGET_VIDEO_FPS) + (double)offset_sec) * 1000.0;
	if(frame_cap.set(CV_CAP_PROP_POS_FRAMES, aim_frame_time)){
		cout << "Success !!" << endl;
	}
	cout << "target frame time : " << aim_frame_time << " [msec]" << endl;
	frame_cap >> aim_frame;
*/
	if(aim_frame.empty()){
		cerr << "cannnot load target frame"<<endl;
	}

	//imshow("aim_frame",aim_frame);
	//waitKey(0);

	pano_cap.set(CV_CAP_PROP_POS_FRAMES, s_pano);
	double pano_frame_time = pano_cap.get(CV_CAP_PROP_POS_MSEC);

	//cout << "aim_frame_time : " << aim_frame_time << endl;
	//cout << "pano_frame_time : " << pano_frame_time << endl;

	//cout << "load target frame image : " << n_aim_frame << endl;


	if (aim_frame.empty()) {
		cerr << "cannnot open target frame" << endl;
		return -1;
	}
	// create white_img
	white_img = Mat(aim_frame.size(), CV_8U, Scalar::all(255));

	namedWindow("aim", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	imshow("aim", aim_frame);
	//imwrite("target.jpg", aim_frame);
	waitKey(30);
	namedWindow("panorama", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	imshow("panorama", panorama);
	waitKey(30);
	Mat A1Matrix, A2Matrix;
	Mat yaw = Mat::eye(3, 3, CV_64FC1);
	Mat roll = cv::Mat::eye(3, 3, CV_64FC1);
	Mat pitch = cv::Mat::eye(3, 3, CV_64FC1);

	Mat hh = Mat::eye(3, 3, CV_64FC1);
	A1Matrix = Mat::eye(3, 3, CV_64FC1);

	A1Matrix.at<double> (0, 0) = 900;
	A1Matrix.at<double> (1, 1) = 900;
	A1Matrix.at<double> (0, 2) = PANO_W / 2;
	A1Matrix.at<double> (1, 2) = PANO_H / 2;
	cout << "a" << endl;
	SetYawRotationMatrix(&yaw, 0);
	A2Matrix = A1Matrix.clone();
	A1Matrix = A1Matrix.inv();
	//hh = A2Matrix * yaw * A1Matrix;
	cout << hh << endl;
	//transform_image2 = panorama.clone();

	warpPerspective(panorama, transform_image2, hh, Size(PANO_W, PANO_H),
			CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS); // 先頭フレームをパノラマ平面へ投影
	//	make_pano(panorama, transform_image2, mask, Mat(Size(PANO_W, PANO_H), CV_8UC3, Scalar::all(255)));
	imshow("panorama", transform_image2);
	waitKey(30);

	// 合成したいフレームのORIからパノラマ平面へのホモグラフィを計算し重なり具合を計算
	/*string str_time(argv[6]);
	string time_buf;
	ifstream ifs_time(str_time.c_str());
	double s_time;

	if (!ifs_time.is_open()) {
		cerr << "cannnot open TIME file : " << str_time << endl;
		return -1;
	}

	ifs_time >> time_buf;
	ifs_time >> time_buf;
	ifs_time >> time_buf;
	ifs_time >> s_time; // msec

	s_time /= 1000.0;
	cout << "s_time : " << s_time << endl;
	SENSOR_DATA *sensor = (SENSOR_DATA *) malloc(sizeof(SENSOR_DATA) * 5000);

	// 対象フレームのセンサデータを一括読み込み
	LoadSensorData(argv[7], &sensor);

	// 対象フレームの動画の頭からの時間frame_timeに撮影開始時刻s_timeを加算して，実時間に変換
	aim_frame_time += s_time;
	SENSOR_DATA pano_sd, aim_sd;
	GetSensorDataForTime(aim_frame_time, &sensor, &aim_sd);

	cout << "yaw : " << aim_sd.alpha << " pitch : " << aim_sd.beta
			<< " roll : " << aim_sd.gamma << endl;

	// パノラマの時間を取得
	ifs_time.close();
	ifs_time.open(str_pano.c_str());
	if (!ifs_time.is_open()) {
		cerr << "cannnot open TIME file : " << str_time << endl;
		return -1;
	}

	ifs_time >> time_buf;
	ifs_time >> time_buf;
	ifs_time >> time_buf;
	ifs_time >> s_time; // msec

	s_time /= 1000.0;
	cout << "s_time : " << s_time << endl;

	// センサデータを一括読み込み
	LoadSensorData(argv[9], &sensor);

	// 対象フレームの動画の頭からの時間frame_timeに撮影開始時刻s_timeを加算して，実時間に変換
	pano_frame_time += s_time;

	GetSensorDataForTime(aim_frame_time, &sensor, &aim_sd);

	cout << "yaw : " << aim_sd.alpha << " pitch : " << aim_sd.beta
			<< " roll : " << aim_sd.gamma << endl;
*/
	// GCの内部の逆
	A1Matrix.at<double> (0, 0) = 1.1107246554597004e-003;
	A1Matrix.at<double> (0, 2) = -7.0476759105087217e-001;
	A1Matrix.at<double> (1, 1) = 1.0937849972977411e-003;
	A1Matrix.at<double> (1, 2) = -4.2040554903081440e-001;

	// AQの内部の逆
	//A1Matrix.at<double> (0, 0) = 8.1632905612490970e-004;
	//A1Matrix.at<double> (0, 2) = -5.2593546441192318e-001;
	//A1Matrix.at<double> (1, 1) = 8.1390778599629236e-004;
	//A1Matrix.at<double> (1, 2) = -2.7706041350882804e-001;

	Mat inv_a1 = A1Matrix.inv();
	A2Matrix.at<double> (0, 0) = inv_a1.at<double> (0, 0);
	A2Matrix.at<double> (1, 1) = inv_a1.at<double> (1, 1);
	A2Matrix.at<double> (0, 2) = PANO_W / 2;
	A2Matrix.at<double> (1, 2) = PANO_H / 2;

	//SetRollRotationMatrix(&roll, aim_sd.gamma);
	//SetPitchRotationMatrix(&pitch, aim_sd.beta);
	//SetYawRotationMatrix(&yaw, aim_sd.alpha);

	// パノラマ画像と合成したいフレームの特徴点抽出と記述
	cout << "calc features" << endl;
	cvtColor(aim_frame, gray_img1, CV_RGB2GRAY);
	cvtColor(panorama, gray_img2, CV_RGB2GRAY);
	erode(mask, mask2, cv::Mat(), cv::Point(-1, -1), 50);
	feature->operator ()(gray_img1, Mat(), imageKeypoints, imageDescriptors);
	feature->operator ()(gray_img2, mask2, objectKeypoints, objectDescriptors);

	//cout << imageDescriptors << " " << objectDescriptors << endl;
	cout << imageKeypoints[1].pt << " " << objectKeypoints[1].pt << endl;

	//良い対応点の組みを求める
	good_matcher(imageDescriptors, objectDescriptors, &imageKeypoints,
			&objectKeypoints, &matches, &pt1, &pt2);
	cout << "selected good_matches : " << pt1.size() << endl;
	Mat result, r_result;
	//mask = Mat(Size(PANO_W, PANO_H),CV_8UC3,Scalar::all(0));

	cout << "make drawmathces image" << endl;

	//ホモグラフィ行列を計算
	homography = findHomography(Mat(pt1), Mat(pt2), CV_RANSAC, 5.0);

	//合成したいフレームをパノラマ画像に乗るように投影
	warpPerspective(aim_frame, transform_image, homography,
			Size(PANO_W, PANO_H), CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);

	//投影場所のマスク生成
	warpPerspective(white_img, pano_black, homography, Size(PANO_W, PANO_H),
			CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
	namedWindow("panoblack", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	//waitKey(0);

	FileNode node(cvfs.fs, NULL); // Get Top Node

	int frame_num;
	Mat h_base;
	Mat tmp_base;
	stringstream ss;
	long n = 1;
	Mat aria1, aria2;
	Mat v1, v2;
	long max = LONG_MIN;
	long detect_frame_num;
	aria1 = Mat(Size(PANO_W, PANO_H), CV_8U, Scalar::all(0));
	aria2 = Mat(Size(PANO_W, PANO_H), CV_8U, Scalar::all(0));

	while (1) {
		//		frame_num = node["frame"];
		for (; n < pano_cap.get(CV_CAP_PROP_FRAME_COUNT); n++) {
			ss << "homo_" << n;
			read(node[ss.str()], tmp_base);
			if (!tmp_base.empty())
				break;
			ss.clear();
			ss.str("");
		}

		//ここに来たときtmp_baseに取得した行列，nにフレーム番号が格納されている
		// tmp_baseが空なら取り出し終わっているので下でブレイク処理に入る


		if (tmp_base.empty())
			break;
		warpPerspective(white_img, aria1, homography, Size(PANO_W, PANO_H),
				CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);

		warpPerspective(white_img, aria2, tmp_base, Size(PANO_W, PANO_H),
				CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
		bitwise_and(aria1, aria2, aria1);
		namedWindow("mask", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		imshow("mask", aria1);

		threshold(aria1, aria2, 128, 1, THRESH_BINARY);

		long sum;
		sum = 0;
		for (unsigned long i = 0; i < aria2.rows * aria2.cols; i++)
			sum += aria2.data[i];
		if (sum > max) {
			max = sum;
			h_base = tmp_base.clone();
			detect_frame_num = n;
		}
		sum = 0;

	}
	//free(sensor);

	cout << "detected near frame : " << detect_frame_num << endl;
	cout << max << endl;
	//　重なりが大きいフレームを取り出す
	pano_cap.set(CV_CAP_PROP_POS_FRAMES, detect_frame_num);
	pano_cap >> near_frame;
	if (near_frame.empty()) {
		cerr << "cannot detect near frame" << endl;
		return -1;
	}
	cvtColor(near_frame, gray_img2, CV_RGB2GRAY);
	feature->operator ()(gray_img2, Mat(), objectKeypoints, objectDescriptors);
	good_matcher(imageDescriptors, objectDescriptors, &imageKeypoints,
			&objectKeypoints, &matches, &pt1, &pt2);
	drawMatches(aim_frame, imageKeypoints, near_frame, objectKeypoints,
			matches, result);
	resize(result, r_result, Size(), 0.5, 0.5, INTER_LANCZOS4);

	cout << "show matches" << endl;
	namedWindow("matches", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	imshow("matches", result);
	waitKey(30);
	homography = findHomography(Mat(pt1), Mat(pt2), CV_RANSAC, 5.0);
	warpPerspective(white_img, pano_black, h_base * homography, Size(PANO_W,
			PANO_H), CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
	warpPerspective(aim_frame, transform_image, h_base * homography, Size(
			PANO_W, PANO_H), CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);

	namedWindow("transform_image", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	imshow("transform_image", transform_image);
	waitKey(30);

	/*
	 //投影先での対応点を再計算
	 bitwise_and(mask, pano_black, mask);
	 erode(mask, mask2, cv::Mat(), cv::Point(-1, -1), 30);
	 feature->operator ()(gray_img2, mask2, objectKeypoints, objectDescriptors);
	 good_matcher(imageDescriptors, objectDescriptors, &imageKeypoints,
	 &objectKeypoints, &matches, &pt1, &pt2);
	 homography = findHomography(Mat(pt1), Mat(pt2), CV_RANSAC, 5.0);
	 warpPerspective(white_img, pano_black, homography, Size(PANO_W, PANO_H),
	 CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
	 drawMatches(aim_frame, imageKeypoints, panorama, objectKeypoints, matches,
	 result);
	 resize(result, r_result, Size(), 0.5, 0.5, INTER_LANCZOS4);

	 cout << "show matches" << endl;
	 namedWindow("matches", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	 imshow("matches", result);
	 waitKey(30);
	 */

	bitwise_and(mask, pano_black, mask);
	imshow("panoblack", transform_image2);
	waitKey(30);

	make_pano(transform_image, transform_image2, ~mask, pano_black);

	namedWindow("result", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	imshow("result", transform_image2);
	waitKey(30);
	imwrite("result.jpg", transform_image2);

	return 0;
}
