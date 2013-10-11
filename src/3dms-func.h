//
// 3dms-func.h
//

#define MAXDATA_3DMS  5000

typedef struct{
    double alpha, beta, gamma, north;
    //double Accx, Accy, Accz;
    //int wAccx, wAccy, wAccz;
    //int wGyrx, wGyry, wGyrz;
    //int wMagx, wMagy, wMagz;
    //double HH,MM,SS;
    double TT;
}SENSOR_DATA;

// １つのセンサデータを表示．現在は時刻とα, β, γ, α-north を表示
int DispSensorData(SENSOR_DATA sd);

// センサデータファイルからの読み込み
int LoadSensorData(char *oridatafile ,SENSOR_DATA *sd_array[]);
//int LoadSensorData(char *timedatafile,char *accdatafile,char *magdatafile,char *oridatafile , SENSOR_DATA *sd_array[]);

// センサデータを補間して時刻のパラメータを算出する
int GetSensorDataForTime(double TT, SENSOR_DATA *in_sd_array[], SENSOR_DATA *sd);
