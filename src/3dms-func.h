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

// ���ĤΥ��󥵥ǡ�����ɽ�������ߤϻ���Ȧ�, ��, ��, ��-north ��ɽ��
int DispSensorData(SENSOR_DATA sd);

// ���󥵥ǡ����ե����뤫����ɤ߹���
int LoadSensorData(char *oridatafile ,SENSOR_DATA *sd_array[]);
//int LoadSensorData(char *timedatafile,char *accdatafile,char *magdatafile,char *oridatafile , SENSOR_DATA *sd_array[]);

// ���󥵥ǡ�������֤��ƻ���Υѥ�᡼���򻻽Ф���
int GetSensorDataForTime(double TT, SENSOR_DATA *in_sd_array[], SENSOR_DATA *sd);
