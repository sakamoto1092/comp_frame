//
// 3dms-func.c
//

#include <stdio.h>
#include <stdlib.h>

#include "3dms-func.h"

// １つのセンサデータを表示．現在は時刻とα, β, γ, α-north を表示
int DispSensorData(SENSOR_DATA sd)
{
    fprintf(stderr, "%f %f %f %f ",
              sd.alpha, sd.beta, sd.gamma, sd.north);
    fprintf( stderr, "%f\n", sd.TT);
    
    return 0;
}

/* flash.dat の読み込み */
int LoadSensorData(char *oridatafile,SENSOR_DATA *sd_array[])
{ 
    FILE *fp_ori;

    double t1,t2,t3;

    int i;
    char cdum[256];
    
    SENSOR_DATA *p;

     if ( ( fp_ori = fopen( oridatafile, "r" ) ) == NULL ){
        fprintf( stderr , "\nError : Cannot open %s\n\n", oridatafile );
        exit(0);
    }
   
    //fgets( cdum, 1024, fp); // ３行読み飛ばす
    //fgets( cdum, 1024, fp); // ３行読み飛ばす
    //fgets( cdum, 1024, fp); // ３行読み飛ばす
    
    p = sd_array[0];
    for (i=0; i< MAXDATA_3DMS; i++) {
        fscanf( fp_ori, "%lf,%lf,%lf,%lf",
                &t1,&(p->alpha), &(p->beta), &(p->gamma));
        p->north=p->alpha;;
        p->TT = t1/1000; // 絶対時刻(秒)の算出
        p++;
    }  
    fclose(fp_ori);
    
    return 0;
}

// センサデータを補間して時刻のパラメータを算出する
int GetSensorDataForTime(double TT,  // 時刻情報
                         SENSOR_DATA *in_sd_array[], // 入力データ(配列)
                         SENSOR_DATA *out_sd         // 出力データ(１個分)
                         )
{
    int i=0;
    double s;
    SENSOR_DATA *sd0, *sd1;
    int flag=0;
    
    sd0 = in_sd_array[0];
    sd1 = sd0 + 1;
    
    while (sd0->TT < TT) {
        flag = 1;
        sd0++;
        sd1++;
        i++;
        if (i > MAXDATA_3DMS) {
            fprintf(stderr, "OVER MAXDATA\n");
            exit(0);
        }
    }
    
    if (flag==0) {
        out_sd->alpha = sd0->alpha;
        out_sd->beta = sd0->beta;
        out_sd->gamma = sd0->gamma;
        out_sd->north = sd0->north;
        out_sd->TT = sd0->TT;
        return 0;
    }

    sd1 = sd0;
    sd0--;

    //fprintf(stderr, "(%f) < [%f] < (%f)\n",
    //        sd0->TT, TT,  sd1->TT);

    s = (TT - sd0->TT)/(sd1->TT - sd0->TT);
    if(sd0->alpha>180)sd0->alpha=sd0->alpha-360;
    if(sd1->alpha>180)sd1->alpha=sd1->alpha-360;
    if(sd0-> beta>180)sd0-> beta=sd0-> beta-360;
    if(sd1-> beta>180)sd1-> beta=sd1-> beta-360;
    if(sd0->gamma>180)sd0->gamma=sd0->gamma-360;
    if(sd1->gamma>180)sd1->gamma=sd1->gamma-360;

    out_sd->alpha = (1-s)*sd0->alpha + s*sd1->alpha;
    out_sd->beta  = (1-s)*sd0->beta  + s*sd1->beta;
    out_sd->gamma = (1-s)*sd0->gamma + s*sd1->gamma;
    out_sd->north = (1-s)*sd0->north + s*sd1->north;

    //DispSensorData(*out_sd);

    return 0;

}
