#include <iostream>
#include <cstdarg>
#include <cmath>
using namespace std;

#define RATE    0.8

#define HIGH_TARGET_OUT 0.80
#define LOW_TARGET_OUT  0.20
#define DVALUE  ((HIGH_TARGET_OUT-LOW_TARGET_OUT)-0.05)

typedef struct
{
    float* fea;
    int targetVal; //二值，如1和-1；未知为0
} Samp;

typedef struct
{
    int sizeOfSet;
    int numOfFeature;
} Param;

typedef struct
{
    int hNum;
    int oNum;
} BPParam;

float yita = 0.05;

/* 感知器的表征能力 */
/* 表示布尔函数 */
int AND(int n,...)
{
    va_list vl;
    va_start(vl,n);
    int sum = 0;
    for(int i=0; i<n; i++)
    {
        int var = va_arg(vl,int);
        sum += 10 * var;
    }
    return (sum > n * 10 - 5) ? 1 : 0;
}

int OR(int n,...)
{
    va_list vl;
    va_start(vl,n);
    int sum = 0;
    for(int i=0; i<n; i++)
    {
        int var = va_arg(vl,int);
        sum += 10 * var;
    }
    return (sum > 5) ? 1 : 0;
}

int NAND(int n,...)
{
    va_list vl;
    va_start(vl,n);
    int sum = 0;
    for(int i=0; i<n; i++)
    {
        int var = va_arg(vl,int);
        sum += 10 * var;
    }
    return (sum > n * 10 - 5) ? 0 : 1;
}

int NOR(int n,...)
{
    va_list vl;
    va_start(vl,n);
    int sum = 0;
    for(int i=0; i<n; i++)
    {
        int var = va_arg(vl,int);
        sum += 10 * var;
    }
    return (sum > 5) ? 0 : 1;
}

int NOT(int x)
{
    return 0==x ? 1 : 0;
}

/* 三位二进制编码器，输入7位，输出3位 */
void ThreeBinaryEncoder(int* in,int* out)
{
    out[0] = NAND(4,NOT(in[0]),NOT(in[2]),NOT(in[4]),NOT(in[6]));
    out[1] = NAND(4,NOT(in[1]),NOT(in[2]),NOT(in[5]),NOT(in[6]));
    out[2] = NAND(4,NOT(in[3]),NOT(in[4]),NOT(in[5]),NOT(in[6]));
}

/* D触发器 */
int DTrigger(int d,int cp)
{
    static int q;
    if(0==d && 0==cp)
        return q;
    if(1==d && 0==cp)
        return q;
    if(0==d && 1==cp)
    {
        q = 0;
        return q;
    }
    if(1==d && 1==cp)
    {
        q = 1;
        return q;
    }
    return -1;
}

//感知器训练法则
void TrainPerceptron(Samp* samp, float* W, Param& param)
{
    bool isSatisfied;
    do
    {
        isSatisfied = true;
        for(int i=0; i<param.sizeOfSet; i++)
        {
            //计算线性组合
            float sum = W[0];
            for(int j=0; j<param.numOfFeature; j++)
            {
                sum += W[j+1] * samp[i].fea[j];
            }
            int o = sum>0 ? 1 : -1;

            if(o!=samp[i].targetVal)
            {
                isSatisfied = false;
                //更新权值
                W[0] += yita * (samp[i].targetVal-o) * 1;
                for(int j=0; j<param.numOfFeature; j++)
                {
                    W[j+1] += yita * (samp[i].targetVal-o) * samp[i].fea[j];
                }
            }
        }
    }
    while(!isSatisfied);
}

void PerceptronTest(Samp* testSamp, float* W, Param& param)
{
    for(int i=0; i<param.sizeOfSet; i++)
    {
        float sum = W[0];
        for(int j=0; j<param.numOfFeature; j++)
        {
            sum += W[j+1]*testSamp[i].fea[j];
        }
        int o = sum>0 ? 1 : -1;
    }
}

/* 终止条件 */
bool IsSatisfied(Samp* samp, float* W, Param& param)
{
    int cor=0,wr=0;
    for(int i=0; i<param.sizeOfSet; i++)
    {
        float sum = W[0];
        for(int j=0; j<param.numOfFeature; j++)
        {
            sum += W[j+1] * samp[i].fea[j];
        }
        int o = sum>0 ? 1 : -1;
        (o==samp[i].targetVal) ? cor++ : wr++;
    }
    float r = cor/(float)param.sizeOfSet;
    return (r >= RATE) ? true : false;
}

/* 梯度下降 */
void GradientDescent(Samp* samp, float* W, Param& param)
{
    float* dw = new float[param.numOfFeature+1];
    while(!IsSatisfied(samp,W,param))
    {
        for(int t=0; t<param.numOfFeature+1; t++)
            dw[t] = 0;
        for(int i=0; i<param.sizeOfSet; i++)
        {
            float out = W[0];
            for(int j=0; j<param.numOfFeature; j++)
            {
                out += W[j+1] * samp[i].fea[j];
            }
            dw[0] += yita * (samp[i].targetVal - out) * 1;
            for(int t=0; t<param.numOfFeature; t++)
            {
                dw[t+1] += yita * (samp[i].targetVal - out) * samp[i].fea[t];
            }
        }
        for(int k=0; k<param.numOfFeature+1; k++)
        {
            W[k] += dw[k];
        }
    }
    delete[] dw;
}

/* Delta法则(随机梯度下降) */
void DeltaRule(Samp* samp, float* W, Param& param)
{
    while(!IsSatisfied(samp,W,param))
    {
        for(int i=0; i<param.sizeOfSet; i++)
        {
            float out = W[0];
            for(int j=0; j<param.numOfFeature; j++)
            {
                out += W[j+1] * samp[i].fea[j];
            }
            W[0] += yita * (samp[i].targetVal - out) * 1;
            for(int k=0; k<param.numOfFeature; k++)
            {
                W[k+1] += yita * (samp[i].targetVal - out) * samp[i].fea[k];
            }
        }
    }
}

float Sigmoid(float a)
{
    return 1.0 / (1 + exp(-a));
}

bool IsSatisfiedBP(Samp* samp, Param& param, float** WX, float** Wh, BPParam& bpParam)
{
    int cor=0,wr=0;
    float* hOut = new float[bpParam.hNum];
    float* out = new float[bpParam.oNum];
    for(int i=0; i<param.sizeOfSet; i++)
    {
        //处理每一个隐层结点
        for(int k=0; k<bpParam.hNum; k++)
        {
            //计算输入层输出的线性组合
            float hIn = WX[k][0];
            for(int j=0; j<param.numOfFeature; j++)
            {
                hIn += WX[k][j+1]*samp[i].fea[j];
            }
            //计算隐层输出
            hOut[k] = Sigmoid(hIn);
        }
        //处理每一个输出结点
        for(int t=0; t<bpParam.oNum; t++)
        {
            //计算隐层输出的线性组合
            float oIn = 0;
            for(int m=0; m<bpParam.hNum; m++)
            {
                oIn += Wh[t][m]*hOut[m];
            }
            //计算输出层输出
            out[t] = Sigmoid(oIn);
        }
        if(samp[i].targetVal==1)
        {
            if((out[0]-out[1])>=DVALUE)
                cor++;
            else
                wr++;
        }
        else
        {
            if((out[1]-out[0])>=DVALUE)
                cor++;
            else
                wr++;
        }
    }
    delete[] hOut;
    delete[] out;
    float r = cor/(float)param.sizeOfSet;
    return (r >= RATE) ? true : false;
    //return false;
}

void Print(float** WX, float** Wh, BPParam& bpParam ,int feaNum)
{
    cout<<"输入层到隐层："<<endl;
    for(int i=0; i<bpParam.hNum; i++)
    {
        for(int j=0; j<feaNum; j++)
        {
            cout<<WX[i][j]<<' ';
        }
        cout<<endl;
    }
    cout<<"隐层到输出层"<<endl;
    for(int i=0; i<bpParam.oNum; i++)
    {
        for(int j=0; j<bpParam.oNum; j++)
        {
            cout<<Wh[i][j]<<' ';
        }
        cout<<endl;
    }
}

/*
反向传播算法，delta法则版本
ANN：输入层-隐层-输出层
输出设置：若target=1,out[0]目标输出HIGH_TARGET_OUT，out[1]目标输出LOW_TARGET_OUT;否则,反之
*/
void BackPropagation(Samp* samp, Param& param, float** WX, float** Wh, BPParam& bpParam)
{
    float* hOut = new float[bpParam.hNum];
    float* out = new float[bpParam.oNum];
    float* hErr = new float[bpParam.hNum];
    float* oErr = new float[bpParam.oNum];
    while(!IsSatisfiedBP(samp,param,WX,Wh,bpParam))
    {
        //处理每一个训练样例
        for(int i=0; i<param.sizeOfSet; i++)
        {
            /* 前向传播输入 */
            //处理每一个隐层结点
            for(int k=0; k<bpParam.hNum; k++)
            {
                //计算输入层输出的线性组合
                float hIn = WX[k][0];
                for(int j=0; j<param.numOfFeature; j++)
                {
                    hIn += WX[k][j+1]*samp[i].fea[j];
                }
                //计算隐层输出
                hOut[k] = Sigmoid(hIn);
            }
            //处理每一个输出结点
            for(int t=0; t<bpParam.oNum; t++)
            {
                //计算隐层输出的线性组合
                float oIn = 0;
                for(int m=0; m<bpParam.hNum; m++)
                {
                    oIn += Wh[t][m]*hOut[m];
                }
                //计算输出层输出
                out[t] = Sigmoid(oIn);
            }
            /* 反向传播误差 */
            //设置输出目标
            float t[2];
            (samp[i].targetVal==1) ?
            (t[0]=HIGH_TARGET_OUT,t[1]=LOW_TARGET_OUT) :
            (t[0]=LOW_TARGET_OUT,t[1]=HIGH_TARGET_OUT);
            //计算输出单元的误差项
            for(int j=0; j<bpParam.oNum; j++)
            {
                oErr[j] = out[j]*(1-out[j])*(t[j]-out[j]);
            }
            //计算隐层单元的误差项
            for(int k=0; k<bpParam.hNum; k++)
            {
                //间接计算误差值(t-o)
                float h2o = 0;
                for(int m=0; m<bpParam.oNum; m++)
                {
                    h2o += oErr[m]*Wh[m][k];
                }
                //计算误差项
                hErr[k] = hOut[k]*(1-hOut[k])*h2o;
            }
            /* 更新权值 */
            //输入层到隐层
            for(int t=0; t<bpParam.hNum; t++)
            {
                WX[t][0] += yita*hErr[t]*1;
                for(int s=0; s<param.numOfFeature; s++)
                {
                    WX[t][s+1] += yita*hErr[t]*samp[i].fea[s];
                }
            }
            //隐层到输出层
            for(int t=0; t<bpParam.oNum; t++)
            {
                for(int s=0; s<bpParam.hNum; s++)
                {
                    Wh[t][s] += yita*oErr[t]*hOut[s];
                }
            }
        }
        Print(WX,Wh,bpParam,param.numOfFeature);
    }
    delete[] hOut;
    delete[] out;
    delete[] hErr;
    delete[] oErr;
}
