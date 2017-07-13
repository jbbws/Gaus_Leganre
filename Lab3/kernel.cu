
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <math.h>
#include <stdio.h>

__constant__  int _stepLeganre;
__constant__ int _countQuadr;
__constant__ float _node3GPU[3];
__constant__ float _koef3GPU[3];

__constant__ float _node4GPU[4];
__constant__ float _koef4GPU[4];

const float node3step[3] = { -0.7745967f, 0, 0.7745967f };
const float koef3step[3] = { 0.5555556f, 0.8888889f, 0.5555556f };

const float node4step[4] = { -0.8611363f, -0.3399810f, 0.3399810f, 0.8611363f };
const float koef4step[4] = { 0.3478548f, 0.6521451f, 0.6521451f, 0.3478548f };

__device__ __host__ float IntegralFunction(float x)
{
	return 31*x + log(5*x)+5;
}

__global__ void GaussLeganre(float* left, float *right, float* summ)
{
	int thr = blockIdx.x + threadIdx.x;
	float _s = 0;
	for (int i = 0; i < 3; i++)
	{
		float _v = _node3GPU[i] * (right[thr] - left[thr]) / 2 + (right[thr]+left[thr])/2;
		_s += _koef3GPU[i] * IntegralFunction(_v);
	}
	summ[thr] = _s*(right[thr]-left[thr])/2;
}




void initArrays(float left[], float right[],int size, int leftInt,int rightInt)
{
	float h = (rightInt - leftInt)/size;
	left[0] = leftInt;
	right[0] = left[0] + h;
	for (int i = 0; i < size; i++)
	{
		left[i] = leftInt + i*h;
		right[i] = left[i] + h;
	}
}
void SequenceIntegral(float *left, float *right, float *summ, int Quadr)
{
	for (int c = 0; c < Quadr; c++)
	{
		summ[c] = 0;
		for (int i = 0; i < 3; i++)
		{
			float _v = node3step[i] * (right[c] - left[c]) / 2 + (right[c] + left[c]) / 2;
			summ[c] += koef3step[i] * IntegralFunction(_v);
		}
		summ[c] = summ[c] * (right[c] - left[c]) / 2;
	}
}

int main()
{
#pragma region Variables declaration
	 
	cudaEvent_t start;
	cudaEvent_t stop;

    const int arraySize = 5;
	const int countQuandr = 500;
	const int stepLeganre = 3;

	int stInt = 0, fnInt = 100000;
	
	float ArrayLeftNodes[countQuandr];
	float ArrayRightNodes[countQuandr];
	float QuadroSum[countQuandr];
	float Result;

	float* _devLeftArray = 0;
	float* _devRightArray = 0;
	float* _summ = 0;
	float _cudatime = 0;
#pragma endregion	

	initArrays(ArrayLeftNodes, ArrayRightNodes, countQuandr, stInt, fnInt);

#pragma region Copying data from host to device	

	cudaMalloc((void**)&_devLeftArray, countQuandr * sizeof(float));
	cudaMalloc((void**)&_devRightArray, countQuandr * sizeof(float));
	cudaMalloc((void**)&_summ, countQuandr * sizeof(float));


	cudaMemcpy(_devLeftArray, ArrayLeftNodes, countQuandr * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(_devRightArray, ArrayRightNodes, countQuandr * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(_summ, QuadroSum, countQuandr * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(_node3GPU, node3step, sizeof(node3step), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_koef3GPU, koef3step, sizeof(koef3step), 0, cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(_node4GPU, node4step,  sizeof(node4step), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_koef4GPU, koef4step, sizeof(koef4step), 0, cudaMemcpyHostToDevice);

#pragma endregion	
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	GaussLeganre <<<1,countQuandr>> >(_devLeftArray, _devRightArray, _summ);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
		
	cudaMemcpy(QuadroSum, _summ, countQuandr * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventElapsedTime(&_cudatime, start, stop);

#pragma region Summnig CUDA

	Result = 0;
	for (int i = 0; i < countQuandr; i++)
	{
		Result += QuadroSum[i];
	}
#pragma endregion
	printf("TOTAL CUDA SUMM: %f\n", Result);
	printf("TOTAL CUDA TIME: %f\n", _cudatime);
#pragma region Sequence
	float SeqSum[countQuandr];
	float stSeq = clock() / (float)CLOCKS_PER_SEC;
	SequenceIntegral(ArrayLeftNodes, ArrayRightNodes, SeqSum, countQuandr);
	float fnSeq = clock() / (float)CLOCKS_PER_SEC;
	Result = 0;
	for (int i = 0; i < countQuandr; i++)
	{
		Result += SeqSum[i];
	}
	printf("TOTAL SEQUENCE SUMM: %f\n", Result);
	printf("TOTAL SEQUENCE TIME: %f\n", fnSeq-stSeq);

#pragma endregion

	getchar();

 return 0;
}




