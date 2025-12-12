/**************************************************************

 HPCA PROJECT, SUBJECT 6

***************************************************************/

#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>

#define NB 2048
#define NTPB 1024
#define T 10

// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void init_curand_state_k(curandStateXORWOW_t* states)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(0,id,0,&states[id]);
}

__device__ float calculate_Price(float sigma, float dt,float a,float rs, float s, float t,curandStateXORWOW_t* states)
{
	float sintegral = 0;
	curandStateXORWOW_t localState = states[threadIdx.x + blockIdx.x*blockDim.x];
	float2 G = curand_normal2(&localState);
	int X = (t-s)/dt;
	float s_temp = s;
	float r_temp = rs;
	for(int f = 0; f <= X; f++)
	{
		float integral = 0;
		float w = s + f*dt;
		int N = (t-w)/dt;
		// implements integral term of m
		for(int k = 0; k <= N; k++)
		{	
			float i = w + k*dt;
			integral += dt/2 * expf(-a*(t-i)) * ((i < 5) ? (0.012+0.0014*i) : (0.019+0.001*(i-5))) * ((k != 0 && k != N) ? 2 : 1);
		}
		float m = r_temp * expf(-a*(t-s_temp)) + integral;
		float sigmaBig = sqrt(sigma*sigma*(1-expf(-2*a*(t-s_temp)))/(2*a));
		r_temp = m + sigmaBig*G.x;
		sintegral += dt/2 * r_temp * ((f != 0 && f != X) ? 2 : 1);
		s_temp = w; 
	}
	return sintegral;
}

__global__ void Bond_Price_k(float sigma,float dt, float a,float rs,float s,float t,curandStateXORWOW_t* states,float* Pout,float* Aout)
{
	extern __shared__ float sdata[];
	int tid = threadIdx.x;

	sdata[tid] = expf(-calculate_Price(sigma,dt,a,rs,s,t,states));
	sdata[blockDim.x+tid] =  (calculate_Price(sigma,dt,a,rs,s,t+0.1,states) - calculate_Price(sigma,dt,a,rs,s,t-0.1,states) )/(2*0.1);
	__syncthreads();
	for(int k = blockDim.x/2; k > 0; k /= 2)
	{
		if(tid < k){
			sdata[tid] += sdata[k+tid];
			sdata[blockDim.x + tid] += sdata[k+tid+blockDim.x];
		}
		__syncthreads();
	}

	if(tid == 0){
		atomicAdd(Pout, sdata[0]);
		atomicAdd(Aout, sdata[blockDim.x]);
	}
}

// Theta functions not tested

__device__ float theta(float* f,float sigma,float t,float a,float dt, int maxindex)
{
	int s = roundf(t/dt);
	float result = a*f[s] + (sigma*sigma*(1-expf(-2*a*t))/(2*a)); 

	if(s > 0 && s < maxindex)
		return result + (f[s+1] - f[s-1])/(2*dt);
	
	if(s == 0)
		return result + (f[s+1] - f[s])/(dt);
	
	return result + (f[s] - f[s-1])/(dt);
}

__global__ void theta_k(float* f,float sigma,float a,float dt, int maxindex,float* res)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;

	if(tid <= maxindex)
	{
		float t = tid*dt;
		res[tid] = theta(f,sigma,t,a,dt,maxindex);
	}
}

__global__ void ZBC_k(float S1,float S2,float K,float* f,float* p,float* theta,float sigma,float dtstep,float dt,float a, float rs,curandStateXORWOW_t* states,float *ZBC)
{
	// Calculate A and B to get P
	// Simulate r, and obtain the integral (can use function of before with a minor tweak)
	// put everything together and do reduction as usual
	int tid = threadIdx.x;
	extern __shared__ float sdata[];
	curandStateXORWOW_t localState = states[threadIdx.x + blockIdx.x*blockDim.x];
	float2 G = curand_normal2(&localState);
	// convert S1 and S2 in the closest step.
	int S1step = roundf(S1/dtstep);
	int S2step = roundf(S2/dtstep);
	float B = (1-expf(-a*(S2-S1)))/a;
	float A = p[S2step]/p[S1step] * expf( B * f[S1step] - (sigma*sigma *(1-expf(-2*a*S2)))/(4*a) * B * B);
	float rS1;
	// for P I still miss r(S1). Lets calculate integral of r(t). We are gonna calculate r(S1) during it.

	float sintegral = 0;

	// maybe not elegant but for now it will suffice
	int X = S1/dt;
	float s_temp = S1;
	float r_temp = rs;
	for(int f = 0; f <= X; f++)
	{
		float integral = 0;
		float w = f*dt;
		int N = (S1-w)/dt;
		for(int k = 0; k <= N; k++)
		{	
			float i = w + k*dt;
			int step = roundf(i/dtstep);
			integral += dt/2 * expf(-a*(S1-i)) * theta[step] * ((k != 0 && k != N) ? 2 : 1);
		}
		float m = r_temp * expf(-a*(S1-s_temp)) + integral;
		float sigmaBig = sqrt(sigma*sigma*(1-expf(-2*a*(S1-s_temp)))/(2*a));
		r_temp = m + sigmaBig*G.x;
		s_temp = w;
		if(f == X) rS1 = r_temp;
		sintegral += dt/2 * r_temp * ((f != 0 && f != X) ? 2 : 1);
	}
	float P = A*expf(- B*rS1);
	sdata[tid] = fmaxf(0.0f,expf(-sintegral)*(P-K));
	__syncthreads();

	for(int k = blockDim.x/2; k > 0; k /= 2)
	{
		if(tid < k){
			sdata[tid] += sdata[k+tid];
			sdata[blockDim.x + tid] += sdata[k+tid+blockDim.x];
		}
		__syncthreads();
	}

	if(tid == 0){
		atomicAdd(ZBC, sdata[0]);
	}
}

__global__ void ZBC_derivative_k(float S1,float S2,float K,float* f,float* p,float* theta,float sigma,float dtstep,float dt,float a, float rs,curandStateXORWOW_t* states,float *ZBC)
{
	// derivative of P(S_1,S_2)
	// we need: derivative of A and derivative of r
	int tid = threadIdx.x;
	extern __shared__ float sdata[];
	curandStateXORWOW_t localState = states[threadIdx.x + blockIdx.x*blockDim.x];
	float2 G = curand_normal2(&localState);
	// need to calculate P(S_1,S_2) otherwise I cant decide which path to take
	int S1step = roundf(S1/dtstep);
	int S2step = roundf(S2/dtstep);
	float B = (1-expf(-a*(S2-S1)))/a;
	float A = p[S2step]/p[S1step] * expf( B * f[S1step] - (sigma*sigma *(1-expf(-2*a*S2)))/(4*a) * B * B);
	float rS1;

	// calculates integral of r_s from 0 to S_1
	int X = S1/dt;
	float s_temp = S1;
	float r_temp = rs;
	float sintegral = 0;
	for(int f = 0; f <= X; f++)
	{
		float integral = 0;
		float w = f*dt;
		int N = (S1-w)/dt;
		for(int k = 0; k <= N; k++)
		{	
			float i = w + k*dt;
			int step = roundf(i/dtstep);
			integral += dt/2 * expf(-a*(S1-i)) * theta[step] * ((k != 0 && k != N) ? 2 : 1);
		}
		float m = r_temp * expf(-a*(S1-s_temp)) + integral;
		float sigmaBig = sqrt(sigma*sigma*(1-expf(-2*a*(S1-s_temp)))/(2*a));
		r_temp = m + sigmaBig*G.x;
		s_temp = w;
		if(f == X) rS1 = r_temp;
		sintegral += dt/2 * r_temp * ((f != 0 && f != X) ? 2 : 1);
	}
	float P = A*expf(- B*rS1);

	float temp_dev = 0;
	float dev_integral = 0;
	for(int f = 1; f <= X ; f++)
	{
		float w = f*dt;
		float m = temp_dev * expf(-a*(S1-w)) + (2*sigma*expf(-a*S1)*(cosh(a*S1)-cosh(a*w)))/(a*a);
		float sigmaBig = sqrt(sigma*sigma*(1-expf(-2*a*(S1-w)))/(2*a));
		temp_dev = m+ sigmaBig/sigma * G.x;
		dev_integral = dt/2*temp_dev*((f != X) ? 2 : 1);
	}

	float result = fmaxf(0.0f,dev_integral*expf(-sintegral)*(P-K));	

	if(P > K)
	{
		// need to calculate derivative of P(S_1,S_2)
		temp_dev = 0;
		for(int f = 0; f <= X ; f++)
		{
			float w = f*dt;
			float m = temp_dev * expf(-a*(S1-w)) + (2*sigma*expf(-a*S1)*(cosh(a*S1)-cosh(a*w)))/(a*a);
			float sigmaBig = sqrt(sigma*sigma*(1-expf(-2*a*(S1-w)))/(2*a));
			temp_dev = m+ sigmaBig/sigma * G.x;
		}
		float Adev = 0;
		Adev = A*(-(2*sigma*(1-expf(-2*a*S2)))/(4*a)*B*B);
		float Pdev = A*expf(-B*rS1)*temp_dev + Adev*expf(-B*rS1);
		sdata[tid] = Pdev*expf(-sintegral)-result;
	}else{sdata[tid] = -result;}
	__syncthreads();
	for(int k = blockDim.x/2; k > 0; k /= 2)
	{
		if(tid < k){
			sdata[tid] += sdata[k+tid];
			sdata[blockDim.x + tid] += sdata[k+tid+blockDim.x];
		}
		__syncthreads();
	}

	if(tid == 0){
		atomicAdd(ZBC, sdata[0]);
	}
}	

int main(void) {
	int n = NB * NTPB;
	int steps = 20;
	float sigma = 0.1;
	float s = 0;
	float dt = 0.01;
	float rzero = 0.012;
	float a = 1.0;
	float* PGPU; 
	float* FGPU;

	float* P = (float*)malloc(sizeof(float)*steps);
	float* F = (float*)malloc(sizeof(float)*steps);

	cudaMalloc(&PGPU,sizeof(float));
	cudaMalloc(&FGPU,sizeof(float));
	cudaMemset(PGPU, 0, sizeof(float));
	cudaMemset(FGPU, 0, sizeof(float));
	curandStateXORWOW_t* states;
	cudaMalloc(&states,n*sizeof(curandStateXORWOW_t));

	init_curand_state_k<<<NB, NTPB>>>(states);
	cudaDeviceSynchronize();

	float PCPU,FCPU;
	/*for(int i = 0; i < steps; i++)
	{
		float t = ((float)T)/((float)steps)*i;
		Bond_Price_k<<<NB,NTPB,2*NTPB*sizeof(float)>>>(sigma,dt,a,rzero,s,t,states,PGPU,FGPU);
		cudaDeviceSynchronize();
		cudaMemcpy(&PCPU, PGPU, sizeof(float), cudaMemcpyDeviceToHost);
    	cudaMemcpy(&FCPU, FGPU, sizeof(float), cudaMemcpyDeviceToHost);
    	P[i] = PCPU/n;
    	F[i] = FCPU/n;
	}*/
	float t = 5.0;
	Bond_Price_k<<<NB,NTPB,2*NTPB*sizeof(float)>>>(sigma,dt,a,rzero,s,t,states,PGPU,FGPU);
	cudaDeviceSynchronize();
	cudaMemcpy(&PCPU, PGPU, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&FCPU, FGPU, sizeof(float), cudaMemcpyDeviceToHost);
	// call theta_k and get an array with its piecewise linear expression

	// use it to simulate ZBC (tedious but easy)

	// divide out by number of total threads
	printf("The zero coupon bond price is %f\nThe forward rate is %f\n", PCPU/n,FCPU/n);
	
	cudaFree(PGPU);
	cudaFree(FGPU);
	cudaFree(states);
	return 0;
}