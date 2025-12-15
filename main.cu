/**************************************************************

 HPCA PROJECT, SUBJECT 6

***************************************************************/

#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>

#define NB 1024
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
	float sintegral = dt/2*rs;
	curandStateXORWOW_t localState = states[threadIdx.x + blockIdx.x*blockDim.x];
	float A,B;
	int X = (t-s)/dt;
	float r_temp = rs;
	for(int f = 1; f <= X; f++)
	{
		float2 G = curand_normal2(&localState);
		float integral = 0;
		float w = s + f*dt;
		if( w < 5)
		{	
			A = 0.012;
			B = 0.0014;
		}else{
			A = 0.019;
			B = 0.001;
		}
		// implements integral term of m
		integral = 1/(a*a)*((a*(A+B*w)-B)-expf(-a*dt)*(a*(A+B*(w-dt)-B)));
		float m = r_temp * expf(-a*(dt)) + integral;
		float sigmaBig = sqrt(sigma*sigma*(1-expf(-2*a*(dt)))/(2*a));
		r_temp = m + sigmaBig*G.x;
		sintegral += dt/2 * r_temp * ((f != 0 && f != X) ? 2 : 1);
	}
	return sintegral;
}

__global__ void Bond_Price_k(float sigma,float dt, float a,float rs,float s,float t,curandStateXORWOW_t* states,float* Pout,float* Aout)
{
	extern __shared__ float sdata[];
	int tid = threadIdx.x;
	sdata[tid] = expf(-calculate_Price(sigma,dt,a,rs,s,t,states));
	sdata[blockDim.x+tid] =  (calculate_Price(sigma,dt,a,rs,s,t+dt*0.5,states) - calculate_Price(sigma,dt,a,rs,s,t-dt*0.5,states) )/(2*dt*0.5);
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

__global__ void ZBC_k(float S1,float S2,float K,float* f,float* p,float* theta,float sigma,float dt,float a, float rs,curandStateXORWOW_t* states,float *ZBC)
{
	// Calculate A and B to get P
	// Simulate r, and obtain the integral (can use function of before with a minor tweak)
	// put everything together and do reduction as usual
	int tid = threadIdx.x;
	extern __shared__ float sdata[];
	curandStateXORWOW_t localState = states[threadIdx.x + blockIdx.x*blockDim.x];
	// convert S1 and S2 to the closest step.
	int S1step = roundf(S1/dt)-1;
	int S2step = roundf(S2/dt)-1;
	float B = (1-expf(-a*(S2-S1)))/a;
	float A = p[S2step]/p[S1step] * expf( B * f[S1step] - (sigma*sigma *(1-expf(-2*a*S2)))/(4*a) * B * B);
	//printf("%d %d %f %f %f \n",S2step,S1step,p[S2step],p[S1step],f[S1step]);
	float rS1;
	// for P I still miss r(S1). Lets calculate integral of r(t). We are gonna calculate r(S1) during it.

	float sintegral = dt/2*rs;

	// maybe not elegant but for now it will suffice
	int X = S1/dt;
	float r_temp = rs;
	for(int f = 1; f <= X; f++)
	{
		float integral = 0;
		float2 G = curand_normal2(&localState);
		integral += expf(-a*(dt)) * theta[f-1]; 
		integral += 1.0f * theta[f];
		integral *= dt/2;

		float m = r_temp * expf(-a*(dt)) + integral;
		float sigmaBig = sqrtf(sigma*sigma*(1-expf(-2*a*(dt)))/(2*a));
		r_temp = m + sigmaBig*G.x;
		if(f == X) rS1 = r_temp;
		sintegral += dt/2 * r_temp * ((f != X) ? 2 : 1);
	}
	float P = A*expf(-B*rS1);
	sdata[tid] = expf(-sintegral)*fmaxf(0.0f,P-K);
	__syncthreads();

	for(int k = blockDim.x/2; k > 0; k /= 2)
	{
		if(tid < k){
			sdata[tid] += sdata[k+tid];
		}
		__syncthreads();
	}

	if(tid == 0){
		atomicAdd(ZBC, sdata[0]);
	}
}

__global__ void ZBC_derivative_k(float S1,float S2,float K,float* f,float* p,float* theta,float sigma,float dt,float a, float rs,curandStateXORWOW_t* states,float *ZBC)
{
	// derivative of P(S_1,S_2)
	// we need: derivative of A and derivative of r
	int tid = threadIdx.x;
	extern __shared__ float sdata[];
	curandStateXORWOW_t localState = states[threadIdx.x + blockIdx.x*blockDim.x];
	// need to calculate P(S_1,S_2) otherwise I cant decide which path to take
	int S1step = roundf(S1/dt)-1;
	int S2step = roundf(S2/dt)-1;
	float B = (1-expf(-a*(S2-S1)))/a;
	float A = p[S2step]/p[S1step] * expf( B * f[S1step] - (sigma*sigma *(1-expf(-2*a*S2)))/(4*a) * B * B);
	float rS1;

	// calculates integral of r_s from 0 to S_1
	int X = S1/dt;
	float r_temp = rs;
	float temp_dev = 0;
	float dev_integral = 0;
	float sintegral = dt/2*rs;
	for(int f = 1; f <= X; f++)
	{
		float2 G = curand_normal2(&localState);
		float integral = 0;

		integral += expf(-a*(dt)) * theta[f-1]; 
		integral += 1.0f * theta[f];
		integral *= dt/2;

		float m = r_temp * expf(-a*(dt)) + integral;
		float sigmaBig = sqrtf(sigma*sigma*(1-expf(-2*a*(dt)))/(2*a));
		r_temp = m + sigmaBig*G.x;
		if(f == X) rS1 = r_temp;
		sintegral += dt/2 * r_temp * ((f != X) ? 2 : 1);

		float w = f*dt;
		float m_dev = temp_dev * expf(-a*(dt)) + (2*sigma*expf(-a*w)*(cosh(a*w)-cosh(a*(w-dt))))/(a*a);
		temp_dev = m_dev + sigmaBig/sigma * G.x;
		dev_integral += dt/2*temp_dev*((f != X) ? 2 : 1);
	}
	float P = A*expf(- B*rS1);


	float result = fmaxf(0.0f,(P-K));
	result *= dev_integral*expf(-sintegral);	

	if(P > K)
	{
		float Adev = 0;
		Adev = A*(-(2*sigma*(1-expf(-2*a*S2)))/(4*a)*B*B);
		float Pdev = A*expf(-B*rS1)*(-B)*temp_dev + Adev*expf(-B*rS1);
		sdata[tid] = Pdev*expf(-sintegral)-result;
	}else{sdata[tid] = -result;}
	__syncthreads();
	for(int k = blockDim.x/2; k > 0; k /= 2)
	{
		if(tid < k){
			sdata[tid] += sdata[k+tid];
		}
		__syncthreads();
	}

	if(tid == 0){
		atomicAdd(ZBC, sdata[0]);
	}
}	

int main(void) {
	int n = NB * NTPB;
	int steps = 50;
	float sigma = 0.1;
	float s = 0;
	float dt =  ((float)T)/((float)steps);
	float rzero = 0.012;
	float a = 1.0;
	float* PGPU; 
	float* FGPU;
	float* thetaGPU;
	float* ZBCGPU;
	float* ZBCGPUD;
	float ZBC;
	float ZBC2;

	float* P = (float*)malloc(sizeof(float)*steps);
	float* F = (float*)malloc(sizeof(float)*steps);

	cudaMalloc(&PGPU,sizeof(float));
	cudaMalloc(&FGPU,sizeof(float));
	cudaMalloc(&thetaGPU,sizeof(float)*steps);
	cudaMemset(PGPU, 0, sizeof(float));
	cudaMemset(FGPU, 0, sizeof(float));
	curandStateXORWOW_t* states;
	cudaMalloc(&states,n*sizeof(curandStateXORWOW_t));
	cudaMalloc(&ZBCGPU,sizeof(float));
	cudaMalloc(&ZBCGPUD,sizeof(float));
	cudaMemset(ZBCGPU,0,sizeof(float));
	cudaMemset(ZBCGPUD,0,sizeof(float));

	init_curand_state_k<<<NB, NTPB>>>(states);
	cudaDeviceSynchronize();

	float PCPU,FCPU;
	for(int i = 1; i <= steps; i++)
	{
		float t = dt*i;
		Bond_Price_k<<<NB,NTPB,2*NTPB*sizeof(float)>>>(sigma,dt,a,rzero,s,t,states,PGPU,FGPU);
		cudaDeviceSynchronize();
		cudaMemcpy(&PCPU, PGPU, sizeof(float), cudaMemcpyDeviceToHost);
    	cudaMemcpy(&FCPU, FGPU, sizeof(float), cudaMemcpyDeviceToHost);
    	P[i-1] = PCPU/n;
    	F[i-1] = FCPU/n;
    	cudaMemset(PGPU,0,sizeof(float));
    	cudaMemset(FGPU,0,sizeof(float));
	}
	cudaMalloc(&FGPU,sizeof(float)*steps);
	cudaMalloc(&PGPU,sizeof(float)*steps);
	cudaMemcpy(FGPU,F,sizeof(float)*steps,cudaMemcpyHostToDevice);
	cudaMemcpy(PGPU,P,sizeof(float)*steps,cudaMemcpyHostToDevice);

	//theta_k(float* f,float sigma,float a,float dt, int maxindex,float* res)
	//theta_k<<<1,steps>>>(FGPU,sigma,a,dt,steps,thetaGPU);

	// float S1,float S2,float K,float* f,float* p,float* theta,float sigma,float dtstep,float dt,float a, float rs,curandStateXORWOW_t* states,float *ZBC)
	/*ZBC_k<<<NB,NTPB,NTPB*sizeof(float)>>>(5,10,expf(-0.1),FGPU,PGPU,thetaGPU,sigma,dt,a,rzero,states,ZBCGPU);
	cudaMemcpy(&ZBC,ZBCGPU,sizeof(float),cudaMemcpyDeviceToHost);
	printf("ZBC value is %f\n",ZBC/n);*/
	theta_k<<<1,steps>>>(FGPU,sigma-0.0001,a,dt,steps,thetaGPU);
	ZBC_k<<<NB,NTPB,NTPB*sizeof(float)>>>(5,10,expf(-0.1),FGPU,PGPU,thetaGPU,sigma-0.0001,dt,a,rzero,states,ZBCGPU);
	cudaDeviceSynchronize();
	cudaMemcpy(&ZBC,ZBCGPU,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemset(ZBCGPU,0,sizeof(float));
	theta_k<<<1,steps>>>(FGPU,sigma+0.0001,a,dt,steps,thetaGPU);
	ZBC_k<<<NB,NTPB,NTPB*sizeof(float)>>>(5,10,expf(-0.1),FGPU,PGPU,thetaGPU,sigma+0.0001,dt,a,rzero,states,ZBCGPU);
	cudaDeviceSynchronize();
	cudaMemcpy(&ZBC2,ZBCGPU,sizeof(float),cudaMemcpyDeviceToHost);
	ZBC /= n;
	ZBC2 /= n;
	printf("Derivative with first method of ZBC is %f\n",(ZBC2-ZBC)/(2*0.0001));
	theta_k<<<1,steps>>>(FGPU,sigma,a,dt,steps,thetaGPU);
	ZBC_derivative_k<<<NB,NTPB,NTPB*sizeof(float)>>>(5,10,expf(-0.1),FGPU,PGPU,thetaGPU,sigma,dt,a,rzero,states,ZBCGPUD);
	cudaDeviceSynchronize();
	cudaMemcpy(&ZBC,ZBCGPUD,sizeof(float),cudaMemcpyDeviceToHost);
	printf("Derivative of ZBC is %f\n",ZBC/n);

/*	Bond_Price_k<<<NB,NTPB,2*NTPB*sizeof(float)>>>(sigma,dt,a,rzero,s,T,states,PGPU,FGPU);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
    	printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
	}

	cudaDeviceSynchronize();
	cudaMemcpy(&PCPU, PGPU, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&FCPU, FGPU, sizeof(float), cudaMemcpyDeviceToHost);

	// divide out by number of total threads
	printf("The zero coupon bond price is %f\nThe forward rate is %f\n", PCPU/n,FCPU/n);*/
	
	cudaFree(PGPU);
	cudaFree(FGPU);
	cudaFree(thetaGPU);
	cudaFree(ZBCGPU);
	cudaFree(states);
	cudaFree(ZBCGPUD);
	free(P);
	free(F);
	return 0;
}