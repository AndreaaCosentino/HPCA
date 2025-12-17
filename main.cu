/**************************************************************

 HPCA PROJECT, SUBJECT 6

***************************************************************/

#include <stdio.h>
#include <curand_kernel.h>
#include <iostream>
#include <math.h>

#define NB 10000
#define NTPB 1024

/// Helper for catching errors
void testCUDA(cudaError_t error, const char *file, int line) {
	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void init_curand_state_k(curandStateXORWOW_t *states) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(0, id, 0, &states[id]);
}

__device__ float calculate_price(float param_sigma, float param_a, float starting_r, float starting_time,
                                 float target_time, float time_delta, curandStateXORWOW_t *states) {
	curandStateXORWOW_t localState = states[threadIdx.x + blockIdx.x * blockDim.x];

	int total_steps = (target_time - starting_time) / time_delta;
	float noise_sensitivity = sqrt(param_sigma * param_sigma * (1 - expf(-2 * param_a * time_delta)) / (2 * param_a));

	float integral = 0;
	float r_step = starting_r;

	for (int step = 1; step <= total_steps; step++) {
		float r_prev_step = r_step;

		float step_time = starting_time + step * time_delta;
		float previous_step_time = step_time - time_delta;
		// Computing the mean
		float m = 0;
		{
			// First term of the mean
			m += r_step * expf(-param_a * time_delta);

			float A, B;
			// Second term of the mean (formula for the integral)
			if (step_time < 5 || previous_step_time >= 5) {
				A = (step_time < 5) ? 0.012 : 0.019;
				B = (step_time < 5) ? 0.0014 : 0.001;
				step_time = (step_time < 5) ? step_time : step_time - 5;
				m +=
				(
					param_a * (A + B * step_time) - B
					- expf(-param_a * time_delta) * (param_a * (A + B * (step_time - time_delta) - B))
				) / (param_a * param_a);
			} else {
				float d_1 = 5 - previous_step_time;
				float d_2 = step_time - 5;
				A = 0.012;
				B = 0.0014;
				m += 1 / (param_a * param_a) * (param_a * (A + B * 5) - B - expf(-param_a * d_1) * (
					                                param_a * (A + B * previous_step_time) - B)) * expf(-param_a * d_2);

				A = 0.019;
				B = 0.001;
				// in this case w-dt = 0, so B*0 = 0
				m += 1 / (param_a * param_a) * (param_a * (A + B * d_2) - B - expf(-param_a * d_2) * (param_a * A - B));
			}
		}

		// Computing the noise
		float noise = 0;
		{
			float random = curand_normal(&localState);
			noise = noise_sensitivity * random;
		}

		r_step = m + noise;

		integral += 0.5f * (r_prev_step + r_step) * time_delta;
	}

	return expf(-integral);
}

template<typename CallableValue>
__global__ void tree_sum(CallableValue val, float *out) {
	//if (threadIdx.x == 0 && blockIdx.x == 0) *out = 0;

	extern __shared__ float block_data[];
	block_data[threadIdx.x] = val();

	__syncthreads();

	for (int k = blockDim.x / 2; k > 0; k /= 2) {
		if (threadIdx.x < k)
			block_data[threadIdx.x] += block_data[threadIdx.x + k];

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		atomicAdd(out, block_data[0]);
	}
}

// Theta functions not tested

__device__ float theta(float param_sigma, float param_a, float *f, int f_length, int at_timestep, float delta_T) {
	float T = delta_T * at_timestep;
	float result = param_a * f[at_timestep] + param_sigma * param_sigma * (1 - expf(-2 * param_a * T)) / (2 * param_a);

	// Calculate the derivative of f differently based on whether we are on the edge of the array
	if (at_timestep > 0 && at_timestep < f_length - 1)
		return result + (f[at_timestep + 1] - f[at_timestep - 1]) / (2 * delta_T);
	if (at_timestep == 0)
		return result + (f[at_timestep + 1] - f[at_timestep]) / delta_T;
	if (at_timestep == f_length - 1)
		return result + (f[at_timestep] - f[at_timestep - 1]) / delta_T;

	return -1;
}

__global__ void theta_k(float param_sigma, float param_a, float *f, int f_length, float delta_T, float *out) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < f_length) {
		out[tid] = theta(param_sigma, param_a, f, f_length, tid, delta_T);
	}
}

__device__ float calculate_ZBC(float param_sigma, float param_a, float S1, float S2, float K, float *P, float *f, float *theta,
					 float starting_r, float time_delta, curandStateXORWOW_t *states) {
	// Calculate A and B to get P
	// Simulate r, and obtain the integral (can use function of before with a minor tweak)
	// put everything together and do reduction as usual
	curandStateXORWOW_t localState = states[threadIdx.x + blockIdx.x * blockDim.x];
	// convert S1 and S2 to the closest step.
	int S1step = roundf(S1 / time_delta);
	int S2step = roundf(S2 / time_delta);
	float B = (1 - expf(-param_a * (S2 - S1))) / param_a;
	float A = P[S2step] / P[S1step] * expf(B * f[S1step] - (param_sigma * param_sigma * (1 - expf(-2 * param_a * S2))) / (4 * param_a) * B * B);
	//printf("%d %d %f %f %f \n",S2step,S1step,p[S2step],p[S1step],f[S1step]);
	float rS1;
	// for P I still miss r(S1). Lets calculate integral of r(t). We are gonna calculate r(S1) during it.

	float sintegral = time_delta / 2 * starting_r;

	// maybe not elegant but for now it will suffice
	int X = S1 / time_delta;
	float r_temp = starting_r;
	for (int f = 1; f <= X; f++) {
		float integral = 0;
		float2 G = curand_normal2(&localState);
		integral += expf(-param_a * (time_delta)) * theta[f - 1];
		integral += 1.0f * theta[f];
		integral *= time_delta / 2;

		float m = r_temp * expf(-param_a * (time_delta)) + integral;
		float sigmaBig = sqrtf(param_sigma * param_sigma * (1 - expf(-2 * param_a * (time_delta))) / (2 * param_a));
		r_temp = m + sigmaBig * G.x;
		if (f == X) rS1 = r_temp;
		sintegral += time_delta / 2 * r_temp * ((f != X) ? 2 : 1);
	}
	float P_ = A * expf(-B * rS1);
	return expf(-sintegral) * fmaxf(0.0f, P_ - K);
}

__global__ void ZBC_derivative_k(float S1, float S2, float K, float *f, float *p, float *theta, float sigma, float dt,
                                 float a, float rs, curandStateXORWOW_t *states, float *ZBC) {
	// derivative of P(S_1,S_2)
	// we need: derivative of A and derivative of r
	int tid = threadIdx.x;
	extern __shared__ float sdata[];
	curandStateXORWOW_t localState = states[threadIdx.x + blockIdx.x * blockDim.x];
	// need to calculate P(S_1,S_2) otherwise I cant decide which path to take
	int S1step = roundf(S1 / dt);
	int S2step = roundf(S2 / dt);
	float B = (1 - expf(-a * (S2 - S1))) / a;
	float A = p[S2step] / p[S1step] * expf(B * f[S1step] - (sigma * sigma * (1 - expf(-2 * a * S2))) / (4 * a) * B * B);
	float rS1;

	// calculates integral of r_s from 0 to S_1
	int X = S1 / dt;
	float r_temp = rs;
	float temp_dev = 0;
	float dev_integral = 0;
	float sintegral = dt / 2 * rs;
	for (int f = 1; f <= X; f++) {
		float2 G = curand_normal2(&localState);
		float integral = 0;

		integral += expf(-a * (dt)) * theta[f - 1];
		integral += 1.0f * theta[f];
		integral *= dt / 2;

		float m = r_temp * expf(-a * (dt)) + integral;
		float sigmaBig = sqrtf(sigma * sigma * (1 - expf(-2 * a * (dt))) / (2 * a));
		r_temp = m + sigmaBig * G.x;
		if (f == X) rS1 = r_temp;
		sintegral += dt / 2 * r_temp * ((f != X) ? 2 : 1);

		float w = f * dt;
		float m_dev = temp_dev * expf(-a * (dt)) + (2 * sigma * expf(-a * w) * (cosh(a * w) - cosh(a * (w - dt)))) / (
			              a * a);
		temp_dev = m_dev + sigmaBig / sigma * G.x;
		dev_integral += dt / 2 * temp_dev * ((f != X) ? 2 : 1);
	}
	float P = A * expf(-B * rS1);


	float result = fmaxf(0.0f, (P - K));
	result *= dev_integral * expf(-sintegral);

	if (P > K) {
		float Adev = 0;
		Adev = A * (-(2 * sigma * (1 - expf(-2 * a * S2))) / (4 * a) * B * B);
		float Pdev = A * expf(-B * rS1) * (-B) * temp_dev + Adev * expf(-B * rS1);
		sdata[tid] = Pdev * expf(-sintegral) - result;
	} else { sdata[tid] = -result; }
	__syncthreads();
	for (int k = blockDim.x / 2; k > 0; k /= 2) {
		if (tid < k) {
			sdata[tid] += sdata[k + tid];
		}
		__syncthreads();
	}

	if (tid == 0) {
		atomicAdd(ZBC, sdata[0]);
	}
}

int main() {
	// Question 1 :
	int steps = 30;

	float param_sigma = 0.1;
	float param_a = 1.0;
	float T_start = 0;
	float T_end = 10;
	float delta_T = (T_end - T_start) / (steps - 1);

	curandStateXORWOW_t *states;

	cudaMalloc(&states,NB * NTPB * sizeof(curandStateXORWOW_t));
	init_curand_state_k<<<NB, NTPB>>>(states);
	cudaDeviceSynchronize();

	float *P;
	cudaMallocManaged(&P, steps * sizeof(float));
	float *f;
	cudaMallocManaged(&f, steps * sizeof(float));

	float *sum;
	cudaMallocManaged(&sum, sizeof(float));

	float avg_prev;
	f[0] = 0.012;
	for (int i = 0; i < steps; ++i) {
		float T_i = T_start + delta_T * i;
		*sum = 0.0f;

		// P(0, T)
		tree_sum<<<NB, NTPB, NTPB * sizeof(float)>>>([param_sigma, param_a, T_i, states] __device__ {
			return calculate_price(param_sigma, param_a, 0.012, 0, T_i, 0.01, states);
		}, sum);
		cudaDeviceSynchronize();

		float avg = *sum / (NB * NTPB);

		P[i] = avg;
		// std::cout << "P(0, " << T_i << ") = " << avg << std::endl;

		// f(0, T)
		if (i != 0) {
			float delta = logf(avg) - logf(avg_prev);
			f[i] = -(delta / delta_T);
			// std::cout << "f(0, " << T_i << ") = " << F[i] << std::endl;
		}

		avg_prev = avg;
	}

	// Question 2.a :
	float *theta;
	cudaMallocManaged(&theta, steps * sizeof(float));

	theta_k<<<1, steps>>>(param_sigma, param_a, f, steps, delta_T, theta);
	cudaDeviceSynchronize();

	for (int i = 0; i < steps; ++i) {
		float T_i = T_start + delta_T * i;
		std::cout << "theta(" << T_i << ") = " << theta[i] << std::endl;
	}

	// Question 2.b
	{
		tree_sum<<<NB, NTPB, NTPB * sizeof(float)>>>([param_sigma, param_a, P, f, theta, states] __device__ {
			return calculate_ZBC(param_sigma, param_a, 5, 10, expf(-0.1), P, f, theta, 0.012, 0.01, states);
		}, sum);

		float avg = *sum / (NB * NTPB);
		std::cout << "ZCB(5, 10, e^-0.1) = " << avg << std::endl;
	}

	cudaFree(sum);
	cudaFree(states);
	cudaFree(P);
	cudaFree(f);
	cudaFree(theta);

	return 0;
}

int main_old() {
	int n = NB * NTPB;
	int steps = 30;
	float sigma = 0.1;
	float s = 0;
	float T = 10;
	float dt = ((float) T) / ((float) steps);
	float rzero = 0.012;
	float a = 1.0;
	float *PGPU;
	float *FGPU;
	float *thetaGPU;
	float *ZBCGPU;
	float *ZBCGPUD;
	float ZBC;
	float ZBC2;
	int num_el = steps + 1;
	float *P = (float *) malloc(sizeof(float) * num_el);
	float *F = (float *) malloc(sizeof(float) * num_el);

	cudaMalloc(&PGPU, sizeof(float));
	cudaMalloc(&thetaGPU, sizeof(float) * num_el);
	cudaMemset(PGPU, 0, sizeof(float));
	curandStateXORWOW_t *states;
	cudaMalloc(&states, n * sizeof(curandStateXORWOW_t));
	cudaMalloc(&ZBCGPU, sizeof(float));
	cudaMalloc(&ZBCGPUD, sizeof(float));
	cudaMemset(ZBCGPU, 0, sizeof(float));
	cudaMemset(ZBCGPUD, 0, sizeof(float));

	init_curand_state_k<<<NB, NTPB>>>(states);
	cudaDeviceSynchronize();

	float PCPU, FCPU;
	P[0] = 1;
	F[0] = rzero;
	for (int i = 1; i <= steps; i++) {
		float t = dt * i;
		std::cout << sigma << " " << a << " " << rzero << " " << s << " " << t << " " << dt << std::endl;
		tree_sum<<<NB, NTPB, NTPB * sizeof(float)>>>([sigma, a, rzero, s, t, dt, states] __device__ {
			return calculate_price(sigma, a, rzero, s, t, dt, states);
		}, PGPU);
		//Bond_Price_k<<<NB,NTPB,2*NTPB*sizeof(float)>>>(sigma,dt,a,rzero,s,t,states,PGPU);
		cudaDeviceSynchronize();
		cudaMemcpy(&PCPU, PGPU, sizeof(float), cudaMemcpyDeviceToHost);

		std::cout << "sum : " << PCPU << std::endl;
		P[i] = PCPU / n;
		std::cout << i << " : " << t << ", " << P[i] << std::endl;

		cudaMemset(PGPU, 0, sizeof(float));
	}

	for (int i = 1; i <= num_el; i++) {
		F[i] = -(logf(P[i]) - logf(P[i - 1])) / dt;
	}
	cudaMalloc(&FGPU, sizeof(float) * num_el);
	cudaMalloc(&PGPU, sizeof(float) * num_el);
	cudaMemcpy(FGPU, F, sizeof(float) * num_el, cudaMemcpyHostToDevice);
	cudaMemcpy(PGPU, P, sizeof(float) * num_el, cudaMemcpyHostToDevice);
	theta_k<<<1,num_el>>>(sigma, a, FGPU, steps, dt, thetaGPU);

	// float S1,float S2,float K,float* f,float* p,float* theta,float sigma,float dtstep,float dt,float a, float rs,curandStateXORWOW_t* states,float *ZBC)
	/*ZBC_k<<<NB,NTPB,NTPB*sizeof(float)>>>(5,10,expf(-0.1),FGPU,PGPU,thetaGPU,sigma,dt,a,rzero,states,ZBCGPU);
	cudaMemcpy(&ZBC,ZBCGPU,sizeof(float),cudaMemcpyDeviceToHost);
	printf("ZBC value is %f\n",ZBC/n);*/
	/*theta_k<<<1,steps>>>(FGPU,sigma-0.0001,a,dt,steps,thetaGPU);
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
	printf("Derivative of ZBC is %f\n",ZBC/n);*/
	/*float i = T;
	while(i <= 10)
	{
		dt =  ((float)i)/((float)steps);
		Bond_Price_k<<<NB,NTPB,2*NTPB*sizeof(float)>>>(sigma,dt,a,rzero,s,i,states,PGPU,FGPU);

		cudaDeviceSynchronize();
		cudaMemcpy(&PCPU, PGPU, sizeof(float), cudaMemcpyDeviceToHost);
    	cudaMemcpy(&FCPU, FGPU, sizeof(float), cudaMemcpyDeviceToHost);

		// divide out by number of total threads
		printf("round %f %f %f\n",i,PCPU/n,FCPU/n);
		cudaMemset(PGPU,0,sizeof(float));
		cudaMemset(FGPU,0,sizeof(float));
		i += 0.5;
	}*/
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
