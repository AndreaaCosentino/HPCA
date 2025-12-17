/**************************************************************

 HPCA PROJECT, SUBJECT 6

***************************************************************/

#include <stdio.h>
#include <curand_kernel.h>
#include <iostream>
#include <math.h>

#define NB 1024
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
	curandStateXORWOW_t localState = states[threadIdx.x + blockIdx.x * blockDim.x];
	float noise_sensitivity = sqrt(param_sigma * param_sigma * (1 - expf(-2 * param_a * time_delta)) / (2 * param_a));
	// convert S1 and S2 to the closest step.
	int S1step = roundf(S1 / time_delta);
	int S2step = roundf(S2 / time_delta);
	// Values A(S1,S2) and B(S1,S2)
	float B = (1 - expf(-param_a * (S2 - S1))) / param_a;
	float A = P[S2step] / P[S1step] * expf(B * f[S1step] - (param_sigma * param_sigma * (1 - expf(-2 * param_a * S2))) / (4 * param_a) * B * B);

	float integral = 0;

	int X = S1 / time_delta;
	float r_step = starting_r;
	for (int n = 1; n <= X; n++) {
		float r_prev_step = r_step;

		// Computing the mean
		float m = 0;
		{
			// First term of the mean
			m = r_step * expf(-param_a * (time_delta));

			// Second term of the mean (formula for the integral)
			m += 
				(
					expf(-param_a * (time_delta)) * theta[n - 1] +
						1.0f * theta[n] 
				) * time_delta / 2;
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
	float P_ana = A * expf(-B * r_step);
	return expf(-integral) * fmaxf(0.0f, P_ana - K);
}

__device__ float calculate_ZBC_dev(float param_sigma, float param_a, float S1, float S2, float K, float *P, float *f, float *theta,
					 float starting_r, float time_delta, curandStateXORWOW_t *states) {
	curandStateXORWOW_t localState = states[threadIdx.x + blockIdx.x * blockDim.x];
	float noise_sensitivity = sqrt(param_sigma * param_sigma * (1 - expf(-2 * param_a * time_delta)) / (2 * param_a));
	
	// convert S1 and S2 to the closest step.
	int S1step = roundf(S1 / time_delta);
	int S2step = roundf(S2 / time_delta);
	// Values A(S1,S2) and B(S1,S2)
	float B = (1 - expf(-param_a * (S2 - S1))) / param_a;
	float A = P[S2step] / P[S1step] * expf(B * f[S1step] - (param_sigma * param_sigma * (1 - expf(-2 * param_a * S2))) / (4 * param_a) * B * B);

	int X = S1 / time_delta;
	float r_step = starting_r;
	float dev_step = 0;
	float dev_integral = 0;
	float integral = 0;

	for (int n = 1; n <= X; n++) {
		float r_prev_step = r_step;
		float dev_prev_step = dev_step;
		// Computing the mean
		float m = 0;
		{
			// First term of the mean
			m = r_step * expf(-param_a * (time_delta));

			// Second term of the mean (formula for the integral)
			m += expf(-param_a * (time_delta)) * theta[n - 1] +
				1.0f * theta[n] * time_delta / 2;
		}
			
		// Computing the noise
		float noise = 0;
		float random = curand_normal(&localState);
		{
			noise = noise_sensitivity * random;
		}
		r_step = m + noise;

		integral += 0.5f * (r_prev_step + r_step) * time_delta;	

		float step_time = n * time_delta;

		// Compute derivative
		float m_dev = dev_step * expf(-param_a * (time_delta)) + 
							(2 * param_sigma * expf(-param_a * step_time) * 
								(cosh(param_a * step_time) - cosh(param_a * (step_time - time_delta)))) / (
			              			param_a * param_a);
		dev_step = m_dev + noise_sensitivity / param_sigma * random;
		// Integral of derivate of r(s) term
		dev_integral += 0.5f * (dev_prev_step + dev_step) * time_delta;	
	}
	float P_ana = A * expf(-B * r_step);


	float result = fmaxf(0.0f, (P_ana - K)) * dev_integral * expf(-integral);

	if (P_ana > K) {
		float Adev = 0;
		
		Adev = A * (-(2 * param_sigma * (1 - expf(-2 * param_a * S2))) / (4 * param_a) * B * B);
		float Pdev = A * expf(-B * r_step) * (-B) * dev_step + Adev * expf(-B * r_step);

		return Pdev * expf(-integral) - result;
	} 
	return -result; 
}

int main() {
	// Question 1 :
	int steps = 20;

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
		std::cout << "P(0, " << T_i << ") = " << avg << std::endl;

		// f(0, T)
		if (i != 0) {
			float delta = logf(avg) - logf(avg_prev);
			f[i] = -(delta / delta_T);
			std::cout << "f(0, " << T_i << ") = " << f[i] << std::endl;
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

	// Question 2.b :
	{
		*sum = 0.0f;
		tree_sum<<<NB, NTPB, NTPB * sizeof(float)>>>([param_sigma, param_a, P, f, theta,delta_T, states] __device__ {
			return calculate_ZBC(param_sigma, param_a, 5, 10, expf(-0.1), P, f, theta, 0.012, delta_T, states);
		}, sum);
		cudaDeviceSynchronize();
		float avg = *sum / (NB * NTPB);
		std::cout << "ZCB(5, 10, e^-0.1) = " << avg << std::endl;
	}

	//Question 3 :
	{	
		*sum = 0.0f;
		tree_sum<<<NB, NTPB, NTPB * sizeof(float)>>>([param_sigma, param_a, P, f, theta, delta_T, states] __device__ {
			return calculate_ZBC_dev(param_sigma, param_a, 5, 10, expf(-0.1), P, f, theta, 0.012, delta_T, states);
		}, sum);
		cudaDeviceSynchronize();
		float avg_dev = *sum / (NB * NTPB);


		*sum = 0.0f;
		float diff = 0.0001;
		float temp_param_sigma = param_sigma + diff;
		theta_k<<<1, steps>>>(temp_param_sigma, param_a, f, steps, delta_T, theta);
		cudaDeviceSynchronize();
		tree_sum<<<NB, NTPB, NTPB * sizeof(float)>>>([temp_param_sigma, param_a, P, f, theta, delta_T, states] __device__ {
			return calculate_ZBC(temp_param_sigma, param_a, 5, 10, expf(-0.1), P, f, theta, 0.012, delta_T, states);
		}, sum);
		cudaDeviceSynchronize();
		float diff_approx = (*sum / (NB * NTPB)) / (2*diff);


		temp_param_sigma = param_sigma - diff;
		*sum = 0.0f;
		theta_k<<<1, steps>>>(temp_param_sigma, param_a, f, steps, delta_T, theta);
		cudaDeviceSynchronize();
		tree_sum<<<NB, NTPB, NTPB * sizeof(float)>>>([temp_param_sigma, param_a, P, f, theta,delta_T, states] __device__ {
			return calculate_ZBC(temp_param_sigma, param_a, 5, 10, expf(-0.1), P, f, theta, 0.012, delta_T, states);
		}, sum);
		cudaDeviceSynchronize();
		diff_approx -= (*sum / (NB * NTPB)) / (2*diff);

		std::cout << "dev ZCB(5, 10, e^-0.1) = " << avg_dev << std::endl << "dev numerical method = " << diff_approx << std::endl;
	}

	cudaFree(sum);
	cudaFree(states);
	cudaFree(P);
	cudaFree(f);
	cudaFree(theta);

	return 0;
}