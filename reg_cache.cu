// Copyright (c) 2014 Thomas L. Falch
// For terms for useage and distribution, see the accompanying LICENCE file

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

//#define PLAIN
//#define SHARED
//#define SHUFFLE
//#define TEXTURE

//#define N_ITERATIONS 2
#define N_THREADS_BLOCK 128
#define N_THREADS_WARP 32
#define N_BLOCKS_X 128
#define N_BLOCKS_Y 128
#define BLOCK_DIM_Y 2
#define BLOCK_DIM_Z ((N_THREADS_BLOCK/N_THREADS_WARP)/BLOCK_DIM_Y)
//#define N_OUTPUT_X 2//2
//#define N_OUTPUT_Y 5//4
#define WARP_DIM_X 8
#define WARP_DIM_Y 4
//#define STENCIL_RADIUS_X 0
//#define STENCIL_RADIUS_Y 1

#define WARP_SIZE_X (WARP_DIM_X*N_OUTPUT_X)
#define WARP_SIZE_Y (WARP_DIM_Y*N_OUTPUT_Y)

#define INPUT_WARP_SIZE_X ((WARP_SIZE_X) - (STENCIL_RADIUS_X*((N_ITERATIONS -1)*2)))
#define INPUT_WARP_SIZE_Y ((WARP_SIZE_Y) - (STENCIL_RADIUS_Y*((N_ITERATIONS -1)*2)))

#define BLOCK_SIZE_Z  (WARP_SIZE_X*BLOCK_DIM_Z)
#define BLOCK_SIZE_Y (WARP_SIZE_Y*BLOCK_DIM_Y)

#ifdef SHUFFLE
#define INPUT_BLOCK_SIZE_Z (BLOCK_DIM_Z*INPUT_WARP_SIZE_X)
#define INPUT_BLOCK_SIZE_Y (BLOCK_DIM_Y*INPUT_WARP_SIZE_Y)
#else
#define INPUT_BLOCK_SIZE_Z ((BLOCK_SIZE_Z) - (STENCIL_RADIUS_X*((N_ITERATIONS -1)*2)))
#define INPUT_BLOCK_SIZE_Y ((BLOCK_SIZE_Y) - (STENCIL_RADIUS_Y*((N_ITERATIONS -1)*2)))
#endif

#define SIZE_X (INPUT_BLOCK_SIZE_Z * N_BLOCKS_X)
#define SIZE_Y (INPUT_BLOCK_SIZE_Y * N_BLOCKS_Y)


#define PADDED_SIZE_X  (INPUT_BLOCK_SIZE_Z * (N_BLOCKS_X+2))
#define PADDED_SIZE_Y  (INPUT_BLOCK_SIZE_Y * (N_BLOCKS_Y+2))
#define PADDED_SIZE (PADDED_SIZE_X*PADDED_SIZE_Y)
#define PADDING_X INPUT_BLOCK_SIZE_Z
#define PADDING_Y INPUT_BLOCK_SIZE_Y

#define REG_SIZE_X ((2*STENCIL_RADIUS_X) + N_OUTPUT_X)
#define REG_SIZE_Y ((2*STENCIL_RADIUS_Y) + N_OUTPUT_Y)
#define SHARED_SIZE_X  (BLOCK_SIZE_Z + 2*STENCIL_RADIUS_X)
#define SHARED_SIZE_Y  (BLOCK_SIZE_Y + 2*STENCIL_RADIUS_Y)



texture<float, cudaTextureType2D, cudaReadModeElementType> buffer_texture;



__device__ inline int index(int row, int col){
	return (row+PADDING_Y) * PADDED_SIZE_X + PADDING_X + col;
}

__device__ inline int temp_index(int row, int col){
	return (row+BLOCK_SIZE_Y) * (BLOCK_SIZE_Z*(N_BLOCKS_X+2)) + BLOCK_SIZE_Z + col;
}


__device__ inline int reg_index(int row, int col){
	return (row+STENCIL_RADIUS_Y)*REG_SIZE_X + col +STENCIL_RADIUS_X;
}


#ifdef PLAIN
__global__ void plain(float* input, float* output, float* temp, int iterations){

	int shift_y = (BLOCK_SIZE_Y - INPUT_BLOCK_SIZE_Y)/2;
	int shift_x = (BLOCK_SIZE_Z - INPUT_BLOCK_SIZE_Z)/2;
	int row = -shift_y + blockIdx.y * INPUT_BLOCK_SIZE_Y + threadIdx.y * WARP_SIZE_Y + (threadIdx.x/WARP_DIM_X)*N_OUTPUT_Y;
	int col = -shift_x + blockIdx.x * INPUT_BLOCK_SIZE_Z + threadIdx.z * WARP_SIZE_X + (threadIdx.x%WARP_DIM_X)*N_OUTPUT_X;
	int trow = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y * WARP_SIZE_Y + (threadIdx.x/WARP_DIM_X)*N_OUTPUT_Y;
	int tcol = blockIdx.x * BLOCK_SIZE_Z + threadIdx.z * WARP_SIZE_X + (threadIdx.x%WARP_DIM_X)*N_OUTPUT_X;

	float v[REG_SIZE_X*REG_SIZE_Y];
	float w[REG_SIZE_X*REG_SIZE_Y];


	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			v[reg_index(i,j)] = input[index(row+i, col+j)];
		}
	}

	for(int i = 0; i < STENCIL_RADIUS_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			v[reg_index(-STENCIL_RADIUS_Y + i, j)] = input[index(row-STENCIL_RADIUS_Y+i,col+j)];
			v[reg_index(N_OUTPUT_Y + i, j)] = input[index(row+N_OUTPUT_Y+i, col+j)];
		}
	}

	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < STENCIL_RADIUS_X; j++){
			v[reg_index(i,-STENCIL_RADIUS_X+j)] = input[index(row+i,col-STENCIL_RADIUS_X+j)];
			v[reg_index(i,N_OUTPUT_X+j)] = input[index(row+i, col+N_OUTPUT_X+j)];
		}
	}


	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			w[reg_index(i,j)] = 0.0f;

			for(int k = - STENCIL_RADIUS_X; k <= STENCIL_RADIUS_X; k++){
				w[reg_index(i,j)] += v[reg_index(i,j+k)];
			}
			for(int k = - STENCIL_RADIUS_Y; k <= STENCIL_RADIUS_Y; k++){
				w[reg_index(i,j)] += v[reg_index(i+k,j)];
			}
			w[reg_index(i,j)] /= (float)((2*STENCIL_RADIUS_X + 1) + (2*STENCIL_RADIUS_Y + 1));
		}
	}

	for(int it = 0; it < N_ITERATIONS-1; it++){

			__syncthreads();

			for(int i = 0; i < N_OUTPUT_Y; i++){
				for(int j = 0; j < N_OUTPUT_X; j++){
					temp[temp_index(trow+i, tcol+j)] = w[reg_index(i,j)];
				}
			}

			__syncthreads();

			for(int i = 0; i < N_OUTPUT_Y; i++){
				for(int j = 0; j < N_OUTPUT_X; j++){
					v[reg_index(i,j)] = temp[temp_index(trow+i, tcol+j)];
				}
			}

			for(int i = 0; i < STENCIL_RADIUS_Y; i++){
				for(int j = 0; j < N_OUTPUT_X; j++){
					v[reg_index(-STENCIL_RADIUS_Y + i, j)] = temp[temp_index(trow-STENCIL_RADIUS_Y+i,tcol+j)];
					v[reg_index(N_OUTPUT_Y + i, j)] = temp[temp_index(trow+N_OUTPUT_Y+i, tcol+j)];
				}
			}

			for(int i = 0; i < N_OUTPUT_Y; i++){
				for(int j = 0; j < STENCIL_RADIUS_X; j++){
					v[reg_index(i,-STENCIL_RADIUS_X+j)] = temp[temp_index(trow+i,tcol-STENCIL_RADIUS_X+j)];
					v[reg_index(i,N_OUTPUT_X+j)] = temp[temp_index(trow+i, tcol+N_OUTPUT_X+j)];
				}
			}

			for(int i = 0; i < N_OUTPUT_Y; i++){
				for(int j = 0; j < N_OUTPUT_X; j++){
					w[reg_index(i,j)] = 0.0f;

					for(int k = - STENCIL_RADIUS_X; k <= STENCIL_RADIUS_X; k++){
						w[reg_index(i,j)] += v[reg_index(i,j+k)];
					}
					for(int k = - STENCIL_RADIUS_Y; k <= STENCIL_RADIUS_Y; k++){
						w[reg_index(i,j)] += v[reg_index(i+k,j)];
					}
					w[reg_index(i,j)] /= (float)((2*STENCIL_RADIUS_X + 1) + (2*STENCIL_RADIUS_Y + 1));
				}
			}


	}

	int row_lower = blockIdx.y * INPUT_BLOCK_SIZE_Y;
	int row_upper = row_lower + INPUT_BLOCK_SIZE_Y;
	int col_lower = blockIdx.x * INPUT_BLOCK_SIZE_Z;
	int col_upper = col_lower + INPUT_BLOCK_SIZE_Z;
	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			if((row+i) >= row_lower && (row+i) < row_upper && (col+j) >= col_lower && (col+j) < col_upper){
				output[index(row+i, col+j)] = w[reg_index(i,j)];
			}
		}
	}

}
#endif

__device__ inline int shared_index(int row, int col){
	return (row + STENCIL_RADIUS_Y)*SHARED_SIZE_X + col + STENCIL_RADIUS_X;
}

#ifdef SHARED
__global__ void shared(float* input, float* output, int iterations){
	int shift_y = (BLOCK_SIZE_Y - INPUT_BLOCK_SIZE_Y)/2;
	int shift_x = (BLOCK_SIZE_Z - INPUT_BLOCK_SIZE_Z)/2;
	int row = -shift_y + blockIdx.y * INPUT_BLOCK_SIZE_Y + threadIdx.y * WARP_SIZE_Y + (threadIdx.x/WARP_DIM_X)*N_OUTPUT_Y;
	int col = -shift_x + blockIdx.x * INPUT_BLOCK_SIZE_Z + threadIdx.z * WARP_SIZE_X + (threadIdx.x%WARP_DIM_X)*N_OUTPUT_X;

	int srow = threadIdx.y * WARP_SIZE_Y + (threadIdx.x/WARP_DIM_X)*N_OUTPUT_Y;
	int scol = threadIdx.z * WARP_SIZE_X + (threadIdx.x%WARP_DIM_X)*N_OUTPUT_X;

	float v[REG_SIZE_X*REG_SIZE_Y];
	float w[REG_SIZE_X*REG_SIZE_Y];


	__shared__ float shared[SHARED_SIZE_X*SHARED_SIZE_Y];

	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			shared[shared_index(srow+ i,scol +j)] = input[index(row+i, col+j)];
		}
	}


	if(srow == 0){
		for(int i = 0; i < STENCIL_RADIUS_Y; i++){
			for(int j = 0; j < N_OUTPUT_X; j++){
				shared[shared_index(srow-STENCIL_RADIUS_Y+i, scol+j)] = input[index(row-STENCIL_RADIUS_Y+i, col +j)];
			}
		}
	}
	if(scol == 0){
		for(int i = 0; i < N_OUTPUT_Y; i++){
			for(int j = 0; j < STENCIL_RADIUS_X; j++){
				shared[shared_index(srow+i, scol-STENCIL_RADIUS_X+j)] = input[index(row+i, col-STENCIL_RADIUS_X +j)];
			}
		}

	}


	if(srow == (BLOCK_DIM_Y-1) * WARP_SIZE_Y + ((N_THREADS_WARP-1)/WARP_DIM_X)*N_OUTPUT_Y){
		for(int i = 0; i < STENCIL_RADIUS_Y; i++){
			for(int j = 0; j < N_OUTPUT_X; j++){
				shared[shared_index(srow+N_OUTPUT_Y+i, scol+j)] = input[index(row+N_OUTPUT_Y+i, col +j)];
			}
		}
	}

	if(scol == (BLOCK_DIM_Z-1) * WARP_SIZE_X + ((N_THREADS_WARP-1)%WARP_DIM_X)*N_OUTPUT_X){
		for(int i = 0; i < N_OUTPUT_Y; i++){
			for(int j = 0; j < STENCIL_RADIUS_X; j++){
				shared[shared_index(srow+i, scol+N_OUTPUT_X+j)] = input[index(row+i, col+N_OUTPUT_X+j)];
			}
		}
	}

	__syncthreads();

	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			v[reg_index(i,j)] = shared[shared_index(srow+i, scol+j)];
		}
	}

	for(int i = 0; i < STENCIL_RADIUS_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			v[reg_index(-STENCIL_RADIUS_Y + i, j)] = shared[shared_index(srow-STENCIL_RADIUS_Y+i,scol+j)];
			v[reg_index(N_OUTPUT_Y + i, j)] = shared[shared_index(srow+N_OUTPUT_Y+i, scol+j)];
		}
	}

	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < STENCIL_RADIUS_X; j++){
			v[reg_index(i,-STENCIL_RADIUS_X+j)] = shared[shared_index(srow+i,scol-STENCIL_RADIUS_X+j)];
			v[reg_index(i,N_OUTPUT_X+j)] = shared[shared_index(srow+i, scol+N_OUTPUT_X+j)];
		}
	}


	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			w[reg_index(i,j)] = 0.0f;

			for(int k = - STENCIL_RADIUS_X; k <= STENCIL_RADIUS_X; k++){
				w[reg_index(i,j)] += v[reg_index(i,j+k)];
			}
			for(int k = - STENCIL_RADIUS_Y; k <= STENCIL_RADIUS_Y; k++){
				w[reg_index(i,j)] += v[reg_index(i+k,j)];
			}
			w[reg_index(i,j)] /= (float)((2*STENCIL_RADIUS_X + 1) + (2*STENCIL_RADIUS_Y + 1));
		}
	}

	for(int it = 0; it < N_ITERATIONS -1; it++){

		__syncthreads();
		for(int i = 0; i < N_OUTPUT_Y; i++){
			for(int j = 0; j < N_OUTPUT_X; j++){
				shared[shared_index(srow+i, scol+j)] = w[reg_index(i,j)];
			}
		}

		__syncthreads();

		for(int i = 0; i < N_OUTPUT_Y; i++){
			for(int j = 0; j < N_OUTPUT_X; j++){
				v[reg_index(i,j)] = shared[shared_index(srow+i, scol+j)];
			}
		}

		for(int i = 0; i < STENCIL_RADIUS_Y; i++){
			for(int j = 0; j < N_OUTPUT_X; j++){
				v[reg_index(-STENCIL_RADIUS_Y + i, j)] = shared[shared_index(srow-STENCIL_RADIUS_Y+i,scol+j)];
				v[reg_index(N_OUTPUT_Y + i, j)] = shared[shared_index(srow+N_OUTPUT_Y+i, scol+j)];
			}
		}

		for(int i = 0; i < N_OUTPUT_Y; i++){
			for(int j = 0; j < STENCIL_RADIUS_X; j++){
				v[reg_index(i,-STENCIL_RADIUS_X+j)] = shared[shared_index(srow+i,scol-STENCIL_RADIUS_X+j)];
				v[reg_index(i,N_OUTPUT_X+j)] = shared[shared_index(srow+i, scol+N_OUTPUT_X+j)];
			}
		}


		for(int i = 0; i < N_OUTPUT_Y; i++){
			for(int j = 0; j < N_OUTPUT_X; j++){
				w[reg_index(i,j)] = 0.0f;

				for(int k = - STENCIL_RADIUS_X; k <= STENCIL_RADIUS_X; k++){
					w[reg_index(i,j)] += v[reg_index(i,j+k)];
				}
				for(int k = - STENCIL_RADIUS_Y; k <= STENCIL_RADIUS_Y; k++){
					w[reg_index(i,j)] += v[reg_index(i+k,j)];
				}
				w[reg_index(i,j)] /= (float)((2*STENCIL_RADIUS_X + 1) + (2*STENCIL_RADIUS_Y + 1));
			}
		}
	}


	int row_lower = blockIdx.y * INPUT_BLOCK_SIZE_Y;
	int row_upper = row_lower + INPUT_BLOCK_SIZE_Y;
	int col_lower = blockIdx.x * INPUT_BLOCK_SIZE_Z;
	int col_upper = col_lower + INPUT_BLOCK_SIZE_Z;
	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			if((row+i) >= row_lower && (row+i) < row_upper && (col+j) >= col_lower && (col+j) < col_upper){
				output[index(row+i, col+j)] = w[reg_index(i,j)];
				//printf("%d\n", index(row+i, col+j));
			}
		}
	}

}
#endif

#ifdef SHUFFLE
__global__ void shuffle(float* input, float* output, int iterations){

	int shift_y = STENCIL_RADIUS_Y*(N_ITERATIONS -1);
	int shift_x = STENCIL_RADIUS_X*(N_ITERATIONS-1);
	int row = -shift_y + blockIdx.y * INPUT_BLOCK_SIZE_Y + threadIdx.y * INPUT_WARP_SIZE_Y + (threadIdx.x/WARP_DIM_X)*N_OUTPUT_Y;
	int col = -shift_x + blockIdx.x * INPUT_BLOCK_SIZE_Z + threadIdx.z * INPUT_WARP_SIZE_X + (threadIdx.x%WARP_DIM_X)*N_OUTPUT_X;
	int lane_id = threadIdx.x;

	float v[REG_SIZE_X*REG_SIZE_Y];
	float w[REG_SIZE_X*REG_SIZE_Y];

	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			v[reg_index(i,j)] = input[index(row+i, col+j)];
		}
	}

	for(int i = 0; i < STENCIL_RADIUS_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){

			v[reg_index(-STENCIL_RADIUS_Y + i, j)] = __shfl(v[reg_index(N_OUTPUT_Y-STENCIL_RADIUS_Y+i,j)], lane_id - WARP_DIM_X);
			if(lane_id < 8){
				v[reg_index(-STENCIL_RADIUS_Y + i, j)] = input[index(row-STENCIL_RADIUS_Y+i, col+j)];
			}

			v[reg_index(N_OUTPUT_Y + i, j)] = __shfl(v[reg_index(i,j)], lane_id + WARP_DIM_X);
			if(lane_id >= 24){
				v[reg_index(N_OUTPUT_Y + i, j)] = input[index(row+N_OUTPUT_Y+i, col + j)];
			}
		}
	}

	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < STENCIL_RADIUS_X; j++){

			v[reg_index(i,-STENCIL_RADIUS_X+j)] = __shfl(v[reg_index(i,N_OUTPUT_X-STENCIL_RADIUS_X+j)], lane_id -1);
			if(lane_id == 0 || lane_id == 8 || lane_id == 16 || lane_id == 24){
				v[reg_index(i,-STENCIL_RADIUS_X+j)] = input[index(row+i, col-STENCIL_RADIUS_X+j)];
			}

			v[reg_index(i,N_OUTPUT_X+j)] = __shfl(v[reg_index(i,j)], lane_id+1);
			if(lane_id == 7 || lane_id == 15 || lane_id == 23 || lane_id == 31){
				v[reg_index(i,N_OUTPUT_X+j)] = input[index(row+i, col + N_OUTPUT_X+j)];
			}
		}
	}

	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			w[reg_index(i,j)] = 0.0f;

			for(int k = - STENCIL_RADIUS_X; k <= STENCIL_RADIUS_X; k++){
				w[reg_index(i,j)] += v[reg_index(i,j+k)];
			}
			for(int k = - STENCIL_RADIUS_Y; k <= STENCIL_RADIUS_Y; k++){
				w[reg_index(i,j)] += v[reg_index(i+k,j)];
			}
			w[reg_index(i,j)] /= (float)((2*STENCIL_RADIUS_X + 1) + (2*STENCIL_RADIUS_Y + 1));
		}
	}


	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			v[reg_index(i,j)] = w[reg_index(i,j)];
		}
	}


	for(int it = 0; it < N_ITERATIONS-1; it++){

		for(int i = 0; i < STENCIL_RADIUS_Y; i++){
			for(int j = 0; j < N_OUTPUT_X; j++){
				v[reg_index(-STENCIL_RADIUS_Y + i, j)] = __shfl(v[reg_index(N_OUTPUT_Y-STENCIL_RADIUS_Y+i,j)], lane_id - WARP_DIM_X);
				v[reg_index(N_OUTPUT_Y + i, j)] = __shfl(v[reg_index(i,j)], lane_id + WARP_DIM_X);// input[index(row+N_OUTPUT_Y+i, col+j)];
			}
		}

		for(int i = 0; i < N_OUTPUT_Y; i++){
			for(int j = 0; j < STENCIL_RADIUS_X; j++){
				v[reg_index(i,-STENCIL_RADIUS_X+j)] = __shfl(v[reg_index(i,N_OUTPUT_X-STENCIL_RADIUS_X+j)], lane_id -1);
				v[reg_index(i,N_OUTPUT_X+j)] = __shfl(v[reg_index(i,j)], lane_id+1);
			}
		}


		for(int i = 0; i < N_OUTPUT_Y; i++){
			for(int j = 0; j < N_OUTPUT_X; j++){
				w[reg_index(i,j)] = 0.0f;

				for(int k = - STENCIL_RADIUS_X; k <= STENCIL_RADIUS_X; k++){
					w[reg_index(i,j)] += v[reg_index(i,j+k)];
				}
				for(int k = - STENCIL_RADIUS_Y; k <= STENCIL_RADIUS_Y; k++){
					w[reg_index(i,j)] += v[reg_index(i+k,j)];
				}
				w[reg_index(i,j)] /= (float)((2*STENCIL_RADIUS_X + 1) + (2*STENCIL_RADIUS_Y + 1));
			}
		}


		for(int i = 0; i < N_OUTPUT_Y; i++){
			for(int j = 0; j < N_OUTPUT_X; j++){
				v[reg_index(i,j)] = w[reg_index(i,j)];
			}
		}

	}

	int row_lower = blockIdx.y * INPUT_BLOCK_SIZE_Y + threadIdx.y * INPUT_WARP_SIZE_Y;
	int row_upper = row_lower + INPUT_WARP_SIZE_Y;
	int col_lower = blockIdx.x * INPUT_BLOCK_SIZE_Z + threadIdx.z * INPUT_WARP_SIZE_X;
	int col_upper = col_lower + INPUT_WARP_SIZE_X;
	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			if((row+i) >= row_lower && (row+i) < row_upper && (col+j) >= col_lower && (col+j) < col_upper){
				output[index(row+i, col+j)] = v[reg_index(i,j)];
			}
		}
	}
}
#endif


#ifdef TEXTURE
__global__ void text(float* input, float* output, int iterations){



	int shift_y = (BLOCK_SIZE_Y - INPUT_BLOCK_SIZE_Y)/2;
	int shift_x = (BLOCK_SIZE_Z - INPUT_BLOCK_SIZE_Z)/2;
	int row = -shift_y + blockIdx.y * INPUT_BLOCK_SIZE_Y + threadIdx.y * WARP_SIZE_Y + (threadIdx.x/WARP_DIM_X)*N_OUTPUT_Y;
	int col = -shift_x + blockIdx.x * INPUT_BLOCK_SIZE_Z + threadIdx.z * WARP_SIZE_X + (threadIdx.x%WARP_DIM_X)*N_OUTPUT_X;
	row += PADDING_Y;
	col += PADDING_X;



	float v[REG_SIZE_X*REG_SIZE_Y];
	float w[REG_SIZE_X*REG_SIZE_Y];

	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			v[reg_index(i,j)] = tex2D(buffer_texture, col+j,row+i);
		}
	}

	for(int i = 0; i < STENCIL_RADIUS_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			v[reg_index(-STENCIL_RADIUS_Y + i, j)] = tex2D(buffer_texture,col+j,row-STENCIL_RADIUS_Y+i);
			v[reg_index(N_OUTPUT_Y + i, j)] = tex2D(buffer_texture,col+j,row+N_OUTPUT_Y+i);
		}
	}

	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < STENCIL_RADIUS_X; j++){
			v[reg_index(i,-STENCIL_RADIUS_X+j)] = tex2D(buffer_texture,col-STENCIL_RADIUS_X+j,row+i);
			v[reg_index(i,N_OUTPUT_X+j)] = tex2D(buffer_texture,  col+N_OUTPUT_X+j,row+i);
		}
	}


	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			w[reg_index(i,j)] = 0.0f;

			for(int k = - STENCIL_RADIUS_X; k <= STENCIL_RADIUS_X; k++){
				w[reg_index(i,j)] += v[reg_index(i,j+k)];
			}
			for(int k = - STENCIL_RADIUS_Y; k <= STENCIL_RADIUS_Y; k++){
				w[reg_index(i,j)] += v[reg_index(i+k,j)];
			}
			w[reg_index(i,j)] /= (float)((2*STENCIL_RADIUS_X + 1) + (2*STENCIL_RADIUS_Y + 1));
		}
	}

	row -= PADDING_Y;
	col -= PADDING_X;
	for(int i = 0; i < N_OUTPUT_Y; i++){
		for(int j = 0; j < N_OUTPUT_X; j++){
			output[index(row+i, col+j)] = w[reg_index(i,j)];
		}
	}
}
#endif

void print2d(float* buffer, int dimx, int dimy, int paddingx, int paddingy){
    for(int i = paddingy; i < dimy-paddingy; i++){
        for(int j = paddingx; j < dimx-paddingx; j++){
            printf("%f ", buffer[i*dimx+j]);
        }
        printf("\n");
    }
    printf("\n");
}

void gold(float* input, float* output, int iterations){

	for(int it = 0; it < N_ITERATIONS; it++){
		for(int i = STENCIL_RADIUS_Y; i < PADDED_SIZE_Y-STENCIL_RADIUS_Y; i++){
			for(int j = STENCIL_RADIUS_X; j < PADDED_SIZE_X-STENCIL_RADIUS_X; j++){
				float w = 0;
				for(int k = - STENCIL_RADIUS_X; k <= STENCIL_RADIUS_X; k++){
					w += input[i * PADDED_SIZE_X + j+k];
				}
				for(int k = - STENCIL_RADIUS_Y; k <= STENCIL_RADIUS_Y; k++){
					w += input[(i+k) * PADDED_SIZE_X + j];
				}
				w /= (float)((2*STENCIL_RADIUS_X + 1) + (2*STENCIL_RADIUS_Y + 1));
				output[i*PADDED_SIZE_X + j] = w;
			}
		}

		for(int i = 1; i < PADDED_SIZE_Y-1; i++){
			for(int j = 1; j < PADDED_SIZE_X-1; j++){
				input[i*PADDED_SIZE_X + j] = output[i*PADDED_SIZE_X + j];
			}
		}
	}
}

void diff(float* diff, float* a, float* b){
	for(int i = 0; i < PADDED_SIZE; i++){
		diff[i] = a[i] - b[i];
	}
}

int check(float* a, float* b, int dimx, int dimy, int paddingx, int paddingy){
	float maxerr = 0;
    for(int i = paddingy; i < dimy-paddingy; i++){
        for(int j = paddingx; j < dimx-paddingx; j++){
        	float diff = a[i*dimx+j] - b[i*dimx+j];
        	if(diff > maxerr){
        		maxerr = diff;
        	}
        }
    }

    if(maxerr > 1e-1){
    	//printf("%f\n ", maxerr);
    	return 1;
    }
    return 0;
}

int main(int argc, char** argv){

	if(SIZE_X <= 0 || SIZE_Y <= 0){
		printf("-0.0\n");
		exit(-1);
	}
	if(INPUT_WARP_SIZE_X <= 0 || INPUT_WARP_SIZE_Y <= 0){
          printf("-0.0\n");
          exit(-1);
        }
        //printf("%d, %d, %d, %d, %d\n", PADDING_X, PADDING_Y, SIZE_X, SIZE_Y, PADDED_SIZE);
        //printf("%d, %d\n", PADDED_SIZE_X, PADDED_SIZE_Y);

	int iterations = 1;
	
	cudaSetDevice(0);
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    //printf("  Device name: %sn", prop.name);

	float* buffer_h = (float*)malloc(sizeof(float)*PADDED_SIZE);
	float* output_buffer_h = (float*)calloc(PADDED_SIZE,sizeof(float));
	float* gold_buffer = (float*)calloc(PADDED_SIZE, sizeof(float));
	float* diff_buffer = (float*)calloc(PADDED_SIZE, sizeof(float));

	for(int i = 0; i < PADDED_SIZE_Y; i++){
		for(int j = 0; j < PADDED_SIZE_X; ++j){
			buffer_h[i*PADDED_SIZE_X + j] = 3*((i/10) % 2) + (j/10)%2 + (float)(4.0*(i+j))/PADDED_SIZE_X;
		}
	}

    float* buffer_d;
    float* output_buffer_d;
    cudaMalloc((void**)&buffer_d, sizeof(float) * PADDED_SIZE);
    cudaMalloc((void**)&output_buffer_d, sizeof(float) * PADDED_SIZE);

#ifdef PLAIN
    float* temp_buffer_d;
    int temp_buffer_size = BLOCK_SIZE_Z*(N_BLOCKS_X+2) * BLOCK_SIZE_Y*(N_BLOCKS_Y+2);
    cudaMalloc((void**)&temp_buffer_d, sizeof(float) * temp_buffer_size);
#endif

    float* pitched_buffer_d;
    size_t pitch;
    cudaMallocPitch((void**)&pitched_buffer_d, &pitch, PADDED_SIZE_X*sizeof(float), PADDED_SIZE_Y);

    cudaMemcpy(buffer_d, buffer_h, sizeof(float)*PADDED_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy2D(pitched_buffer_d, pitch, buffer_h,sizeof(float)*PADDED_SIZE_X, sizeof(float)*PADDED_SIZE_X, PADDED_SIZE_Y, cudaMemcpyHostToDevice);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    buffer_texture.filterMode = cudaFilterModePoint;
    buffer_texture.normalized = false;
    size_t offset;
    cudaBindTexture2D(&offset, buffer_texture, pitched_buffer_d, channelDesc, PADDED_SIZE_X, PADDED_SIZE_Y, pitch);

    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    dim3 n_threads(N_THREADS_WARP,BLOCK_DIM_Y,BLOCK_DIM_Z);
    dim3 n_blocks(N_BLOCKS_X,N_BLOCKS_Y,1);
#ifdef PLAIN
    plain<<<n_blocks, n_threads>>>(buffer_d, output_buffer_d, temp_buffer_d, iterations);
#endif
#ifdef SHARED
    shared<<<n_blocks, n_threads>>>(buffer_d, output_buffer_d, iterations);
#endif
#ifdef SHUFFLE
    shuffle<<<n_blocks, n_threads>>>(buffer_d, output_buffer_d, iterations);
#endif
#ifdef TEXTURE
    text<<<n_blocks, n_threads>>>(buffer_d, output_buffer_d, iterations);
#endif


    cudaDeviceSynchronize();
    cudaEventRecord(end);
    //printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    float time;
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);




    cudaMemcpy(output_buffer_h, output_buffer_d, sizeof(float)*PADDED_SIZE, cudaMemcpyDeviceToHost);

    gold(buffer_h, gold_buffer, iterations);

    diff(diff_buffer, gold_buffer, output_buffer_h);

#ifndef TEXTURE
    if(check(gold_buffer, output_buffer_h, PADDED_SIZE_X, PADDED_SIZE_Y, PADDING_X, PADDING_Y)){
    	printf("-0.0\n");
    	exit(0);
    }
#endif

    printf("%f\n", time/((((float)(SIZE_X*SIZE_Y))/1000000.0f)));

    //print2d(diff_buffer, PADDED_SIZE_X, PADDED_SIZE_Y, PADDING_X, PADDING_Y);
    //print2d(gold_buffer, PADDED_SIZE_X, PADDED_SIZE_Y, PADDING_X, PADDING_Y);
    //print2d(buffer_h, PADDED_SIZE_X, PADDED_SIZE_Y, PADDING_X, PADDING_Y);
    //print2d(output_buffer_h,PADDED_SIZE_X, PADDED_SIZE_Y, PADDING_X, PADDING_Y);

}


