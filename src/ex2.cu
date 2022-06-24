/* This file should be almost identical to ex2.cu from homework 2. */
/* once the TODOs in this file are complete, the RPC version of the server/client should work correctly. */

#include "ex3.h"
#include "ex2.h"
#include <cuda/atomic>

#define HISTOGRAM_SIZE 256
#define WRAP_SIZE 32
#define SHARED_MEM_USAGE 2048
#define REGISTERS_PER_THREAD 32
#define INVALID_IMAGE -2
#define KILL_IMAGE -1

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;
    for (int stride = 1; stride < arr_size; stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}


/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
 void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);


__device__ void process_image(uchar *all_in, uchar *all_out, uchar* maps) {
    //TODO complete according to HW2
    int ti = threadIdx.x;
    int tg = ti / TILE_WIDTH;
    int workForThread = (TILE_WIDTH * TILE_WIDTH) / blockDim.x; // in bytes
    uchar imageVal;

    __shared__ int sharedHist[HISTOGRAM_SIZE]; // maybe change to 16 bit ? will be confilcits on same bank 

    int tileStartIndex;
    int insideTileIndex;
    int curIndex;
    for (int i = 0 ; i < TILE_COUNT * TILE_COUNT; i++)
    {
        // calc tile index in image buffer (shared between al threads in block)
        tileStartIndex = i % TILE_COUNT * TILE_WIDTH + (i / TILE_COUNT) * (TILE_WIDTH *TILE_WIDTH) * TILE_COUNT;
        // zero shared buffer histogram values
        if (ti < 256)
        {
            sharedHist[ti] = 0;
        }
        __syncthreads();
       for (int j = 0; j < workForThread; j++)
            {
                // calc index in tile buffer for each thread
                insideTileIndex = tg * TILE_WIDTH * TILE_COUNT + ti % TILE_WIDTH + (blockDim.x / TILE_WIDTH) * TILE_WIDTH * TILE_COUNT * j;
                // sum tile index and index inside tile to find relevant byte for thread in cur iteration
                curIndex = tileStartIndex + insideTileIndex;
                // update histogram
                imageVal = all_in[curIndex];
                atomicAdd(sharedHist + imageVal, 1);
        }
    
        __syncthreads();
        
        // calc CDF using prefix sumpwdon histogram buffer

        prefix_sum(sharedHist, HISTOGRAM_SIZE);

        __syncthreads();
        // calc map value for each index
        if (ti < 256)
        {
            maps[HISTOGRAM_SIZE * i + ti] = (float(sharedHist[ti]) * 255)  / (TILE_WIDTH * TILE_WIDTH);
        }
    }

    __syncthreads();
    // interpolate image using given maps buffer
    interpolate_device(maps, all_in, all_out);
    return; 
}


__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar* maps)
{
    process_image(all_in, all_out, maps);
}


// TODO complete according to HW2:
//          implement a lock,
//          implement a MPMC queue,
//          implement the persistent kernel,
//          implement a function for calculating the threadblocks count
// (This file should be almost identical to ex2.cu from homework 2.)

typedef cuda::atomic<int, cuda::thread_scope_device> gpu_atomic_int;
__device__ gpu_atomic_int* push_lock;
__device__ gpu_atomic_int* pop_lock;

__global__
void init_locks() 
{ 
    push_lock = new gpu_atomic_int(0); 
    pop_lock = new gpu_atomic_int(0); 
}

__global__
void free_locks() 
{ 
    delete push_lock; 
    delete pop_lock; 
}

__device__
 void lock(gpu_atomic_int * l) 
{
   do 
    {
        while (l->load(cuda::memory_order_relaxed)) continue;
    } while (l->exchange(1, cuda::memory_order_relaxed)); // actual atomic locking
  
   // while (l->exchange(1, cuda::memory_order_acq_rel));
}

__device__
 void unlock(gpu_atomic_int * l) 
{
    l->store(0, cuda::memory_order_release);
}

struct request
{
	    int imgID;	
    	uchar *imgIn;
    	uchar *imgOut;
};

class ring_buffer {
	private:
            
		
	public:

		request* _mailbox;
        int N;
        cuda::atomic<int> _head, _tail;
		ring_buffer()
		{   
            _mailbox = NULL;
			_head = 0, _tail = 0;
		}; // def contructor
        
		~ring_buffer()
        {
                if(_mailbox != NULL)
                {
                    CUDA_CHECK(cudaFreeHost(_mailbox));
                }
		} 
		ring_buffer(int size)
        {
			 N = size;
             CUDA_CHECK(cudaMallocHost(&_mailbox, sizeof(request)*N));
			_head = 0, _tail = 0;
		}

		__device__ __host__
		bool push(const request data) 
        {
	 		int tail = _tail.load(cuda::memory_order_relaxed);
            // printf("push function - tail is: %d img id is - %d\n" , tail, data.imgID);
	 		if (tail - _head.load(cuda::memory_order_acquire) != N){
				_mailbox[tail % N] = data;
	 			_tail.store(tail + 1, cuda::memory_order_release);
				return true;
			} else{
				return false;
			}
	 	}

		__device__ __host__
	 	request pop() 
         {
	 		int head = _head.load(cuda::memory_order_relaxed);
            // printf("pop function - head is: %d \n" , head);
			request item;
	 		if (_tail.load(cuda::memory_order_acquire) != head){
	 			item = _mailbox[head % N];
	 			_head.store(head + 1, cuda::memory_order_release);
			} else{
				item.imgID = INVALID_IMAGE;//item is not valid
			}
	 		return item;
	 	}
};


__global__
void process_image_kernel_queue(ring_buffer* cpu_to_gpu, ring_buffer* gpu_to_cpu, uchar* maps)
{
	__shared__ request req_i;
    int tid = threadIdx.x;
    uchar* block_maps = maps + blockIdx.x * TILE_COUNT * TILE_COUNT * HISTOGRAM_SIZE;
	while(1)
    {

        if (tid == 0)
        {   
            lock(pop_lock);
            req_i = cpu_to_gpu->pop();
            unlock(pop_lock);
        }

        // got request to stop
        __syncthreads();
        if (req_i.imgID == KILL_IMAGE)
        {
          return;
        }
		else if (req_i.imgID != INVALID_IMAGE && req_i.imgID != KILL_IMAGE) 
        {


            //  printf("image id poped by gpu = %d\n",req_i.imgID);

            __syncthreads();
             process_image(req_i.imgIn, req_i.imgOut, block_maps);
             __syncthreads();

            if(tid == 0) {
                // printf("gpu proccess - befor push image id : %d\n", req_i.imgID);
                lock(push_lock);
                while(!gpu_to_cpu->push(req_i));
                unlock(push_lock);
                // printf("gpu proccess - affter push image id : %d\n", req_i.imgID);

            }
		}	
	}
}



int calc_max_thread_blocks(int threads)
{
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));


        //constraints
    // int max_tb_sm = deviceProp.maxBlocksPerMultiProcessor;
    int max_shared_mem_sm = deviceProp.sharedMemPerMultiprocessor;
    int max_regs_per_sm = deviceProp.regsPerMultiprocessor;
    int max_threads_per_sm = deviceProp.maxThreadsPerMultiProcessor;
    // int max_block_per_sm = deviceProp.maxBlocksPerMultiProcessor;

    int max_tb_mem_constraint = max_shared_mem_sm / SHARED_MEM_USAGE;
    int max_tb_reg_constraint = max_regs_per_sm / (REGISTERS_PER_THREAD * threads);
    int max_tb_threads_constraint = max_threads_per_sm / threads;

    int max_tb = std::min(max_tb_mem_constraint,std::min(max_tb_reg_constraint, max_tb_threads_constraint));
    int max_num_sm = deviceProp.multiProcessorCount;
    return max_num_sm * max_tb;

}

class queue_server : public image_processing_server
{

private:

	uchar* server_maps;
    int tb_num;
public:

    ring_buffer* cpu_to_gpu;
	ring_buffer* gpu_to_cpu;

	char* cpu_to_gpu_buf;
	char* gpu_to_cpu_buf;
    //TODO complete according to HW2
    //(This file should be almost identical to ex2.cu from homework 2.)

    queue_server(int threads)
    {
        tb_num = calc_max_thread_blocks(threads);//TODO calc from calc_max_thread_blocks
        int ring_buf_size = std::pow(2, std::ceil(std::log(16*tb_num)/std::log(2)));//TODO - calc 2^celling(log2(16*tb_num)/log2(2))
        ring_buf_size = std::min(ring_buf_size, OUTSTANDING_REQUESTS);
        printf("tb_num %d\n", tb_num);
        printf("ring_buf_size %d\n", ring_buf_size);

        CUDA_CHECK(cudaMalloc((void**)&server_maps, tb_num * TILE_COUNT * TILE_COUNT * HISTOGRAM_SIZE));

        CUDA_CHECK(cudaMallocHost(&cpu_to_gpu_buf, sizeof(ring_buffer)));
        CUDA_CHECK(cudaMallocHost(&gpu_to_cpu_buf, sizeof(ring_buffer)));
 
        cpu_to_gpu = new (cpu_to_gpu_buf) ring_buffer(ring_buf_size);
        gpu_to_cpu = new (gpu_to_cpu_buf) ring_buffer(ring_buf_size);

            //  launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
        // process_image_kernel_queue<<<tb_num,threads>>>(cpu_to_gpu,gpu_to_cpu, server_maps);
        
        init_locks<<<1, 1>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
        process_image_kernel_queue<<<tb_num, threads>>>(cpu_to_gpu, gpu_to_cpu, server_maps);
    }

    ~queue_server() override
    {
          //Kill kernel
        for (int i = 0 ; i<tb_num; i++)
        {
            // send enough kills to kill all tb
            this->enqueue(KILL_IMAGE, NULL, NULL);
        }
        CUDA_CHECK(cudaDeviceSynchronize()); 
        free_locks<<<1, 1>>>();
        CUDA_CHECK(cudaDeviceSynchronize()); 
        cpu_to_gpu->~ring_buffer();
        gpu_to_cpu->~ring_buffer();
        CUDA_CHECK(cudaFreeHost(cpu_to_gpu_buf));
        CUDA_CHECK(cudaFreeHost(gpu_to_cpu_buf));
        CUDA_CHECK(cudaFree(server_maps));
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        request request_i;
	    request_i.imgID = img_id;
	    request_i.imgIn = img_in;
	    request_i.imgOut = img_out;

        if(cpu_to_gpu->push(request_i)){
		    return true;
	    } else{

		    return false;
	    }
        return false;
    }

    bool dequeue(int *img_id) override
    {
        request request_i = gpu_to_cpu->pop();
        // printf("cpu dequeue - after pop, img ID = %d , taskmaps ptr = %p \n", request_i.imgID,request_i.taskMaps);
        if(request_i.imgID == INVALID_IMAGE){
            return false;// queue is empty
        } else {
            *img_id = request_i.imgID;
            return true;	
        }
    }
};


std::unique_ptr<queue_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
