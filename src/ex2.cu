/* This file should be almost identical to ex2.cu from homework 2. */
/* once the TODOs in this file are complete, the RPC version of the server/client should work correctly. */

#include "ex3.h"
#include "ex2.h"
#include <cuda/atomic>


__device__ void prefix_sum(int arr[], int arr_size) {
    //TODO complete according to HW2
    //(This file should be almost identical to ex2.cu from homework 2.)
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
    //(This file should be almost identical to ex2.cu from homework 2.)
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


class queue_server : public image_processing_server
{
public:
    //TODO complete according to HW2
    //(This file should be almost identical to ex2.cu from homework 2.)

    queue_server(int threads)
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)
    }

    ~queue_server() override
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)
        return false;
    }

    bool dequeue(int *img_id) override
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)
        return false;
    }
};


std::unique_ptr<queue_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
