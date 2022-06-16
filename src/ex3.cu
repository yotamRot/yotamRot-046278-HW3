/* CUDA 10.2 has a bug that prevents including <cuda/atomic> from two separate
 * object files. As a workaround, we include ex2.cu directly here. */
#include "ex2.cu"

#include <cassert>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <infiniband/verbs.h>

class server_rpc_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> gpu_context;

public:
    explicit server_rpc_context(uint16_t tcp_port) : rdma_server_context(tcp_port),
        gpu_context(create_queues_server(256))
    {
    }

    virtual void event_loop() override
    {
        /* so the protocol goes like this:
         * 1. we'll wait for a CQE indicating that we got an Send request from the client.
         *    this tells us we have new work to do. The wr_id we used in post_recv tells us
         *    where the request is.
         * 2. now we send an RDMA Read to the client to retrieve the request.
         *    we will get a completion indicating the read has completed.
         * 3. we process the request on the GPU.
         * 4. upon completion, we send an RDMA Write with immediate to the client with
         *    the results.
         */
        rpc_request* req;
        uchar *img_in;
        uchar *img_out;

        bool terminate = false, got_last_cqe = false;

        while (!terminate || !got_last_cqe) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		VERBS_WC_CHECK(wc);

                switch (wc.opcode) {
                case IBV_WC_RECV:
                    /* Received a new request from the client */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];

                    /* Terminate signal */
                    if (req->request_id == -1) {
                        printf("Terminating...\n");
                        terminate = true;
                        goto send_rdma_write;
                    }

                    /* Step 2: send RDMA Read to client to read the input */
                    post_rdma_read(
                        img_in,             // local_src
                        req->input_length,  // len
                        mr_images_in->lkey, // lkey
                        req->input_addr,    // remote_dst
                        req->input_rkey,    // rkey
                        wc.wr_id);          // wr_id
                    break;

                case IBV_WC_RDMA_READ:
                    /* Completed RDMA read for a request */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];
                    img_out = &images_out[wc.wr_id * IMG_SZ];

                    // Step 3: Process on GPU
                    while(!gpu_context->enqueue(wc.wr_id, img_in, img_out)){};
		    break;
                    
                case IBV_WC_RDMA_WRITE:
                    /* Completed RDMA Write - reuse buffers for receiving the next requests */
                    post_recv(wc.wr_id % OUTSTANDING_REQUESTS);

		    if (terminate)
			got_last_cqe = true;

                    break;
                default:
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }

            // Dequeue completed GPU tasks
            int dequeued_img_id;
            if (gpu_context->dequeue(&dequeued_img_id)) {
                req = &requests[dequeued_img_id];
                img_out = &images_out[dequeued_img_id * IMG_SZ];

send_rdma_write:
                // Step 4: Send RDMA Write with immediate to client with the response
		post_rdma_write(
                    req->output_addr,                       // remote_dst
                    terminate ? 0 : req->output_length,     // len
                    req->output_rkey,                       // rkey
                    terminate ? 0 : img_out,                // local_src
                    mr_images_out->lkey,                    // lkey
                    dequeued_img_id + OUTSTANDING_REQUESTS, // wr_id
                    (uint32_t*)&req->request_id);           // immediate
            }
        }
    }
};

class client_rpc_context : public rdma_client_context {
private:
    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;

    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
public:
    explicit client_rpc_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
    }

    ~client_rpc_context()
    {
        kill();
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        /* register a memory region for the input images. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        if (requests_sent - send_cqes_received == OUTSTANDING_REQUESTS)
            return false;

        struct ibv_sge sg; /* scatter/gather element */
        struct ibv_send_wr wr; /* WQE */
        struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */

        /* step 1: send request to server using Send operation */
        
        struct rpc_request *req = &requests[requests_sent % OUTSTANDING_REQUESTS];
        req->request_id = img_id;
        req->input_rkey = img_in ? mr_images_in->rkey : 0;
        req->input_addr = (uintptr_t)img_in;
        req->input_length = IMG_SZ;
        req->output_rkey = img_out ? mr_images_out->rkey : 0;
        req->output_addr = (uintptr_t)img_out;
        req->output_length = IMG_SZ;

        /* RDMA send needs a gather element (local buffer)*/
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey = mr_requests->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = img_id; /* helps identify the WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed");
            exit(1);
        }

        ++requests_sent;

        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* When WQE is completed we expect a CQE */
        /* We also expect a completion of the RDMA Write with immediate operation from the server to us */
        /* The order between the two is not guarenteed */

        struct ibv_wc wc; /* CQE */
        int ncqes = ibv_poll_cq(cq, 1, &wc);
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        if (ncqes == 0)
            return false;

	VERBS_WC_CHECK(wc);

        switch (wc.opcode) {
        case IBV_WC_SEND:
            ++send_cqes_received;
            return false;
        case IBV_WC_RECV_RDMA_WITH_IMM:
            *img_id = wc.imm_data;
            break;
        default:
            printf("Unexpected completion type\n");
            assert(0);
        }

        /* step 2: post receive buffer for the next RPC call (next RDMA write with imm) */
        post_recv();

        return true;
    }

    void kill()
    {
        while (!enqueue(-1, // Indicate termination
                       NULL, NULL)) ;
        int img_id = 0;
        bool dequeued;
        do {
            dequeued = dequeue(&img_id);
        } while (!dequeued || img_id != -1);
    }
};

class server_queues_context : public rdma_server_context {
private:
    std::unique_ptr<image_processing_server> server;

    /* TODO: add memory region(s) for CPU-GPU queues */

public:
    explicit server_queues_context(uint16_t tcp_port) : rdma_server_context(tcp_port)
    {
        /* TODO Initialize additional server MRs as needed. */

        /* TODO Exchange rkeys, addresses, and necessary information (e.g.
         * number of queues) with the client */
    }

    ~server_queues_context()
    {
        /* TODO destroy the additional server MRs here */
    }

    virtual void event_loop() override
    {
        /* TODO simplified version of server_rpc_context::event_loop. As the
         * client use one sided operations, we only need one kind of message to
         * terminate the server at the end. */
    }
};

class client_queues_context : public rdma_client_context {
private:
    /* TODO add necessary context to track the client side of the GPU's
     * producer/consumer queues */

    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
    /* TODO define other memory regions used by the client here */

public:
    client_queues_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
        /* TODO communicate with server to discover number of queues, necessary
         * rkeys / address, or other additional information needed to operate
         * the GPU queues remotely. */
    }

    ~client_queues_context()
    {
	/* TODO terminate the server and release memory regions and other resources */
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        // TODO register memory
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        // TODO register memory
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        /* TODO use RDMA Write and RDMA Read operations to enqueue the task on
         * a CPU-GPU producer consumer queue running on the server. */
        return false;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* TODO use RDMA Write and RDMA Read operations to detect the completion and dequeue a processed image
         * through a CPU-GPU producer consumer queue running on the server. */
        return false;
    }
};

std::unique_ptr<rdma_server_context> create_server(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<server_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<server_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}

std::unique_ptr<rdma_client_context> create_client(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<client_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<client_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}
