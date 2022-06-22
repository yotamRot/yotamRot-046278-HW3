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
   	std::unique_ptr<queue_server> server;
	struct server_init_info server_info; 
 
	struct ibv_mr* cpu_gpu_ring_buffer_mr ;
	struct ibv_mr* cpu_gpu_mail_box_mr ;
	
	struct ibv_mr* gpu_cpu_ring_buffer_mr ;
	struct ibv_mr* gpu_cpu_mail_box_mr ;
public:
    explicit server_queues_context(uint16_t tcp_port) : rdma_server_context(tcp_port)
    {
 
        server = create_queues_server(256);  
        int mail_box_size =  server->cpu_to_gpu->N * sizeof(request);

        // create memory regions
        cpu_gpu_ring_buffer_mr = ibv_reg_mr(pd, server->cpu_to_gpu_buf, sizeof(ring_buffer), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |IBV_ACCESS_REMOTE_READ );
        server_info.cpu_gpu_ring_buffer_mr = *cpu_gpu_ring_buffer_mr;
        if (!server_info.cpu_gpu_ring_buffer_mr) {
            perror("ibv_reg_mr() failed for cpu_gpu_ring_buffer_mr");
            exit(1);
        }

        server_info.cpu_gpu_mail_box_mr = ibv_reg_mr(pd, server->cpu_to_gpu->_mailbox, mail_box_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!server_info.cpu_gpu_mail_box_mr) {
            perror("ibv_reg_mr() failed for cpu_gpu_mail_box_mr");
            exit(1);
        }

        server_info.gpu_cpu_ring_buffer_mr = ibv_reg_mr(pd, server->cpu_to_gpu_buf, sizeof(ring_buffer), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |IBV_ACCESS_REMOTE_READ );
        if (!server_info.gpu_cpu_ring_buffer_mr) {
            perror("ibv_reg_mr() failed for gpu_cpu_ring_buffer_mr");
            exit(1);
        }

        server_info.gpu_cpu_mail_box_mr = ibv_reg_mr(pd, server->cpu_to_gpu->_mailbox, mail_box_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!server_info.gpu_cpu_mail_box_mr) {
            perror("ibv_reg_mr() failed for gpu_cpu_mail_box_mr");
            exit(1);
        }

    server_info.cpu_gpu_queue = *cpu_gpu_queue;
    server_info.cpu_gpu_ring_buffer_addr = (uint64_t *)server->cpu_to_gpu_buf ;
    server_info.cpu_gpu_mail_box = *cpu_gpu_mail_box ;
    server_info.cpu_gpu_mail_addr =  (uint64_t *)server->cpu_to_gpu->_mailbox;
    // gpu_pcu
    server_info.gpu_cpu_queue = *gpu_cpu_queue;
    server_info.gpu_cpu_addr =  (uint64_t *)server->gpu_to_cpu_buf;
    server_info.gpu_cpu_mail_box = *gpu_cpu_mail_box;
    server_info.gpu_cpu_mail_addr =  (uint64_t *)server->cpu_to_gpu->_mailbox;

     // cpu_gpu
       server_info.img_in = *mr_images_in;
        server_info.img_in_addr =  (uint64_t *)images_in;
       server_info.img_out = *mr_images_in;
       server_info.img_out_addr = (uint64_t *) images_out;


        send_over_socket(&server_info, sizeof(server_info));
        // send_over_socket(cpu_gpu_queue, sizeof(cpu_gpu_queue));
        // send_over_socket(server->cpu_to_gpu_buf, sizeof(server->cpu_to_gpu_buf));

        // send_over_socket(cpu_gpu_mail_box, sizeof(cpu_gpu_mail_box));
        // send_over_socket(server->cpu_to_gpu->_mailbox, sizeof(server->cpu_to_gpu->_mailbox));

        // send_over_socket(gpu_cpu_queue, sizeof(gpu_cpu_queue));
        // send_over_socket(server->gpu_to_cpu_buf, sizeof(server->gpu_to_cpu_buf));

        // send_over_socket(gpu_cpu_mail_box, sizeof(gpu_cpu_mail_box));
        // send_over_socket(server->gpu_to_cpu->_mailbox, sizeof(server->gpu_to_cpu->_mailbox));

        // send_over_socket(&mail_box_size, sizeof(mail_box_size));

        // // Send gpu buffers
        // images_out mr_images_out
        // images_in mr_images_in

        // send_over_socket(images_in, sizeof(images_in));
        // send_over_socket(mr_images_in, sizeof(mr_images_in));

        // send_over_socket(images_out, sizeof(images_out));
        // send_over_socket(mr_images_out, sizeof(server->cpu_to_gpu_buf));


        // /* TODO Exchange rkeys, addresses, and necessary information (e.g.
        //  * number of queues) with the client */
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
        struct ibv_wc wc;

        // wait for end message
        int ncqes = ibv_poll_cq(cq, 1, &wc);
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
    }
           

};

class client_queues_context : public rdma_client_context {
private:
    /* TODO add necessary context to track the client side of the GPU's
     * producer/consumer queues */
    char* cpu_gpu_local;
    struct ibv_mr* cpu_gpu_queue;
    char* cpu_gpu_mail_local;
    char* gpu_cpu_local;
    char* gpu_cpu_mail_local;
    
    // struct ibv_mr *mr_images_in; /* Memory region for input images */
    // struct ibv_mr *mr_images_out; /* Memory region for output images */
    // /* TODO define other memory regions used by the client here */
    // struct ibv_mr *mr_cpu_gpu_queue;
    // char* cpu_to_gpu_buf_add;

    // struct ibv_mr *mr_cpu_gpu_mail_box;
    // request* cpu_to_gpu_mail_box_add;

    // struct ibv_mr *mr_gpu_cpu_queue;
    // char* gpu_to_cpu_buf_add;

    // struct ibv_mr *mr_gpu_cpu_mail_box;
    // request* gpu_to_cpu_mail_box_add;
     struct server_init_info server_info;


public:
    client_queues_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
        /* TODO communicate with server to discover number of queues, necessary
         * rkeys / address, or other additional information needed to operate
         * the GPU queues remotely. */
        server_info = {0};
            recv_over_socket(&server_info, sizeof(server_info));

        //Alocate locl buffer for DMA
        cpu_gpu_local = (char*)malloc(sizeof(ring_buffer));
        cpu_gpu_mail_local = (char*)malloc(sizeof(request)* mail_box_size);

        gpu_cpu_local = (char*)malloc(sizeof(ring_buffer));
        gpu_cpu_mail_local = (char*)malloc(sizeof(request)* mail_box_size);
    }

    ~client_queues_context()
    {
	/* TODO terminate the server and release memory regions and other resources */
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        // TODO register memory
        /* register a memory region for the input images. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        // TODO register memory
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

//flow for enqueue
    // rdma read cpugpu ring buffer
    //check if full(with tain head)
    	//if full return false
    //rdma write img to img_in[img_id]
    //rdma write req(img id,img_in_host_addr[img_id],img_out_host_addr[img_id]) to mail_box
    //rdma write cpugpu ring buffer with updated head-tail

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        /* TODO use RDMA Write and RDMA Read operations to enqueue the task on
         * a CPU-GPU producer consumer queue running on the server. */
        struct ibv_wc wc; /* CQE */
        int ncqes;

        ring_buffer* cpu_to_gpu;
        wc.wr_id = img_id;
        // Check if there is place in cpu-gpu queue
        post_rdma_read(
                        cpu_gpu_local,           // local_src
                        sizeof(ring_buffer),  // len
                        mr_cpu_gpu_queue->lkey, // lkey
                        (uint64_t)cpu_to_gpu_buf_add,    // remote_dst
                        mr_cpu_gpu_queue->rkey,    // rkey
                        wc.wr_id);          // wr_id
        
        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }

        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }

        cpu_to_gpu = new (cpu_gpu_local) ring_buffer(1);
        // check for place
        if (cpu_to_gpu->_tail - cpu_to_gpu->_head == cpu_to_gpu->N) {
            return false;
        }

        // Write Image to gpu buffers in sever
        post_rdma_write(
                    req->output_addr,                       // remote_dst
                    terminate ? 0 : req->output_length,     // len
                    req->output_rkey,                       // rkey
                    terminate ? 0 : img_out,                // local_src
                    mr_images_out->lkey,                    // lkey
                    dequeued_img_id + OUTSTANDING_REQUESTS, // wr_id
                    (uint32_t*)&req->request_id);           // immediate

        // Update cpu-gpu queue

        // Update head/tail

        return false;
    }

//flow for dnqueue
    // rdma read gpucpu ring buffer
    
    //check if empty(with tain head)
    	//if empty return false
    //rdma read mail_box from tail/head? and get img id
    //rdma read img from cuda_host img_in[img_id] to our img in
    //rdma write cpugpu ring buffer with updated head-tail


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
