/* CUDA 10.2 has a bug that prevents including <cuda/atomic> from two separate
 * object files. As a workaround, we include ex2.cu directly here. */
#include "ex2.cu"

#include <cassert>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>


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
                        ////printf("Terminating...\n");
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
                    ////printf("Unexpected completion\n");
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
            ////printf("Unexpected completion type\n");
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
 
	struct ibv_mr* cpu_gpu_ring_buffer_mr;
	struct ibv_mr* cpu_gpu_mail_box_mr;
	struct ibv_mr* cpu_gpu_tail_mr;
	
	struct ibv_mr* gpu_cpu_ring_buffer_mr ;
	struct ibv_mr* gpu_cpu_mail_box_mr;
	struct ibv_mr* gpu_cpu_head_mr;

    int terminate;
    struct ibv_mr* terminate_mr;

public:
    explicit server_queues_context(uint16_t tcp_port) : rdma_server_context(tcp_port)
    {
 
        server = create_queues_server(256);  
        int mail_box_size =  server->cpu_to_gpu->N * sizeof(request);
        terminate = 0;

        // create memory regions

        // cpu - gpu
        //ring buff
        cpu_gpu_ring_buffer_mr = ibv_reg_mr(pd, server->cpu_to_gpu_buf, sizeof(ring_buffer), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |IBV_ACCESS_REMOTE_READ );
        server_info.cpu_gpu_ring_buffer_mr = *cpu_gpu_ring_buffer_mr;
        if (!cpu_gpu_ring_buffer_mr) {
            perror("ibv_reg_mr() failed for cpu_gpu_ring_buffer_mr");
            exit(1);
        }

        // mail_box
        cpu_gpu_mail_box_mr = ibv_reg_mr(pd, server->cpu_to_gpu->_mailbox, mail_box_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        server_info.cpu_gpu_mail_box_mr = *cpu_gpu_mail_box_mr;
        if (!cpu_gpu_mail_box_mr) {
            perror("ibv_reg_mr() failed for cpu_gpu_mail_box_mr");
            exit(1);
        }

        // tail
        cpu_gpu_tail_mr = ibv_reg_mr(pd, &server->cpu_to_gpu->_tail, sizeof(server->cpu_to_gpu->_tail), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        server_info.cpu_gpu_tail_mr = *cpu_gpu_tail_mr;
        if (!cpu_gpu_tail_mr) {
            perror("ibv_reg_mr() failed for cpu_gpu_mail_box_mr");
            exit(1);
        }

         // gpu - cpu

        // ring buff
        gpu_cpu_ring_buffer_mr = ibv_reg_mr(pd, server->gpu_to_cpu_buf, sizeof(ring_buffer), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |IBV_ACCESS_REMOTE_READ );
        server_info.gpu_cpu_ring_buffer_mr =*gpu_cpu_ring_buffer_mr;
        if (!gpu_cpu_ring_buffer_mr) {
            perror("ibv_reg_mr() failed for gpu_cpu_ring_buffer_mr");
            exit(1);
        }

        // mail box
        gpu_cpu_mail_box_mr = ibv_reg_mr(pd, server->gpu_to_cpu->_mailbox, mail_box_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ);
        server_info.gpu_cpu_mail_box_mr = *gpu_cpu_mail_box_mr;
        if (!gpu_cpu_mail_box_mr) {
            perror("ibv_reg_mr() failed for gpu_cpu_mail_box_mr");
            exit(1);
        }

        //head
        gpu_cpu_head_mr = ibv_reg_mr(pd, &server->gpu_to_cpu->_head, sizeof(server->gpu_to_cpu->_head), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        server_info.gpu_cpu_head_mr = *gpu_cpu_head_mr;
        if (!gpu_cpu_head_mr) {
            perror("ibv_reg_mr() failed for cpu_gpu_mail_box_mr");
            exit(1);
        }

        terminate_mr = ibv_reg_mr(pd, &terminate, sizeof(terminate), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!terminate_mr) {
            perror("ibv_reg_mr() failed for terminate_mr");
            exit(1);
        }

        server_info.cpu_gpu_ring_buffer_addr = (uint64_t *)server->cpu_to_gpu_buf;
        server_info.cpu_gpu_mail_box_addr = (uint64_t *)server->cpu_to_gpu->_mailbox;
        server_info.cpu_gpu_tail_addr = (uint64_t *)&server->cpu_to_gpu->_tail;

        // gpu_pcu
        server_info.gpu_cpu_ring_buffer_addr = (uint64_t *)server->gpu_to_cpu_buf;
        server_info.gpu_cpu_mail_box_addr = (uint64_t*)server->gpu_to_cpu->_mailbox;
        server_info.gpu_cpu_head_addr = (uint64_t*)&server->gpu_to_cpu->_head;

        //gpu buffers 
        server_info.img_in_addr =  (uint64_t *)images_in;
        server_info.img_in_mr =  *mr_images_in;
        server_info.img_out_addr = (uint64_t *) images_out;
        server_info.img_out_mr =  *mr_images_out;


        server_info.terminate_mr = *terminate_mr;
        server_info.terminate_addr = (uint64_t *)&terminate;

        ////printf("server contructor: sending client mrs and stuff over socket\n");
        send_over_socket(&server_info, sizeof(server_info));
        
        ////printf("server contructor: sending client mrs and stuff over socket - completed\n");
    }

    ~server_queues_context()
    {
    	ibv_dereg_mr(cpu_gpu_ring_buffer_mr);
    	ibv_dereg_mr(cpu_gpu_mail_box_mr);
    	ibv_dereg_mr(gpu_cpu_ring_buffer_mr);
    	ibv_dereg_mr(gpu_cpu_mail_box_mr);
    	ibv_dereg_mr(terminate_mr);
    }

    virtual void event_loop() override
    {
        /* TODO simplified version of server_rpc_context::event_loop. As the
         * client use one sided operations, we only need one kind of message to
         * terminate the server at the end. */

        // wait for end message;
        printf("server event loop: starting to wait to recieve over socket term msg from client\n");
        

        while(terminate == 0) {
        }
        printf("server event loop: recvd term over socket from client, commiting suicide.. XXX\n");
    }
};

class client_queues_context : public rdma_client_context {
private:
     /* producer/consumer queues */
    struct server_init_info server_info;
    request local_request;
    struct ibv_mr* local_request_mr;
    ring_buffer cpu_gpu_ring_buff;
    struct ibv_mr* cpu_gpu_ring_buff_mr;
    ring_buffer gpu_cpu_ring_buff;
    struct ibv_mr* gpu_cpu_ring_buff_mr;

    cuda::atomic<int> gpu_cpu_head, cpu_gpu_tail;
    ibv_mr * gpu_cpu_head_mr;
    ibv_mr *  cpu_gpu_tail_mr;

    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */

    uchar* _images_out;

    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;


public:
    client_queues_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
        /* TODO communicate with server to discover number of queues, necessary
         * rkeys / address, or other additional information needed to operate
         * the GPU queues remotely. */
        server_info = {0};
    	////printf("client constructor: waiting for server to send over socket all the mrs and stuff\n");
        recv_over_socket(&server_info, sizeof(server_info));
    	////printf("client constructor: recieved from server over socket all server info\n");
    	////printf("%p\n", server_info.cpu_gpu_ring_buffer_addr);
        


         // create memory regions
        local_request_mr = ibv_reg_mr(pd, &local_request, sizeof(request), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ );
        if (!local_request_mr) {
            perror("ibv_reg_mr() failed for request");
            exit(1);
        }

        cpu_gpu_ring_buff_mr = ibv_reg_mr(pd, &cpu_gpu_ring_buff, sizeof(cpu_gpu_ring_buff), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ );
        if (!cpu_gpu_ring_buff_mr) {
            perror("ibv_reg_mr() failed for cpu_gpu_ring_buff_mr");
            exit(1);
        }

        gpu_cpu_ring_buff_mr = ibv_reg_mr(pd, &gpu_cpu_ring_buff, sizeof(cpu_gpu_ring_buff), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |IBV_ACCESS_REMOTE_READ );
        if (!gpu_cpu_ring_buff_mr) {
            perror("ibv_reg_mr() failed for gpu_cpu_ring_buff_mr");
            exit(1);
        }
    	////printf("client constructor: finished constructor\n");

        gpu_cpu_head_mr = ibv_reg_mr(pd, &gpu_cpu_head, sizeof(gpu_cpu_head), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |IBV_ACCESS_REMOTE_READ );
        if (!gpu_cpu_head_mr) {
            perror("ibv_reg_mr() failed for gpu_cpu_head_mr");
            exit(1);
        }
    	////printf("client constructor: finished constructor\n");

        cpu_gpu_tail_mr = ibv_reg_mr(pd, &cpu_gpu_tail, sizeof(cpu_gpu_tail), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |IBV_ACCESS_REMOTE_READ );
        if (!cpu_gpu_tail_mr) {
            perror("ibv_reg_mr() failed for cpu_gpu_tail");
            exit(1);
        }
    	////printf("client constructor: finished constructor\n");


    }

    ~client_queues_context()
    {
        /* TODO terminate the server and release memory regions and other resources */
        ibv_dereg_mr(local_request_mr);
        ibv_dereg_mr(cpu_gpu_ring_buff_mr);
        ibv_dereg_mr(gpu_cpu_ring_buff_mr);
        ibv_dereg_mr(mr_images_in);
        ibv_dereg_mr(mr_images_out);
        ibv_dereg_mr(gpu_cpu_head_mr);
        ibv_dereg_mr(cpu_gpu_tail_mr);

        // Ugly but needed
        gpu_cpu_ring_buff._mailbox = NULL;
        cpu_gpu_ring_buff._mailbox = NULL;

        // kill server

        /* step 1: send request to server using Send operation */

        /* RDMA send needs a gather element (local buffer)*/
        int ncqes;
        struct ibv_wc wc ={0}; /* CQE */

        printf("killing server!\n");
        int terminate = 1;
        // create memory regions
        struct ibv_mr * terminate_mr = ibv_reg_mr(pd, &terminate, sizeof(terminate), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ );
        if (!terminate_mr) {
            perror("ibv_reg_mr() failed for terminate_mr");
            exit(1);
        }

         wc.wr_id = 77;

         
        post_rdma_write(
                    (uint64_t)server_info.terminate_addr,                       // remote_dst
                    sizeof(int),     // len
                    server_info.terminate_mr.rkey,  // rkey
                    &terminate,                // local_src
                    terminate_mr->lkey,         // lkey
                    wc.wr_id, // wr_id
                    NULL);           

        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
        VERBS_WC_CHECK(wc);
        ////printf("wr-id =  %lu\n", wc.wr_id);
        terminate = 0;
        printf("terminate %d\n", terminate);

        post_rdma_read(
                &terminate, // local_src
                sizeof(int),  // len
                terminate_mr->lkey, // lkey
                (uint64_t)server_info.terminate_addr, // remote_dst
                server_info.terminate_mr.rkey,  // rkey
                wc.wr_id);      
        VERBS_WC_CHECK(wc);
        printf("terminate %d\n", terminate);

        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }

        ibv_dereg_mr(terminate_mr);

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
	    _images_out = images_out;
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
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
        if (requests_sent - send_cqes_received == OUTSTANDING_REQUESTS)
        {
            return false;
        }
	    ////printf("enq start, img: %d\n", img_id);

    
        /* TODO use RDMA Write and RDMA Read operations to enqueue the task on
         * a CPU-GPU producer consumer queue running on the server. */
        struct ibv_wc wc; /* CQE */
        int ncqes;

        wc.wr_id = 11;
        // Check if there is place in cpu-gpu queue
        post_rdma_read(
                        &cpu_gpu_ring_buff,           // local_src
                        sizeof(ring_buffer),  // len
                        cpu_gpu_ring_buff_mr->lkey, // lkey
                        (uint64_t)server_info.cpu_gpu_ring_buffer_addr,    // remote_dst
                        server_info.cpu_gpu_ring_buffer_mr.rkey,    // rkey
                        wc.wr_id);          // wr_id
        
        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
        ////printf("Got wc id %lu\n",wc.wr_id);

        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }

        // check for place
        if (cpu_gpu_ring_buff._tail - cpu_gpu_ring_buff._head == cpu_gpu_ring_buff.N) {
              ////printf("no place in cpu-gpu queue\n");
            return false;
        }
        ////printf("place in cpu-gpu queue\n");

        // Write Image to gpu buffers in sever
        wc.wr_id = 12;
        post_rdma_write(
                    (uint64_t)server_info.img_in_addr + (img_id  % OUTSTANDING_REQUESTS * IMG_SZ) ,                       // remote_dst
                     IMG_SZ,     // len
                    server_info.img_in_mr.rkey,                       // rkey
                     img_in,                // local_src
                    mr_images_in->lkey,                    // lkey
                    wc.wr_id, // wr_id
                    NULL);           
        ////printf("wrote image in cpu-gpu queue\n");

        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
                ////printf("Got wc id %lu\n",wc.wr_id);

        VERBS_WC_CHECK(wc);
        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }

        // Update cpu-gpu queue
        local_request.imgID  = img_id;
        local_request.imgIn  = (uchar*)(server_info.img_in_addr) + img_id % OUTSTANDING_REQUESTS * IMG_SZ;
        local_request.imgOut = (uchar*)(server_info.img_out_addr) + img_id % OUTSTANDING_REQUESTS * IMG_SZ;
        wc.wr_id = 13;
        

       	post_rdma_write(
                    (uint64_t)server_info.cpu_gpu_mail_box_addr + sizeof(request)*(cpu_gpu_ring_buff._tail % cpu_gpu_ring_buff.N),                       // remote_dst
                     sizeof(request),     // len
                    server_info.cpu_gpu_mail_box_mr.rkey,  // rkey
                    &local_request ,                // local_src
                    local_request_mr->lkey,                    // lkey
                    wc.wr_id, // wr_id
                    NULL);           

        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
        ////printf("Got wc id %lu\n",wc.wr_id);

        VERBS_WC_CHECK(wc);

        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }

        // Update head/tail
        cpu_gpu_tail= cpu_gpu_ring_buff._tail+1;
                wc.wr_id = 14;

        post_rdma_write(
                        (uint64_t)server_info.cpu_gpu_tail_addr,                       // remote_dst
                        sizeof(cpu_gpu_tail),     // len
                        server_info.cpu_gpu_tail_mr.rkey,  // rkey
                        &cpu_gpu_tail,                // local_src
                        cpu_gpu_tail_mr->lkey,                    // lkey
                        wc.wr_id, // wr_id
                        NULL);           


        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
                ////printf("Got wc id %lu\n",wc.wr_id);

         VERBS_WC_CHECK(wc);

        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }
   
	    //printf("enq end, img: %d\n",img_id);
        requests_sent++;
        return true;
    }


    virtual bool dequeue(int *img_id) override
    {
	    ////printf("dq start, img: ?\n");

        struct ibv_wc wc = {0}; /* CQE */
            int ncqes;
        wc.wr_id = 1;

        //flow for dqueue
        // rdma read gpucpu ring buffer
        ////printf("dequeu: sent first rdma read \n");
      	post_rdma_read(
                &gpu_cpu_ring_buff,           // local_src
                sizeof(ring_buffer),  // len
                gpu_cpu_ring_buff_mr->lkey, // lkey
                (uint64_t)server_info.gpu_cpu_ring_buffer_addr,    // remote_dst
                server_info.gpu_cpu_ring_buffer_mr.rkey,    // rkey
                wc.wr_id);          // wr_id

	    
	    while ((ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
        ////printf("Got wc id %lu\n",wc.wr_id);
        VERBS_WC_CHECK(wc);
        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }
        // check for work to dequee
        // //printf("gpu_cpu_ring_buff._tail - %d\n", gpu_cpu_ring_buff._tail.load());
        // //printf("gpu_cpu_ring_buff._head - %d\n", gpu_cpu_ring_buff._head.load());

        if (gpu_cpu_ring_buff._tail == gpu_cpu_ring_buff._head) {
	        // //printf("dequeu: gpu cpu empty\n");
            return false;
        }
        ////printf("dequeu: gpu cpu not empty\n");
	    //rdma read mail_box from tail/head? and get img id
        wc.wr_id = 2;
      	post_rdma_read(
                &local_request,           // local_src
                sizeof(local_request),  // len
                local_request_mr->lkey, // lkey
                (uint64_t)server_info.gpu_cpu_mail_box_addr + sizeof(local_request) * (gpu_cpu_ring_buff._head % gpu_cpu_ring_buff.N),    // remote_dst
                server_info.gpu_cpu_mail_box_mr.rkey,    // rkey
                wc.wr_id);          // wr_id

	    while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
        //printf("!!dequeue image1 %d!!\n", local_request.imgID);
        ////printf("Got wc id %lu\n",wc.wr_id);
         VERBS_WC_CHECK(wc);
        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }

         wc.wr_id = 3;

	    //rdma read img from cuda_host img_in[img_id] to our img in
      	post_rdma_read(
                _images_out + IMG_SZ * (local_request.imgID % N_IMAGES),           // local_src
                IMG_SZ,  // len
                mr_images_out->lkey, // lkey
                (uint64_t)local_request.imgOut,    // remote_dst
                server_info.img_out_mr.rkey,    // rkey
                wc.wr_id);          // wr_id
        //printf("!!dequeue image2 %d!!\n", local_request.imgID);

	    while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
            ////printf("Got wc id %lu\n",wc.wr_id);
         VERBS_WC_CHECK(wc);

        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }

	//rdma write cpugpu ring buffer with updated head-tail
        // Update head/tail
	gpu_cpu_head =  gpu_cpu_ring_buff._head + 1;
      	wc.wr_id = 4;
   	post_rdma_write(
                    (uint64_t)server_info.gpu_cpu_head_addr,                       // remote_dst
                    sizeof(gpu_cpu_head),     // len
                    server_info.gpu_cpu_head_mr.rkey,  // rkey
                    &gpu_cpu_head ,                // local_src
                    gpu_cpu_head_mr->lkey,                    // lkey
                    wc.wr_id, // wr_id
                    NULL);           

        while ((ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
          ////printf("Got wc id %lu\n",wc.wr_id);
         VERBS_WC_CHECK(wc);

        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }
            //printf("!!dequeue image3 %d!!\n", local_request.imgID);

	//printf("dq end, img: %d \n", local_request.imgID);
    send_cqes_received++;
    *img_id = local_request.imgID;
     return true;
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
        ////printf("Unknown mode.\n");
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
        ////printf("Unknown mode.\n");
        exit(1);
    }
}
