#include "lib.cuh"

void cpu_to_gpu(gpu_ptc_t *gptc_data,gpu_tmp_t *gtmp_data,cpu_input_t *ginput_dat,cpu_param_t *cparam,gpu_socket_t **gsocket)
{
    for(int i=0;i<cparam->gpu_num;i++)
    {
        cudaSetDevice(i);
        int elm_num = gsocket[i]->total_ptc_num;
        cudaMemcpy(gptc_data[i].pos,ginput_dat[i].pos,sizeof(float3)*elm_num,cudaMemcpyHostToDevice);
        cudaMemcpy(gptc_data[i].vel,ginput_dat[i].vel,sizeof(float3)*elm_num,cudaMemcpyHostToDevice);
        cudaMemcpy(gptc_data[i].rhop,ginput_dat[i].rhop,sizeof(float2)*elm_num,cudaMemcpyHostToDevice);
        cudaMemcpy(gptc_data[i].type,ginput_dat[i].type,sizeof(int)*elm_num,cudaMemcpyHostToDevice);
        cudaMemcpy(gptc_data[i].table,ginput_dat[i].table,sizeof(int)*elm_num,cudaMemcpyHostToDevice);
        cudaMemcpy(gptc_data[i].is_ptc,ginput_dat[i].is_ptc,sizeof(int)*elm_num,cudaMemcpyHostToDevice);
        cudaMemcpy(gptc_data[i].gpu_id,ginput_dat[i].gpu_id,sizeof(int)*elm_num,cudaMemcpyHostToDevice);

        cudaMemcpy(gtmp_data[i].acc_drhodt,ginput_dat[i].acc_drhodt,sizeof(float4)*elm_num,cudaMemcpyHostToDevice);    
        check_gerr(__FILE__,__LINE__);    
    }
}

void gpu_to_cpu(cpu_output_t *coutput_dat,gpu_ptc_t *gptc_dat,gpu_tmp_t *gtmp_dat,cpu_param_t *cparam,gpu_socket_t **gsocket)
{
    for(int i=0;i<cparam->gpu_num;i++)
    {
        cudaSetDevice(i);
        cudaMemcpy(coutput_dat[i].pos,gptc_dat[i].pos,sizeof(float)*3*gsocket[i]->total_ptc_num,cudaMemcpyDeviceToHost);

        #ifdef ZSPH_OUTPUT_VEL
        cudaMemcpy(coutput_dat[i].vel,gptc_dat[i].vel,sizeof(float)*3*gsocket[i]->total_ptc_num,cudaMemcpyDeviceToHost);
        #endif

        #if (defined ZSPH_OUTPUT_RHO) || (defined ZSPH_OUTPUT_P)
        cudaMemcpy(coutput_dat[i].rhop,gptc_dat[i].rhop,sizeof(float)*2*gsocket[i]->total_ptc_num,cudaMemcpyDeviceToHost);
        #endif

        #ifdef ZSPH_OUTPUT_TYPE
        cudaMemcpy(coutput_dat[i].type,gptc_dat[i].type,sizeof(int)*gsocket[i]->total_ptc_num,cudaMemcpyDeviceToHost);
        #endif

        #ifdef ZSPH_OUTPUT_TABLE
        cudaMemcpy(coutput_dat[i].table,gptc_dat[i].table,sizeof(int)*gsocket[i]->total_ptc_num,cudaMemcpyDeviceToHost);
        #endif

        #ifdef ZSPH_OUTPUT_ISPTC
        cudaMemcpy(coutput_dat[i].is_ptc,gptc_dat[i].is_ptc,sizeof(int)*gsocket[i]->total_ptc_num,cudaMemcpyDeviceToHost);
        #endif

        #ifdef ZSPH_OUTPUT_GPUID
        cudaMemcpy(coutput_dat[i].gpu_id,gptc_dat[i].gpu_id,sizeof(int)*gsocket[i]->total_ptc_num,cudaMemcpyDeviceToHost);
        #endif

        #if (defined ZSPH_OUTPUT_ACC) || (defined ZSPH_OUTPUT_DRHODT)
        cudaMemcpy(coutput_dat[i].acc_drhodt,gtmp_dat[i].acc_drhodt,sizeof(float)*4*gsocket[i]->total_ptc_num,cudaMemcpyDeviceToHost);
        #endif

        #ifdef ZSPH_OUTPUT_HASH
        cudaMemcpy(coutput_dat[i].hash,gtmp_dat[i].hash,sizeof(int)*gsocket[i]->total_ptc_num,cudaMemcpyDeviceToHost);
        #endif

        #ifdef ZSPH_OUTPUT_WSUM
        cudaMemcpy(coutput_dat[i].wsum,gtmp_dat[i].wsum,sizeof(float)*gsocket[i]->total_ptc_num,cudaMemcpyDeviceToHost);
        #endif
        check_gerr(__FILE__,__LINE__);
        coutput_dat[i].ptc_num = gsocket[i]->total_ptc_num;
    }
    for(int i=0;i<cparam->gpu_num;i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        check_gerr(__FILE__,__LINE__);
    }
}

void init_p2p(cpu_param_t *cparam)
{
    for(int i=0;i<cparam->gpu_num;i++)
    {
        cudaSetDevice(i);
        if(i == 0)
        {
            cudaDeviceEnablePeerAccess(i+1,0);
        }
        else if(i == cparam->gpu_num -1 )
        {
            cudaDeviceEnablePeerAccess(i-1,0);
        }
        else
        {
            cudaDeviceEnablePeerAccess(i+1,0);
            cudaDeviceEnablePeerAccess(i-1,0);
        }
        check_gerr(__FILE__,__LINE__);
    }
}

void peer_to_peer(gpu_socket_t **gsocket,cpu_param_t *cparam)
{
    for(int i=0;i<cparam->gpu_num;i++)
    {
        cudaSetDevice(i);
        gsocket[i]->send_ptc_num = gsocket[i]->total_ptc_num - gsocket[i]->send_ptc_num;
        assert(gsocket[i]->send_ptc_num < (cparam->gmax_size-gsocket[i]->total_ptc_num/2) && gsocket[i]->send_ptc_num > 0 && gsocket[i]->total_ptc_num > 0);
        //std::cout << gsocket[i]->send_ptc_num << " " << gsocket[i]->total_ptc_num << std::endl;
        if(i == 0)
        {
            //data to right
            cudaMemcpyPeer(gsocket[i+1]->pos_rec_l,i+1,gsocket[i]->pos_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i+1]->vel_rec_l,i+1,gsocket[i]->vel_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i+1]->rhop_rec_l,i+1,gsocket[i]->rhop_send,i,sizeof(float2)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);

            cudaMemcpyPeer(gsocket[i+1]->tmp_pos_rec_l,i+1,gsocket[i]->tmp_pos_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i+1]->tmp_vel_rec_l,i+1,gsocket[i]->tmp_vel_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i+1]->tmp_rhop_rec_l,i+1,gsocket[i]->tmp_rhop_send,i,sizeof(float2)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);

            cudaMemcpyPeer(gsocket[i+1]->type_rec_l,i+1,gsocket[i]->type_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i+1]->table_rec_l,i+1,gsocket[i]->table_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i+1]->is_ptc_rec_l,i+1,gsocket[i]->is_ptc_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i+1]->gpu_id_rec_l,i+1,gsocket[i]->gpu_id_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
        }
        else if (i == cparam->gpu_num -1)
        {
            //data to left
            cudaMemcpyPeer(gsocket[i-1]->pos_rec_r,i-1,gsocket[i]->pos_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i-1]->vel_rec_r,i-1,gsocket[i]->vel_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i-1]->rhop_rec_r,i-1,gsocket[i]->rhop_send,i,sizeof(float2)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            
            cudaMemcpyPeer(gsocket[i-1]->tmp_pos_rec_r,i-1,gsocket[i]->tmp_pos_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i-1]->tmp_vel_rec_r,i-1,gsocket[i]->tmp_vel_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i-1]->tmp_rhop_rec_r,i-1,gsocket[i]->tmp_rhop_send,i,sizeof(float2)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);

            cudaMemcpyPeer(gsocket[i-1]->type_rec_r,i-1,gsocket[i]->type_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i-1]->table_rec_r,i-1,gsocket[i]->table_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i-1]->is_ptc_rec_r,i-1,gsocket[i]->is_ptc_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i-1]->gpu_id_rec_r,i-1,gsocket[i]->gpu_id_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
        }
        else
        {
            //data to left
            //std::cout << std::hex << gsocket[i-1]->pos_rec_r<<" " <<gsocket[i]->pos_send << std::endl;
            cudaMemcpyPeer(gsocket[i-1]->pos_rec_r,i-1,gsocket[i]->pos_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i-1]->vel_rec_r,i-1,gsocket[i]->vel_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i-1]->rhop_rec_r,i-1,gsocket[i]->rhop_send,i,sizeof(float2)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            
            cudaMemcpyPeer(gsocket[i-1]->tmp_pos_rec_r,i-1,gsocket[i]->tmp_pos_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i-1]->tmp_vel_rec_r,i-1,gsocket[i]->tmp_vel_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i-1]->tmp_rhop_rec_r,i-1,gsocket[i]->tmp_rhop_send,i,sizeof(float2)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);

            cudaMemcpyPeer(gsocket[i-1]->type_rec_r,i-1,gsocket[i]->type_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i-1]->table_rec_r,i-1,gsocket[i]->table_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i-1]->is_ptc_rec_r,i-1,gsocket[i]->is_ptc_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i-1]->gpu_id_rec_r,i-1,gsocket[i]->gpu_id_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);

            //data to right
            cudaMemcpyPeer(gsocket[i+1]->pos_rec_l,i+1,gsocket[i]->pos_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i+1]->vel_rec_l,i+1,gsocket[i]->vel_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i+1]->rhop_rec_l,i+1,gsocket[i]->rhop_send,i,sizeof(float2)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);

            cudaMemcpyPeer(gsocket[i+1]->tmp_pos_rec_l,i+1,gsocket[i]->tmp_pos_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i+1]->tmp_vel_rec_l,i+1,gsocket[i]->tmp_vel_send,i,sizeof(float3)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i+1]->tmp_rhop_rec_l,i+1,gsocket[i]->tmp_rhop_send,i,sizeof(float2)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);

            cudaMemcpyPeer(gsocket[i+1]->type_rec_l,i+1,gsocket[i]->type_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i+1]->table_rec_l,i+1,gsocket[i]->table_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i+1]->is_ptc_rec_l,i+1,gsocket[i]->is_ptc_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
            cudaMemcpyPeer(gsocket[i+1]->gpu_id_rec_l,i+1,gsocket[i]->gpu_id_send,i,sizeof(int)*gsocket[i]->send_ptc_num);
            check_gerr(__FILE__,__LINE__);
        }
    }
    for(int i=0;i<cparam->gpu_num;i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        check_gerr(__FILE__,__LINE__);
    }
    int tmp_ptc_num = 0;
    for(int i=0;i<cparam->gpu_num;i++)
    {
        tmp_ptc_num += gsocket[i]->total_ptc_num;
        //gsocket[i]->total_ptc_num = 0;
        //gsocket[i]->send_ptc_num = 0;
    }
    assert(tmp_ptc_num <= cparam->ptc_num);
}