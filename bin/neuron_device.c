#include "neuron_device.h"




Layer_device *create_neurons_device(cl_kernel kernel, cl_command_queue que, cl_context ctx, int num_neurons, bool new_neurons_ids)
{
    cl_int err;
    // static cl_int global_id = 0;
    static cl_int global_layer_id = 0;

    Layer_device *neurons = (Layer_device *)calloc(1, sizeof(Layer_device));

    neurons->n_neurons = num_neurons;
    neurons->layer_id = global_layer_id++;
    neurons->step = 0;

    neurons->id = clCreateBuffer(ctx, CL_MEM_READ_WRITE, num_neurons * sizeof(cl_int), NULL, &err);
	OCL_CHECK(err, "Id Allocation");
    neurons->neuron_layer_id = clCreateBuffer(ctx, CL_MEM_READ_WRITE, num_neurons * sizeof(cl_int), NULL, &err);
	OCL_CHECK(err, "Layer_id Allocation");

    // if (new_neurons_ids)
    //     for (int i = 0; i < num_neurons; i++)
    //     {
    //         neurons->id[i] = global_id++;
    //         neurons->neuron_layer_id[i] = neurons->layer_id;
    //     }

    neurons->last_spike = clCreateBuffer(ctx, CL_MEM_READ_WRITE, num_neurons * sizeof(cl_int), NULL, &err);
	OCL_CHECK(err, "Id Allocation");


    neurons->V = clCreateBuffer(ctx, CL_MEM_READ_WRITE, num_neurons * sizeof(cl_float), NULL, &err);
	OCL_CHECK(err, "V Allocation");
    neurons->U = clCreateBuffer(ctx, CL_MEM_READ_WRITE, num_neurons * sizeof(cl_float), NULL, &err);
	OCL_CHECK(err, "U Allocation");
    neurons->I = clCreateBuffer(ctx, CL_MEM_READ_WRITE, num_neurons * sizeof(cl_float), NULL, &err);
	OCL_CHECK(err, "I Allocation");
    neurons->I_bias = clCreateBuffer(ctx, CL_MEM_READ_WRITE, num_neurons * sizeof(cl_float), NULL, &err);
	OCL_CHECK(err, "I_bias Allocation");
    neurons->a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, num_neurons * sizeof(cl_float), NULL, &err);
	OCL_CHECK(err, "a Allocation");
    neurons->b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, num_neurons * sizeof(cl_float), NULL, &err);
	OCL_CHECK(err, "b Allocation");
    neurons->c = clCreateBuffer(ctx, CL_MEM_READ_WRITE, num_neurons * sizeof(cl_float), NULL, &err);
	OCL_CHECK(err, "c Allocation");
    neurons->d = clCreateBuffer(ctx, CL_MEM_READ_WRITE, num_neurons * sizeof(cl_float), NULL, &err);
	OCL_CHECK(err, "d Allocation");
    printf("Finito\n");
    // initialize_neurons(neurons, 0, -1, -65.0f, -30.0f, 0.02f, 0.2f, -65.0f, 8.0f);
    cl_event event = initialize_neurons_device(kernel, que, neurons, 0, -1, -65.0f, -30.0f, 0.02f, 0.2f, -65.0f, 8.0f);
    clFinish(que);
    Profile(event, "Init neurons");
    return neurons;
}

cl_event initialize_neurons_device(cl_kernel kernel, cl_command_queue que, Layer_device *neurons, int start_idx, int end_idx, float init_v, float init_u, float init_a, float init_b, float init_c, float init_d)
{
    
    if (end_idx == -1 || end_idx >= neurons->n_neurons)
        end_idx = neurons->n_neurons - 1;

    if (start_idx < 0 || start_idx > end_idx)
    {
        printf("Invalid start or end index: start=%d, end=%d.\n", start_idx, end_idx);
        exit(1);
    }
    int items_to_init = end_idx - start_idx + 1;

    size_t global_size[] = {items_to_init};
    size_t global_offset[] = {start_idx};
    
    cl_event event;
    


    cl_int err;
    int index = 0;
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->V);
    OCL_CHECK(err, "V set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->U);
    OCL_CHECK(err, "U set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->a);
    OCL_CHECK(err, "a set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->b);
    OCL_CHECK(err, "b set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->c);
    OCL_CHECK(err, "c set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->d);
    OCL_CHECK(err, "d set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_int), &start_idx);
    OCL_CHECK(err, "start_idx set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_int), &end_idx);
    OCL_CHECK(err, "end_idx set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_float), &init_v);
    OCL_CHECK(err, "init_v set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_float), &init_u);
    OCL_CHECK(err, "init_u set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_float), &init_a);
    OCL_CHECK(err, "init_a set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_float), &init_b);
    OCL_CHECK(err, "init_b set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_float), &init_c);
    OCL_CHECK(err, "init_c set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_float), &init_d);
    OCL_CHECK(err, "init_d set arg");

    err = clEnqueueNDRangeKernel(que, kernel, 1, global_offset, global_size, NULL, 0, NULL, &event);
    OCL_CHECK(err, "enqueue init_neurons");

    return event;
}


void free_neurons(Layer_device *neurons)
{
    cl_int err;
    //mettere i check
    err = clReleaseMemObject(neurons->id);
    clReleaseMemObject(neurons->neuron_layer_id);
    clReleaseMemObject(neurons->last_spike);

    clReleaseMemObject(neurons->V);
    clReleaseMemObject(neurons->U);
    clReleaseMemObject(neurons->I);
    clReleaseMemObject(neurons->I_bias);
    clReleaseMemObject(neurons->a);
    clReleaseMemObject(neurons->b);
    clReleaseMemObject(neurons->c);
    clReleaseMemObject(neurons->d);

    free(neurons);
}