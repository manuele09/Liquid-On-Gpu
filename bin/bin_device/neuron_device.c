#include "neuron_device.h"

Layer_device *create_neurons_device(cl_kernel kernel, cl_command_queue que, cl_context ctx, int num_neurons, bool new_neurons_ids)
{
    cl_int err;
    static cl_int global_id = 0;
    static cl_int global_layer_id = 0;

    Layer_device *neurons = (Layer_device *)calloc(1, sizeof(Layer_device));

    neurons->n_neurons = num_neurons;
    neurons->layer_id = global_layer_id++;
    neurons->step = 0;

    neurons->id = clCreateBuffer(ctx, CL_MEM_READ_WRITE, num_neurons * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "Id Allocation");
    neurons->neuron_layer_id = clCreateBuffer(ctx, CL_MEM_READ_WRITE, num_neurons * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "Layer_id Allocation");

    int start_neuron_id = -1;
    int layer_id = -1;
    if (new_neurons_ids)
    {
        layer_id = neurons->layer_id;
        start_neuron_id = global_id;
        global_id += num_neurons;
    }

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
    cl_event event = initialize_neurons_device(kernel, que, neurons, 0, -1, -65.0f, -30.0f, 0.02f, 0.2f, -65.0f, 8.0f, start_neuron_id, layer_id);
    clFinish(que);
    Profile(event, "Init neurons");
    return neurons;
}

cl_event initialize_neurons_device(cl_kernel kernel, cl_command_queue que, Layer_device *neurons, cl_int start_idx, cl_int end_idx, cl_float init_v, cl_float init_u, cl_float init_a, cl_float init_b, cl_float init_c, cl_float init_d, cl_int start_neuron_id, cl_int layer_id)
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
    err = clSetKernelArg(kernel, index++, sizeof(cl_int), &start_idx);
    OCL_CHECK(err, "start_idx set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_int), &end_idx);
    OCL_CHECK(err, "end_idx set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_int), &start_neuron_id);
    OCL_CHECK(err, "start_neuron_id set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_int), &layer_id);
    OCL_CHECK(err, "layer_id set arg");

    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->id);
    OCL_CHECK(err, "id set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->neuron_layer_id);
    OCL_CHECK(err, "neuron_layer_id set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->last_spike);
    OCL_CHECK(err, "last_spike set arg");
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

Layer_device *combine_layers_device(cl_context ctx, cl_command_queue que, Layer_device **layers_device, int num_layers)
{
    Layer **layers_host = (Layer **)calloc(num_layers, sizeof(Layer *));
    for (int i = 0; i < num_layers; i++)
        layers_host[i] = neuron_device_to_host(que, layers_device[i]);

    Layer *combined_host = combine_layers(layers_host, num_layers);
    return neuron_host_to_device(ctx, que, combined_host);
}

void simulate_neurons_device(cl_command_queue que, cl_kernel kernel, Layer_device *neurons, int steps, cl_float dt)
{
    size_t global_size[] = {steps};

    cl_int err;
    int index = 0;
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->V);
    OCL_CHECK(err, "V set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->U);
    OCL_CHECK(err, "U set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->I);
    OCL_CHECK(err, "I set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->I_bias);
    OCL_CHECK(err, "I_bias set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->last_spike);
    OCL_CHECK(err, "last_spike set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->a);
    OCL_CHECK(err, "a set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->b);
    OCL_CHECK(err, "b set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->c);
    OCL_CHECK(err, "c set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_mem), &neurons->d);
    OCL_CHECK(err, "d set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_float), &dt);
    OCL_CHECK(err, "dt set arg");
    err = clSetKernelArg(kernel, index++, sizeof(cl_int), &neurons->step);
    OCL_CHECK(err, "step set arg");

    cl_event event;
    float tempo = 0;
    for (int i = 0; i < steps; i++)
    {
        err = clEnqueueNDRangeKernel(que, kernel, 1, NULL, global_size, NULL, 0, NULL, &event);
        clWaitForEvents(1, &event);
        tempo += runtime_ms(event);
        OCL_CHECK(err, "enqueue init_neurons");
        neurons->step++;
    }
    printf("Tempo di esecuzione: %f millisecondi\n", tempo);
}

Layer *neuron_device_to_host(cl_command_queue que, Layer_device *layer_device)
{
    Layer *layer_host = (Layer *)calloc(1, sizeof(Layer));

    layer_host->n_neurons = layer_device->n_neurons;
    layer_host->step = layer_device->step;
    layer_host->layer_id = layer_device->layer_id;

    int n_neurons = layer_host->n_neurons;
    layer_host->id = (int *)calloc(n_neurons, sizeof(int));
    layer_host->neuron_layer_id = (int *)calloc(n_neurons, sizeof(int));
    layer_host->last_spike = (int *)calloc(n_neurons, sizeof(int));
    layer_host->V = (float *)calloc(n_neurons, sizeof(float));
    layer_host->U = (float *)calloc(n_neurons, sizeof(float));
    layer_host->I = (float *)calloc(n_neurons, sizeof(float));
    layer_host->I_bias = (float *)calloc(n_neurons, sizeof(float));
    layer_host->a = (float *)calloc(n_neurons, sizeof(float));
    layer_host->b = (float *)calloc(n_neurons, sizeof(float));
    layer_host->c = (float *)calloc(n_neurons, sizeof(float));
    layer_host->d = (float *)calloc(n_neurons, sizeof(float));

    cl_int err;
    err = clEnqueueReadBuffer(que, layer_device->id, CL_TRUE, 0, n_neurons * sizeof(cl_int), layer_host->id, 0, NULL, NULL);
    OCL_CHECK(err, "read id");
    err = clEnqueueReadBuffer(que, layer_device->neuron_layer_id, CL_TRUE, 0, n_neurons * sizeof(cl_int), layer_host->neuron_layer_id, 0, NULL, NULL);
    OCL_CHECK(err, "read neuron_layer_id");
    err = clEnqueueReadBuffer(que, layer_device->last_spike, CL_TRUE, 0, n_neurons * sizeof(cl_int), layer_host->last_spike, 0, NULL, NULL);
    OCL_CHECK(err, "read last_spike");
    err = clEnqueueReadBuffer(que, layer_device->V, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->V, 0, NULL, NULL);
    OCL_CHECK(err, "read V");
    err = clEnqueueReadBuffer(que, layer_device->U, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->U, 0, NULL, NULL);
    OCL_CHECK(err, "read U");
    err = clEnqueueReadBuffer(que, layer_device->I, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->I, 0, NULL, NULL);
    OCL_CHECK(err, "read I");
    err = clEnqueueReadBuffer(que, layer_device->I_bias, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->I_bias, 0, NULL, NULL);
    OCL_CHECK(err, "read I_bias");
    err = clEnqueueReadBuffer(que, layer_device->a, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->a, 0, NULL, NULL);
    OCL_CHECK(err, "read a");
    err = clEnqueueReadBuffer(que, layer_device->b, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->b, 0, NULL, NULL);
    OCL_CHECK(err, "read b");
    err = clEnqueueReadBuffer(que, layer_device->c, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->c, 0, NULL, NULL);
    OCL_CHECK(err, "read c");
    err = clEnqueueReadBuffer(que, layer_device->d, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->d, 0, NULL, NULL);
    OCL_CHECK(err, "read d");

    return layer_host;
}

Layer_device *neuron_host_to_device(cl_context ctx, cl_command_queue que, Layer *layer_host)
{
    cl_int err;

    Layer_device *layer_device = (Layer_device *)calloc(1, sizeof(Layer_device));

    layer_device->n_neurons = layer_host->n_neurons;
    layer_device->step = layer_host->step;
    layer_device->layer_id = layer_host->layer_id;

    int n_neurons = layer_device->n_neurons;
    layer_device->id = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_neurons * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "Id Allocation");
    layer_device->neuron_layer_id = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_neurons * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "Layer_id Allocation");
    layer_device->last_spike = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_neurons * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "Spike Allocation");
    layer_device->V = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_neurons * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "V Allocation");
    layer_device->U = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_neurons * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "U Allocation");
    layer_device->I = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_neurons * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "I Allocation");
    layer_device->I_bias = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_neurons * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "I_bias Allocation");
    layer_device->a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_neurons * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "a Allocation");
    layer_device->b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_neurons * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "b Allocation");
    layer_device->c = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_neurons * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "c Allocation");
    layer_device->d = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_neurons * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "d Allocation");

    err = clEnqueueWriteBuffer(que, layer_device->id, CL_TRUE, 0, n_neurons * sizeof(cl_int), layer_host->id, 0, NULL, NULL);
    OCL_CHECK(err, "write id");
    err = clEnqueueWriteBuffer(que, layer_device->neuron_layer_id, CL_TRUE, 0, n_neurons * sizeof(cl_int), layer_host->neuron_layer_id, 0, NULL, NULL);
    OCL_CHECK(err, "write neuron_layer_id");
    err = clEnqueueWriteBuffer(que, layer_device->last_spike, CL_TRUE, 0, n_neurons * sizeof(cl_int), layer_host->last_spike, 0, NULL, NULL);
    OCL_CHECK(err, "write last_spike");
    err = clEnqueueWriteBuffer(que, layer_device->V, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->V, 0, NULL, NULL);
    OCL_CHECK(err, "write V");
    err = clEnqueueWriteBuffer(que, layer_device->U, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->U, 0, NULL, NULL);
    OCL_CHECK(err, "write U");
    err = clEnqueueWriteBuffer(que, layer_device->I, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->I, 0, NULL, NULL);
    OCL_CHECK(err, "write I");
    err = clEnqueueWriteBuffer(que, layer_device->I_bias, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->I_bias, 0, NULL, NULL);
    OCL_CHECK(err, "write I_bias");
    err = clEnqueueWriteBuffer(que, layer_device->a, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->a, 0, NULL, NULL);
    OCL_CHECK(err, "write a");
    err = clEnqueueWriteBuffer(que, layer_device->b, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->b, 0, NULL, NULL);
    OCL_CHECK(err, "write b");
    err = clEnqueueWriteBuffer(que, layer_device->c, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->c, 0, NULL, NULL);
    OCL_CHECK(err, "write c");
    err = clEnqueueWriteBuffer(que, layer_device->d, CL_TRUE, 0, n_neurons * sizeof(cl_float), layer_host->d, 0, NULL, NULL);
    OCL_CHECK(err, "write d");

    return layer_device;
}

void free_neurons_device(Layer_device *neurons)
{
    cl_int err;
    // mettere i check
    err = clReleaseMemObject(neurons->id);
    OCL_CHECK(err, "release id");
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