#include "synapse_device.h"

/* Synapse_device *create_synapses_device(cl_context ctx, int n_synapses, bool new_synapses_ids)
{
    cl_int err;
    static int synapse_global_id = 0;

    Synapse_device *synaptic = (Synapse_device *)calloc(1, sizeof(Synapse_device));

    synaptic->synapse_family_id = synapse_global_id++;

    synaptic->synapse_id = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapses * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "synapse_id Allocation");
    synaptic->pre_neuron_idx = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapses * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "pre_neuron_idx Allocation");
    synaptic->post_neuron_idx = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapses * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "post_neuron_idx Allocation");

    // sto ignorando synapses_id

    synaptic->layer = NULL;

    synaptic->pre_location = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapses * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "pre_location Allocation");
    synaptic->post_location = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapses * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "post_location Allocation");

    synaptic->weight = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapses * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "weight Allocation");

    synaptic->gain = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapses * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "gain Allocation");

    synaptic->tau_syn = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapses * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "tau_syn Allocation");

    synaptic->delay = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapses * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "delay Allocation");

    synaptic->n_synapses = n_synapses;

    return synaptic;
}
 */

Synapse_device *create_synapses_device(cl_context ctx, cl_command_queue que, int n_synapses, bool new_synapses_ids)
{
    Synapse *syn_host = create_synapses(n_synapses, new_synapses_ids);
    Synapse_device *syn_dev = synapse_host_to_device(ctx, que, syn_host);
    free(syn_host);
    return syn_dev;
}

Synapse *synapse_device_to_host(cl_command_queue que, Synapse_device *syn_device)
{
    if (syn_device == NULL)
        return NULL;
    Synapse *syn_host = (Synapse *)calloc(1, sizeof(Synapse));
    int n_synapse = syn_device->n_synapses;
    syn_host->synapse_family_id = syn_device->synapse_family_id;
    syn_host->n_synapses = n_synapse;

    syn_host->synapse_id = (int *)calloc(n_synapse, sizeof(int));
    syn_host->pre_neuron_idx = (int *)calloc(n_synapse, sizeof(int));
    syn_host->post_neuron_idx = (int *)calloc(n_synapse, sizeof(int));

    syn_host->pre_location = (int *)calloc(n_synapse, sizeof(int));
    syn_host->post_location = (int *)calloc(n_synapse, sizeof(int));
    syn_host->weight = (float *)calloc(n_synapse, sizeof(float));

    syn_host->gain = (float *)calloc(n_synapse, sizeof(float));
    syn_host->tau_syn = (float *)calloc(n_synapse, sizeof(float));
    syn_host->delay = (float *)calloc(n_synapse, sizeof(float));

    cl_int err;
    err = clEnqueueReadBuffer(que, syn_device->synapse_id, CL_TRUE, 0, n_synapse * sizeof(cl_int), syn_host->synapse_id, 0, NULL, NULL);
    OCL_CHECK(err, "read synapse_id");
    err = clEnqueueReadBuffer(que, syn_device->pre_neuron_idx, CL_TRUE, 0, n_synapse * sizeof(cl_int), syn_host->pre_neuron_idx, 0, NULL, NULL);
    OCL_CHECK(err, "read pre_neuron_idx");
    err = clEnqueueReadBuffer(que, syn_device->post_neuron_idx, CL_TRUE, 0, n_synapse * sizeof(cl_int), syn_host->post_neuron_idx, 0, NULL, NULL);
    OCL_CHECK(err, "read post_neuron_idx");

    err = clEnqueueReadBuffer(que, syn_device->pre_location, CL_TRUE, 0, n_synapse * sizeof(cl_int), syn_host->pre_location, 0, NULL, NULL);
    OCL_CHECK(err, "read pre_location");
    err = clEnqueueReadBuffer(que, syn_device->post_location, CL_TRUE, 0, n_synapse * sizeof(cl_int), syn_host->post_location, 0, NULL, NULL);
    OCL_CHECK(err, "read post_location");
    err = clEnqueueReadBuffer(que, syn_device->weight, CL_TRUE, 0, n_synapse * sizeof(cl_float), syn_host->weight, 0, NULL, NULL);
    OCL_CHECK(err, "read weight");

    err = clEnqueueReadBuffer(que, syn_device->gain, CL_TRUE, 0, n_synapse * sizeof(cl_float), syn_host->gain, 0, NULL, NULL);
    OCL_CHECK(err, "read gain");
    err = clEnqueueReadBuffer(que, syn_device->tau_syn, CL_TRUE, 0, n_synapse * sizeof(cl_float), syn_host->tau_syn, 0, NULL, NULL);
    OCL_CHECK(err, "read tau_syn");
    err = clEnqueueReadBuffer(que, syn_device->delay, CL_TRUE, 0, n_synapse * sizeof(cl_float), syn_host->delay, 0, NULL, NULL);
    OCL_CHECK(err, "read delay");

    syn_host->layer = neuron_device_to_host(que, syn_device->layer);

    return syn_host;
}

Synapse_device *synapse_host_to_device(cl_context ctx, cl_command_queue que, Synapse *syn_host)
{
    if (syn_host == NULL)
        return NULL;
    cl_int err;
    Synapse_device *syn_device = (Synapse_device *)calloc(1, sizeof(Synapse_device));

    int n_synapse = syn_host->n_synapses;

    syn_device->synapse_family_id = syn_host->synapse_family_id;
    syn_device->n_synapses = n_synapse;

    syn_device->synapse_id = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapse * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "synapse_id Allocation");
    syn_device->pre_neuron_idx = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapse * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "pre_neuron_idx Allocation");
    syn_device->post_neuron_idx = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapse * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "post_neuron_idx Allocation");
    syn_device->pre_location = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapse * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "pre_location Allocation");
    syn_device->post_location = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapse * sizeof(cl_int), NULL, &err);
    OCL_CHECK(err, "post_location Allocation");
    syn_device->weight = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapse * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "weight Allocation");
    syn_device->gain = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapse * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "gain Allocation");
    syn_device->tau_syn = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapse * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "tau_syn Allocation");
    syn_device->delay = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_synapse * sizeof(cl_float), NULL, &err);
    OCL_CHECK(err, "delay Allocation");

    err = clEnqueueWriteBuffer(que, syn_device->synapse_id, CL_TRUE, 0, n_synapse * sizeof(cl_int), syn_host->synapse_id, 0, NULL, NULL);
    OCL_CHECK(err, "write synapse_id");
    err = clEnqueueWriteBuffer(que, syn_device->pre_neuron_idx, CL_TRUE, 0, n_synapse * sizeof(cl_int), syn_host->pre_neuron_idx, 0, NULL, NULL);
    OCL_CHECK(err, "write pre_neuron_idx");
    err = clEnqueueWriteBuffer(que, syn_device->post_neuron_idx, CL_TRUE, 0, n_synapse * sizeof(cl_int), syn_host->post_neuron_idx, 0, NULL, NULL);
    OCL_CHECK(err, "write post_neuron_idx");
    err = clEnqueueWriteBuffer(que, syn_device->pre_location, CL_TRUE, 0, n_synapse * sizeof(cl_int), syn_host->pre_location, 0, NULL, NULL);
    OCL_CHECK(err, "write pre_location");
    err = clEnqueueWriteBuffer(que, syn_device->post_location, CL_TRUE, 0, n_synapse * sizeof(cl_int), syn_host->post_location, 0, NULL, NULL);
    OCL_CHECK(err, "write post_location");
    err = clEnqueueWriteBuffer(que, syn_device->weight, CL_TRUE, 0, n_synapse * sizeof(cl_float), syn_host->weight, 0, NULL, NULL);
    OCL_CHECK(err, "write weight");
    err = clEnqueueWriteBuffer(que, syn_device->gain, CL_TRUE, 0, n_synapse * sizeof(cl_float), syn_host->gain, 0, NULL, NULL);
    OCL_CHECK(err, "write gain");
    err = clEnqueueWriteBuffer(que, syn_device->tau_syn, CL_TRUE, 0, n_synapse * sizeof(cl_float), syn_host->tau_syn, 0, NULL, NULL);
    OCL_CHECK(err, "write tau_syn");
    err = clEnqueueWriteBuffer(que, syn_device->delay, CL_TRUE, 0, n_synapse * sizeof(cl_float), syn_host->delay, 0, NULL, NULL);
    OCL_CHECK(err, "write delay");

    syn_device->layer = neuron_host_to_device(ctx, que, syn_host->layer);

    return syn_device;
}

Synapse_device *connect_device(cl_context ctx, cl_command_queue que, Layer_device *pre_layer_device, Layer_device *post_layer_device, int *conn_matrix)
{
    Layer *pre_layer_host = neuron_device_to_host(que, pre_layer_device);
    Layer *post_layer_host = neuron_device_to_host(que, post_layer_device);

    Synapse *synapses_host = connect(pre_layer_host, post_layer_host, conn_matrix);

    Synapse_device *syn_device = synapse_host_to_device(ctx, que, synapses_host);

    return syn_device;
}

void free_synapses_device(Synapse_device *syn)
{
    cl_int err;
    // // mettere i check
    cl_mem pre_neuron_idx;  // (int*) unique id of the presynaptic neuron
    cl_mem post_neuron_idx; // (int*) unique id of the postsynaptic neuron
    cl_mem pre_location;    // (int*) Location of presynaptic neuron in the presynaptic layer (different from the id)
    cl_mem post_location;   // (int*) Location of presynaptic neuron in the presynaptic layer (different from the id)
    cl_mem weight;          //(float*)
    cl_mem gain;            //(float*)
    cl_mem tau_syn;         //(float*) questi parametri dovrebbero essere espressi in ms
    cl_mem delay;           //(float*)

    err = clReleaseMemObject(syn->synapse_id);
    OCL_CHECK(err, "release synapse_id");

    err = clReleaseMemObject(syn->pre_neuron_idx);
    OCL_CHECK(err, "release pre_neuron_idx");
    err = clReleaseMemObject(syn->post_neuron_idx);
    OCL_CHECK(err, "release post_neuron_idx");
    err = clReleaseMemObject(syn->pre_location);
    OCL_CHECK(err, "release pre_location");
    err = clReleaseMemObject(syn->post_location);
    OCL_CHECK(err, "release post_location");
    err = clReleaseMemObject(syn->weight);
    OCL_CHECK(err, "release weight");
    err = clReleaseMemObject(syn->gain);
    OCL_CHECK(err, "release gain");
    err = clReleaseMemObject(syn->tau_syn);
    OCL_CHECK(err, "release tau_syn");
    err = clReleaseMemObject(syn->delay);
    OCL_CHECK(err, "release delay");

    free(syn);
}
