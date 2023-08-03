#include "synapse_device.h"

Synapse_device *create_synapses_device(cl_context ctx, int n_synapses, bool new_synapses_ids)
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

Synapse *synapse_device_to_host(cl_command_queue que, Synapse_device *syn_device)
{
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
    err = clEnqueueReadBuffer(que, syn_device->weight, CL_TRUE, 0, n_synapse * sizeof(cl_int), syn_host->weight, 0, NULL, NULL);
    OCL_CHECK(err, "read weight");

    err = clEnqueueReadBuffer(que, syn_device->gain, CL_TRUE, 0, n_synapse * sizeof(cl_int), syn_host->gain, 0, NULL, NULL);
    OCL_CHECK(err, "read gain");
    err = clEnqueueReadBuffer(que, syn_device->tau_syn, CL_TRUE, 0, n_synapse * sizeof(cl_int), syn_host->tau_syn, 0, NULL, NULL);
    OCL_CHECK(err, "read tau_syn");
    err = clEnqueueReadBuffer(que, syn_device->delay, CL_TRUE, 0, n_synapse * sizeof(cl_int), syn_host->delay, 0, NULL, NULL);
    OCL_CHECK(err, "read delay");

    return syn_host;
}