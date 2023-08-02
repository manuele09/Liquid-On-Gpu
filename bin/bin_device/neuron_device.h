#ifndef NEURON_H
#define NEURON_H 1

#include "ocl_boiler.h"
#include "../bin_host/neuron.h"
#include <stdbool.h>
struct Layer_host
{
    int *id;              // unique id for each neuron
    int *neuron_layer_id; // the layer id of provenience of the neuron, can be different from the layer id of the layer.
                          // The layer of provenience may not exists anymore. Storing the layer id of provenience
                          // is useful to keep track of the layer composition of the final (combined) layer.
    int *last_spike;      // timestamp of last spike emitted

    float *V;      // membrane potential
    float *U;      // recovery variable
    float *I;      // input current
    float *I_bias; // bias current
    float *a;      // parameter
    float *b;      // parameter
    float *c;      // parameter
    float *d;      // parameter

    int n_neurons; // number of neurons
    int step;      // current simulation step
    int layer_id;  // unique id for each layer
};
typedef struct Layer_host Layer_host;

struct Layer_device
{
    cl_mem id;              // (int) unique id for each neuron
    cl_mem neuron_layer_id; // (int) the layer id of provenience of the neuron, can be different from the layer id of the layer.
                            // The layer of provenience may not exists anymore. Storing the layer id of provenience
                            // is useful to keep track of the layer composition of the final (combined) layer.
    cl_mem last_spike;      // (int) timestamp of last spike emitted

    cl_mem V;      // (float) membrane potential
    cl_mem U;      // (float) recovery variable
    cl_mem I;      // (float) input current
    cl_mem I_bias; // (float) bias current
    cl_mem a;      // (float) parameter
    cl_mem b;      // (float) parameter
    cl_mem c;      // (float) parameter
    cl_mem d;      // (float) parameter

    cl_int n_neurons; // number of neurons
    cl_int step;      // current simulation step
    cl_int layer_id;  // unique id for each layer
};
typedef struct Layer_device Layer_device;

/**
 * @brief Create a Layer (unique id) containing num_neurons neurons.
 * Each neuron is identified by a unique id.
 * They are initialized with default values.
 *
 * @param ctx OpenCL context
 * @param num_neurons Number of neurons to create
 * @param new_neurons_ids If true doesn't assigne new indexes to neurons. Usefule when creating a layer
 * to merge two existing.
 * @return Layer_device
 */
Layer_device *create_neurons_device(cl_kernel kernel, cl_command_queue que, cl_context ctx, int num_neurons, bool new_neurons_ids);

cl_event initialize_neurons_device(cl_kernel kernel, cl_command_queue que, Layer_device *neurons, cl_int start_idx, cl_int end_idx, cl_float init_v, cl_float init_u, cl_float init_a, cl_float init_b, cl_float init_c, cl_float init_d, cl_int start_neuron_id, cl_int layer_id);

Layer_device *combine_layers_device(cl_context ctx, cl_command_queue que, Layer_device **layers_device, int num_layers);

Layer *neuron_device_to_host(cl_command_queue que, Layer_device *layer_device);

Layer_device *neuron_host_to_device(cl_context ctx, cl_command_queue que, Layer *layer_host);

/**
 * @brief Free the memory allocated for the neurons.
 *
 * @param neurons
 */
void free_neurons_device(Layer_device *neurons);

#endif