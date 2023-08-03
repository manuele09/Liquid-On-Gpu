#ifndef SYNAPESE_DEVICE_H
#define SYNAPESE_DEVICE_H

#include "ocl_boiler.h"
#include "../bin_host/synapse.h"
#include "neuron_device.h"
#include <stdbool.h>

struct Synapse_device
{
    cl_mem synapse_id; // (int*) id of provenance.

    cl_mem pre_neuron_idx;  // (int*) unique id of the presynaptic neuron
    cl_mem post_neuron_idx; // (int*) unique id of the postsynaptic neuron

    Layer_device *layer; // The layer address

    cl_mem pre_location;  // (int*) Location of presynaptic neuron in the presynaptic layer (different from the id)
    cl_mem post_location; // (int*) Location of presynaptic neuron in the presynaptic layer (different from the id)

    cl_mem weight;  //(float*)
    cl_mem gain;    //(float*)
    cl_mem tau_syn; //(float*) questi parametri dovrebbero essere espressi in ms
    cl_mem delay;   //(float*)

    cl_int synapse_family_id;
    cl_int n_synapses;
};
typedef struct Synapse_device Synapse_device;

Synapse_device *create_synapses_device(cl_context ctx, cl_command_queue que, int n_synapses, bool new_synapses_ids);

Synapse *synapse_device_to_host(cl_command_queue que, Synapse_device *syn_device);

Synapse_device *synapse_host_to_device(cl_context ctx, cl_command_queue que, Synapse *syn_host);

Synapse_device *connect_device(cl_context ctx, cl_command_queue que, Layer_device *pre_layer_device, Layer_device *post_layer_device, int *conn_matrix);

Synapse_device *set_neurons_location_device(cl_context ctx, cl_command_queue que, Layer_device *layer_device, Synapse_device *synpase_device);

void free_synapses_device(Synapse_device *syn);
#endif