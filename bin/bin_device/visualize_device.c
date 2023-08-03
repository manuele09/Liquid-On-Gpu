#include "visualize_device.h"

void visualize_neuron_layer_device(cl_command_queue que, Layer_device *layer_device)
{
    Layer *layer_host = neuron_device_to_host(que, layer_device);
    visualize_neuron_layer(layer_host);
    free_neurons(layer_host);
}

void visualize_synapse_device(cl_command_queue que, Synapse_device *syn_device)
{
    Synapse *syn_host = synapse_device_to_host(que, syn_device);
    visualize_synapse(syn_host);
    free_synapses(syn_host);
}