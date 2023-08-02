#include "visualize_device.h"

void visualize_neuron_layer_device(cl_command_queue que, Layer_device *layer_device)
{
    Layer *layer_host = neuron_device_to_host(que, layer_device);
    visualize_neuron_layer(layer_host);
    free_neurons(layer_host);
}