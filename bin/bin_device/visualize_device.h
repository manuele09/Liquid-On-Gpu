#ifndef VISUALIZE_DEVICE_H_
#define VISUALIZE_DEVICE_H_

#include "ocl_boiler.h"
#include "neuron_device.h"
#include "../bin_host/visualize.h"

void visualize_neuron_layer_device(cl_command_queue que, Layer_device *layer_device);
#endif