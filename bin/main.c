#include <stdio.h>
#include "bin_device/ocl_boiler.h"
#include "bin_device/neuron_device.h"

#include "bin_host/neuron.h"
#include "bin_host/visualize.h"
#include "bin_device/visualize_device.h"
#include <time.h>
#include <unistd.h>

int main(void)
{
    cl_int err;
    clock_t start, end;
    double time_taken;

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    cl_program prog = create_program("bin_device/neuron.ocl", ctx, d);

	cl_kernel init_kernel = clCreateKernel(prog, "initialize_neurons", &err);
    OCL_CHECK(err, "create kernel initialize_neurons");

    Layer_device *layer_device = create_neurons_device(init_kernel, que, ctx, 10, true);
    Layer_device *layer_device_2 = create_neurons_device(init_kernel, que, ctx, 10, true);
    

    // start = clock();
    // Layer *layer = create_neurons(10, true);
    // end = clock();
    // time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    // printf("Tempo di esecuzione: %f millisecondi\n", time_taken);


    visualize_neuron_layer_device(que, layer_device);
    visualize_neuron_layer_device(que, layer_device_2);

    Layer_device *layers_device[2] = {layer_device_2, layer_device};
    Layer_device *combined_device = combine_layers_device(ctx, que, layers_device, 2);
    visualize_neuron_layer_device(que, combined_device);

    free_neurons_device(layer_device);
    free_neurons_device(layer_device_2);
    free_neurons_device(combined_device);
    clReleaseDevice(d);
    clReleaseKernel(init_kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
    return 0;
}
