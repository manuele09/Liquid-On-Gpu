#include <stdio.h>
#include "bin_device/ocl_boiler.h"
#include "bin_device/neuron_device.h"

#include "bin_host/neuron.h"
#include "bin_host/visualize.h"
#include "bin_device/visualize_device.h"
#include <time.h>
#include <unistd.h>
#include "bin_device/synapse_device.h"

int main(void)
{
    cl_int err;
    clock_t start, end;
    double time_taken;
    int steps = 1000;

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    cl_program prog = create_program("bin_device/neuron.ocl", ctx, d);

    cl_kernel init_kernel = clCreateKernel(prog, "initialize_neurons", &err);
    OCL_CHECK(err, "create kernel initialize_neurons");
    cl_kernel simulate_kernel = clCreateKernel(prog, "simulate_neurons", &err);
    OCL_CHECK(err, "create kernel simulate_neurons");

    // Layer_device *layer_device = create_neurons_device(init_kernel, que, ctx, 10000000, true);
    // // visualize_neuron_layer_device(que, layer_device);
    // simulate_neurons_device(que, simulate_kernel, layer_device, steps, 0.1f);

    // Layer *layer_host = neuron_device_to_host(que, layer_device);
    // start = clock();
    // for (int i = 0; i < steps; i++)
    //     simulate_neurons(layer_host, 0.1f, NULL);
    // end = clock();
    // time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    // printf("Tempo di esecuzione host: %f millisecondi\n", time_taken);
    // free_neurons_device(layer_device);

    Synapse_device *syn_device = create_synapses_device(ctx, 10, true);
    visualize_synapse_device(que, syn_device);

    clReleaseDevice(d);
    clReleaseKernel(init_kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
    return 0;
}
