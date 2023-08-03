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

    Layer_device *layer_device_1 = create_neurons_device(init_kernel, que, ctx, 10, true);
    Layer_device *layer_device_2 = create_neurons_device(init_kernel, que, ctx, 10, true);
    Layer_device *all_layers[] = {layer_device_1, layer_device_2};


    int *conn = malloc(sizeof(int) * 10 * 10);
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
            conn[i * 10 + j] = 1;
    Synapse_device *syn_conn = connect_device(ctx, que, layer_device_1, layer_device_2, conn);
    Layer_device *layer_combined = combine_layers_device(ctx, que, all_layers, 2);
    syn_conn = set_neurons_location_device(ctx, que, layer_combined, syn_conn);

    clReleaseDevice(d);
    clReleaseKernel(init_kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
    return 0;
}
