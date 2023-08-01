#include <stdio.h>
#include "bin_device/ocl_boiler.h"
#include "bin_device/neuron_device.h"

#include "bin_host/neuron.h"
#include <time.h>

int main(void)
{
    cl_int err;

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    cl_program prog = create_program("bin_device/neuron.ocl", ctx, d);

	cl_kernel init_kernel = clCreateKernel(prog, "initialize_neurons", &err);
    OCL_CHECK(err, "create kernel initialize_neurons");

    create_neurons_device(init_kernel, que, ctx, 10000000, true);
    printf("Neurons created.\n");

    clock_t start = clock();
    Layer *layer = create_neurons(10000000, true);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Tempo di esecuzione: %f millisecondi\n", time_taken);

    // clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
    return 0;
}
