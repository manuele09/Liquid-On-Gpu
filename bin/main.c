#include <stdio.h>
#include "ocl_boiler.h"
#include "neuron_device.h"


int main(void)
{
    cl_int err;

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    cl_program prog = create_program("neuron.ocl", ctx, d);

	cl_kernel init_kernel = clCreateKernel(prog, "initialize_neurons", &err);
    OCL_CHECK(err, "create kernel initialize_neurons");


    create_neurons_device(init_kernel, que, ctx, 1000, true);

    printf("Main finito\n");
    // clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
    return 0;
}
