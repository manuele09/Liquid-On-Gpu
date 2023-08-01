#include "ocl_boiler.h"
#define OUTPUT 1

const char* clErrorString(cl_int err)
{
    const char* errString = NULL;

    switch (err)
    {
        case CL_SUCCESS:
            errString = "Success";
            break;
            
        case CL_DEVICE_NOT_FOUND:
            errString = "Device not found";
            break;
            
        case CL_DEVICE_NOT_AVAILABLE:
            errString = "Device not available";
            break;
            
        case CL_COMPILER_NOT_AVAILABLE:
            errString = "Compiler not available";
            break;
            
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            errString = "Memory object allocation failure";
            break;
            
        case CL_OUT_OF_RESOURCES:
            errString = "Out of resources";
            break;
            
        case CL_OUT_OF_HOST_MEMORY:
            errString = "Out of host memory";
            break;
            
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            errString = "Profiling info not available";
            break;
            
        case CL_MEM_COPY_OVERLAP:
            errString = "Memory copy overlap";
            break;
            
        case CL_IMAGE_FORMAT_MISMATCH:
            errString = "Image format mismatch";
            break;
            
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            errString = "Image format not supported";
            break;
            
        case CL_BUILD_PROGRAM_FAILURE:
            errString = "Build program failure";
            break;
            
        case CL_MAP_FAILURE:
            errString = "Map failure";
            break;
            
        case CL_INVALID_VALUE:
            errString = "Invalid value";
            break;
            
        case CL_INVALID_DEVICE_TYPE:
            errString = "Invalid device type";
            break;
            
        case CL_INVALID_PLATFORM:
            errString = "Invalid platform";
            break;
            
        case CL_INVALID_DEVICE:
            errString = "Invalid device";
            break;
            
        case CL_INVALID_CONTEXT:
            errString = "Invalid context";
            break;
            
        case CL_INVALID_QUEUE_PROPERTIES:
            errString = "Invalid queue properties";
            break;
            
        case CL_INVALID_COMMAND_QUEUE:
            errString = "Invalid command queue";
            break;
            
        case CL_INVALID_HOST_PTR:
            errString = "Invalid host pointer";
            break;
            
        case CL_INVALID_MEM_OBJECT:
            errString = "Invalid memory object";
            break;
            
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            errString = "Invalid image format descriptor";
            break;
            
        case CL_INVALID_IMAGE_SIZE:
            errString = "Invalid image size";
            break;
            
        case CL_INVALID_SAMPLER:
            errString = "Invalid sampler";
            break;
            
        case CL_INVALID_BINARY:
            errString = "Invalid binary";
            break;
            
        case CL_INVALID_BUILD_OPTIONS:
            errString = "Invalid build options";
            break;
            
        case CL_INVALID_PROGRAM:
            errString = "Invalid program";
            break;
            
        case CL_INVALID_PROGRAM_EXECUTABLE:
            errString = "Invalid program executable";
            break;
            
        case CL_INVALID_KERNEL_NAME:
            errString = "Invalid kernel name";
            break;
            
        case CL_INVALID_KERNEL_DEFINITION:
            errString = "Invalid kernel definition";
            break;
            
        case CL_INVALID_KERNEL:
            errString = "Invalid kernel";
            break;
            
        case CL_INVALID_ARG_INDEX:
            errString = "Invalid argument index";
            break;
            
        case CL_INVALID_ARG_VALUE:
            errString = "Invalid argument value";
            break;
            
        case CL_INVALID_ARG_SIZE:
            errString = "Invalid argument size";
            break;
            
        case CL_INVALID_KERNEL_ARGS:
            errString = "Invalid kernel arguments";
            break;
            
        case CL_INVALID_WORK_DIMENSION:
            errString = "Invalid work dimension";
            break;
            
        case CL_INVALID_WORK_GROUP_SIZE:
            errString = "Invalid work group size";
            break;
            
        case CL_INVALID_WORK_ITEM_SIZE:
            errString = "invalid work item size";
            break;
            
        case CL_INVALID_GLOBAL_OFFSET:
            errString = "Invalid global offset";
            break;
            
        case CL_INVALID_EVENT_WAIT_LIST:
            errString = "Invalid event wait list";
            break;
            
        case CL_INVALID_EVENT:
            errString = "Invalid event";
            break;
            
        case CL_INVALID_OPERATION:
            errString = "Invalid operation";
            break;
            
        case CL_INVALID_GL_OBJECT:
            errString = "Invalid OpenGL object";
            break;
            
        case CL_INVALID_BUFFER_SIZE:
            errString = "Invalid buffer size";
            break;
            
        case CL_INVALID_MIP_LEVEL:
            errString = "Invalid MIP level";
            break;
    }
    
    return errString;
}


/* Check an OpenCL error status, printing a message and exiting
 * in case of failure
 */
void ocl_check(cl_int err, const char *msg, ...) {
	if (err != CL_SUCCESS) {
		char msg_buf[BUFSIZE + 1];
		va_list ap;
		va_start(ap, msg);
		vsnprintf(msg_buf, BUFSIZE, msg, ap);
		va_end(ap);
		msg_buf[BUFSIZE] = '\0';
		fprintf(stderr, "%s (%d): \033[0;31m%s\033[0m\n", msg_buf, err, clErrorString(err));
		exit(1);
	}
}



// Return the ID of the platform specified in the OCL_PLATFORM
// environment variable (or the first one if none specified)
cl_platform_id select_platform() {
	cl_uint nplats;
	cl_int err;
	cl_platform_id *plats;
	const char * const env = getenv("OCL_PLATFORM");
	cl_uint nump = 0;
	if (env && env[0] != '\0')
		nump = atoi(env);
    
	err = clGetPlatformIDs(0, NULL, &nplats);
	ocl_check(err, "counting platforms");
	#if OUTPUT
	printf("number of platforms: %u\n", nplats);
	#endif

	plats = malloc(nplats*sizeof(*plats));

	err = clGetPlatformIDs(nplats, plats, NULL);
	ocl_check(err, "getting platform IDs");

	if (nump >= nplats) {
		fprintf(stderr, "no platform number %u", nump);
		exit(1);
	}

	cl_platform_id choice = plats[nump];

	char buffer[BUFSIZE];

	err = clGetPlatformInfo(choice, CL_PLATFORM_NAME, BUFSIZE,
		buffer, NULL);
	ocl_check(err, "getting platform name");

	#if OUTPUT	
	printf("selected platform %d: %s\n", nump, buffer);	
	#endif

	return choice;
}

// Return the ID of the device (of the given platform p) specified in the
// OCL_DEVICE environment variable (or the first one if none specified)
cl_device_id select_device(cl_platform_id p)
{
	cl_uint ndevs;
	cl_int err;
	cl_device_id *devs;
	const char * const env = getenv("OCL_DEVICE");
	cl_uint numd = 0;
	if (env && env[0] != '\0')
		numd = atoi(env);

	err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, NULL, &ndevs);
	ocl_check(err, "counting devices");

	#if OUTPUT	
	printf("number of devices: %u\n", ndevs);
	#endif

	devs = malloc(ndevs*sizeof(*devs));

	err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, ndevs, devs, NULL);
	ocl_check(err, "devices #2");

	if (numd >= ndevs) {
		fprintf(stderr, "no device number %u", numd);
		exit(1);
	}

	cl_device_id choice = devs[numd];

	char buffer[BUFSIZE];

	err = clGetDeviceInfo(choice, CL_DEVICE_NAME, BUFSIZE,
		buffer, NULL);
	ocl_check(err, "device name");
	#if OUTPUT	
	printf("selected device %d: %s\n", numd, buffer);
	#endif

	return choice;
}

// Create a one-device context
cl_context create_context(cl_platform_id p, cl_device_id d)
{
	cl_int err;

	cl_context_properties ctx_prop[] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)p, 0
	};

	cl_context ctx = clCreateContext(ctx_prop, 1, &d,
		NULL, NULL, &err);
	ocl_check(err, "create context");

	return ctx;
}

// Create a command queue for the given device in the given context
cl_command_queue create_queue(cl_context ctx, cl_device_id d)
{
	cl_int err;

	//cl_command_queue que = clCreateCommandQueue(ctx, d, CL_QUEUE_PROFILING_ENABLE, &err);
	const cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
	cl_command_queue que = clCreateCommandQueueWithProperties(ctx, d, properties, &err);
	ocl_check(err, "create queue");
	return que;
}

// Compile the device part of the program, stored in the external
// file `fname`, for device `dev` in context `ctx`
cl_program create_program(const char * const fname, cl_context ctx,
	cl_device_id dev)
{
	cl_int err, errlog;
	cl_program prg;

	char src_buf[BUFSIZE + 1];
	char *log_buf = NULL;
	size_t logsize;
	const char* buf_ptr = src_buf;
	time_t now = time(NULL);

	memset(src_buf, 0, BUFSIZE);

	snprintf(src_buf, BUFSIZE, "// %s#include \"%s\"\n",
		ctime(&now), fname);
	#if OUTPUT	
	printf("compiling:\n%s", src_buf);
	#endif
	prg = clCreateProgramWithSource(ctx, 1, &buf_ptr, NULL, &err);
	ocl_check(err, "create program");

	err = clBuildProgram(prg, 1, &dev, "-I.", NULL, NULL);
	errlog = clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG,
		0, NULL, &logsize);
	ocl_check(errlog, "get program build log size");
	log_buf = malloc(logsize);
	errlog = clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG,
		logsize, log_buf, NULL);
	ocl_check(errlog, "get program build log");
	while (logsize > 0 &&
		(log_buf[logsize-1] == '\n' ||
		 log_buf[logsize-1] == '\0')) {
		logsize--;
	}
	if (logsize > 0) {
		log_buf[logsize] = '\n';
		log_buf[logsize+1] = '\0';
	} else {
		log_buf[logsize] = '\0';
	}
	#if OUTPUT	
	printf("=== BUILD LOG ===\n%s\n=========\n", log_buf);
	#endif
	ocl_check(err, "build program");

	return prg;
}

// Runtime of an event, in nanoseconds. Note that if NS is the
// runtimen of an event in nanoseconds and NB is the number of byte
// read and written during the event, NB/NS is the effective bandwidth
// expressed in GB/s
cl_ulong runtime_ns(cl_event evt)
{
	cl_int err;
	cl_ulong start, end;
	err = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START,
		sizeof(start), &start, NULL);
    OCL_CHECK(err, "get start");
	err = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,
		sizeof(end), &end, NULL);
	OCL_CHECK(err, "get end");
	return (end - start);
}

cl_ulong total_runtime_ns(cl_event from, cl_event to)
{
	cl_int err;
	cl_ulong start, end;
	err = clGetEventProfilingInfo(from, CL_PROFILING_COMMAND_START,
		sizeof(start), &start, NULL);
	ocl_check(err, "get start");
	err = clGetEventProfilingInfo(to, CL_PROFILING_COMMAND_END,
		sizeof(end), &end, NULL);
	ocl_check(err, "get end");
	return (end - start);
}


// Runtime of an event, in milliseconds
double runtime_ms(cl_event evt)
{
	return runtime_ns(evt)*1.0e-6;
}

double total_runtime_ms(cl_event from, cl_event to)
{
	return total_runtime_ns(from, to)*1.0e-6;
}

/* round gws to the next multiple of lws */
size_t round_mul_up(size_t gws, size_t lws)
{
	return ((gws + lws - 1)/lws)*lws;
}

void Profile(cl_event event, char *name)
{
    double init_runtime = runtime_ms(event);
    printf("%s: %gms\n", name, init_runtime);
    
}
