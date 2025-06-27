#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <map>
#include <algorithm>
#include <cmath>
#define LOC_SIZE 64

cl_device_type parseDeviceType(const std::string& typeStr) {
    if (typeStr == "cpu") {
        return CL_DEVICE_TYPE_CPU;
    }
    else if (typeStr == "gpu") {
        return CL_DEVICE_TYPE_GPU;
    }
    else if (typeStr == "dgpu" || typeStr == "igpu") {
        return CL_DEVICE_TYPE_GPU;
    }
    else if (typeStr == "all") {
        return CL_DEVICE_TYPE_ALL;
    }
    else {
        return 0;
    }
}

std::vector<cl_device_id> getDevicesByType(cl_platform_id platform, cl_device_type deviceType, const std::string& deviceTypeStr) {
    cl_uint numDevices;
    cl_int ret = clGetDeviceIDs(platform, deviceType, 0, nullptr, &numDevices);
    if (ret != CL_SUCCESS || numDevices == 0) {
        return {};
    }
    std::vector<cl_device_id> devices(numDevices);
    ret = clGetDeviceIDs(platform, deviceType, numDevices, devices.data(), nullptr);
    if (ret != CL_SUCCESS) {
        return {};
    }

    // Фильтрация устройств по типу (дискретная или интегрированная)
    if (deviceTypeStr == "igpu" || deviceTypeStr == "dgpu") {
        std::vector<cl_device_id> filteredDevices;
        for (const auto& device : devices) {
            cl_bool unifiedMemory;
            clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unifiedMemory), &unifiedMemory, nullptr);
            if ((deviceTypeStr == "igpu" && unifiedMemory == CL_TRUE) ||
                (deviceTypeStr == "dgpu" && unifiedMemory == CL_FALSE)) {
                filteredDevices.push_back(device);
            }
        }
        return filteredDevices;
    }

    return devices;
}

int main(int argc, char* argv[]) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 < argc) {
            args[argv[i]] = argv[i + 1];
        }
        else {
            std::cout << "Usage: lab0.exe < --input file_name > < --output file_name > [ --device-type { dgpu | igpu | gpu | cpu | all } ] [ --device-index index ]\n";;
            return 1;
        }
    }
    // Переменные для хранения типа и индекса устройства
    std::string deviceTypeStr = args.count("--device-type") ? args["--device-type"] : "all";
    cl_int deviceIndex = args.count("--device-index") ? std::stoi(args["--device-index"]) : 0;

    cl_device_type deviceType = parseDeviceType(deviceTypeStr);
    if (deviceType == NULL) {
        std::cerr << "Invalid device type.\n";
        return 1;
    }

    std::string input_filename;
    std::string output_filename;
    cl_platform_id platformID = NULL;

    cl_uint numDevices;
    cl_int ret;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_filename = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            output_filename = argv[++i];
        }
    }

    if (input_filename.empty() || output_filename.empty()) {
        std::cerr << "Usage: " << argv[0] << " --input <input_file> --output <output_file>" << std::endl;
        return 1;
    }

    cl_uint numPlatforms;
    ret = clGetPlatformIDs(0, nullptr, &numPlatforms);

    std::vector<cl_platform_id> platforms(numPlatforms);
    ret = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to get platform IDs.\n";
        return 1;
    }

    std::vector<cl_device_id> allDevices;
    for (auto& platform : platforms) {
        auto devices = getDevicesByType(platform, deviceType, deviceTypeStr);
        allDevices.insert(allDevices.end(), devices.begin(), devices.end());
    }
    if (allDevices.empty()) {
        std::cerr << "No devices found.\n";
        return 1;
    }

    // Сортировка устройств по типу
    std::vector<std::pair<cl_device_id, int>> devicesWithIndex;
    for (auto device : allDevices) {
        cl_device_type type;
        cl_bool unifiedMemory;
        clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unifiedMemory), &unifiedMemory, nullptr);

        int index_for_all;
        if (type == CL_DEVICE_TYPE_GPU && unifiedMemory == CL_FALSE) {
            index_for_all = 0; // Дискретная видеокарта
        }
        else if (type == CL_DEVICE_TYPE_GPU && unifiedMemory == CL_TRUE) {
            index_for_all = 1; // Интегрированная видеокарта
        }
        else if (type == CL_DEVICE_TYPE_CPU) {
            index_for_all = 2; // Процессор
        }
        else {
            index_for_all = 3; // Остальные устройства
        }
        devicesWithIndex.emplace_back(device, index_for_all);
    }

    std::sort(devicesWithIndex.begin(), devicesWithIndex.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
        });

    std::vector<cl_device_id> sortedDevices;
    for (const auto& pair : devicesWithIndex) {
        sortedDevices.push_back(pair.first);
    }

    // Обновление allDevices отсортированными устройствами
    allDevices = sortedDevices;

    if (deviceIndex >= allDevices.size()) {
        std::cerr << "Device index out of range. Using device 0 instead.\n";
        return 1;
    }

    cl_device_id deviceID = allDevices[deviceIndex];

    char deviceName[128];
    clGetDeviceInfo(deviceID, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    std::cout << "Selected device: " << deviceName << "\n";
    std::cout << "Selected device index: " << deviceIndex << "\n";

    // Чтение данных из файла
    std::ifstream input_file(input_filename);
    if (!input_file.is_open()) {
        std::cerr << "Failed to open input file." << std::endl;
        return 1;
    }

    std::vector<float> input_data;
    int n;
    input_file >> n;  // Чтение количества элементов
    input_data.resize(n);

    for (int i = 0; i < n; ++i) {
        input_file >> input_data[i];  // Чтение каждого элемента
    }
    input_file.close();  // Закрытие файла после чтения

    // Чтение исходного кода ядра из файла
    std::ifstream kernelFile("kernel.cl");
    if (!kernelFile.is_open()) {
        std::cerr << "Error opening kernel file." << std::endl;
        return 1;
    }

    // Создание контекста OpenCL
    cl_context context = clCreateContext(nullptr, 1, &deviceID, nullptr, nullptr, NULL);

    // Создание очереди команд OpenCL
    cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, NULL);

    std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    const char* kernelSourceCStr = kernelSource.c_str();

    // Создание программы OpenCL с исходным кодом ядра
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceCStr, NULL, NULL);

    // Сборка программы
    clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

    // Размер локальной группы
    size_t local_size = LOC_SIZE;
    size_t global_size = (n + local_size - 1) / local_size * local_size; // Округляем до ближайшего большего кратного

    // Создание буферов
    cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(cl_float), input_data.data(), NULL);
    cl_mem buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(cl_float), NULL, NULL);
    cl_mem buffer_group_sums = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (global_size / local_size) * sizeof(float), NULL, NULL);
    
    cl_event event;

    // Создание и установка аргументов ядра
    cl_kernel kernel = clCreateKernel(program, "prefix_sum", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_C);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_group_sums);

    // Запуск ядра
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event);
   
    // Чтение результатов
    std::vector<float> output_data(n);
    clEnqueueReadBuffer(commandQueue, buffer_C, CL_TRUE, 0, n * sizeof(float), output_data.data(), 0, NULL, NULL);

    std::ofstream output_file(output_filename);
    if (!output_file.is_open()) {
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }

    output_file << std::fixed << std::setprecision(6); // 6 знаков после запятой 

    for (const auto& val : output_data) {
        output_file << val << " ";
    }
    output_file.close();

    cl_ulong start_time;
    cl_ulong end_time;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
    
    printf("Execution time of kernel: %g ms\n", (end_time - start_time) * 1E-6);
    printf("LOCAL_WORK_SIZE [%i, %i]\nWI_WORK %i\n", LOC_SIZE, 1, 1);

    clReleaseMemObject(buffer_A);
    clReleaseMemObject(buffer_C);
    clReleaseMemObject(buffer_group_sums);
    clReleaseKernel(kernel);
    clReleaseEvent(event);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    return 0;
}
