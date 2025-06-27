#define LOC_SIZE 64

__kernel void prefix_sum(__global const float* input,
                         __global float* output,
                         __global float* group_sums) 
{
    uint gid = get_global_id(0);     // ���������� ������
    uint lid = get_local_id(0);      // ��������� ������ ������ ������
    uint group_id = get_group_id(0); // ������ ������
    uint local_size = get_local_size(0); // ������ ��������� ������

    __local float local_data[LOC_SIZE]; // ��������� ������ ��� �������� ������������� ����

    // ����������� ������ �� ���������� ������ � ���������
    local_data[lid] = input[gid];
    
    /*
    if (gid < get_global_size(0)) {
        local_data[lid] = input[gid];
    } else {
        local_data[lid] = 0.0f; // ���������� ������ ��� �������� �����
    }
    */

    barrier(CLK_LOCAL_MEM_FENCE);

    // ���������� ���������� ����� ������ ��������� ������
    for (int step = 1; step < local_size; step *= 2) {
        float temp = 0.0f;
        if (lid >= step) {
            temp = local_data[lid - step];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        local_data[lid] += temp;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ������ ��������� ���������� ����� � ���������� ������

    if (gid < get_global_size(0)) {
        output[gid] = local_data[lid];
    }

    // ��������� ������� ��������� ������ ����������� ��� ��������� ����� ������
    if (lid == local_size - 1) {
        group_sums[group_id] = local_data[lid];
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // ������ ����: ��������� ���� � ������ ���������� �����
    if (group_id > 0) {
        float correction = 0.0f;

        // ����������� ����� ���� ���������� �����
        for (int i = 0; i < group_id; i++) {
            correction += group_sums[i];
        }

        // ��������� ��������� � ������� �������� ������� ������
        if (gid < get_global_size(0)) {
            output[gid] += correction;
        }
    }
}