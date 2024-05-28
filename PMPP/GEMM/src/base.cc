#include <iostream>
void mul(float *a, float *b, float *c, uint64_t n1, uint64_t n2, uint64_t n3)
{
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            for (int k = 0; k < n3; k++)
            {
                c[i * n3 + k] += a[i * n2 + j] * b[j * n3 + k];
            }
        }
    }
}