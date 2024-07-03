#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    return (*mat).data[(*mat).cols * row + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    mat->data[mat->cols * row + col] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    // 6. Set the `ref_cnt` field to 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows <= 0 || cols <= 0)  {
        return -1;
    }
    matrix *matmem = malloc(sizeof(matrix));
    matmem->data = calloc(rows * cols, sizeof(double));
    if (matmem == NULL || matmem->data == NULL) {
        return -2;
    }
    matmem->rows = rows;
    matmem->cols = cols;
    matmem->parent = NULL;
    matmem->ref_cnt = 1;
    *mat = matmem;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    if (mat == NULL) {
        return;
    }
    if (mat->parent == NULL) {
        mat->ref_cnt -= 1;
        if (mat->ref_cnt == 0) {
            free(mat->data);
            free(mat);
        }
        return;
    }
    deallocate_matrix(mat->parent);
    free(mat);
    return;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (from == NULL) {
        return -1;
    }
    if (rows <= 0 || cols <= 0)  {
        return -1;
    }
    matrix *matmem = malloc(sizeof(matrix));
    if (matmem == NULL) {
        return -2;
    }
    matmem->data = from->data + offset;
    matmem->rows = rows;
    matmem->cols = cols;
    matmem->parent = from;
    from->ref_cnt += 1;
    *mat = matmem;
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    // Task 1.5 TODO
    int cells = mat->rows * mat->cols;
    double *m1 = mat->data;

    for (int i=0; i < cells/4 * 4; i+=4) {
        __m256d mat127 = _mm256_set1_pd(val);
        _mm256_storeu_pd (m1 + i, mat127);
    }
    // tail case
    for(unsigned int i=cells/4 * 4; i < cells; i++) {
        m1[i] = val;
    }
}

/*

 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    int cells = mat->rows * mat->cols;
    double *m1 = mat->data;
    double *r = result->data;
    __m256d zeroes = _mm256_set1_pd(0.00);
    #pragma omp parallel for 
    for (int i=0; i < cells/16 * 16; i+=16) {
        __m256d load = _mm256_loadu_pd(m1 + i);
        __m256d sub = _mm256_sub_pd(zeroes, load);
        __m256d max = _mm256_max_pd(load, sub);
        _mm256_storeu_pd(r + i, max);
        load = _mm256_loadu_pd(m1 + i + 4);
        sub = _mm256_sub_pd(zeroes, load);
        max = _mm256_max_pd(load, sub);
        _mm256_storeu_pd(r + i + 4, max);
        load = _mm256_loadu_pd(m1 + i + 8);
        sub = _mm256_sub_pd(zeroes, load);
        max = _mm256_max_pd(load, sub);
        _mm256_storeu_pd(r + i + 8, max);
        load = _mm256_loadu_pd(m1 + i + 12);
        sub = _mm256_sub_pd(zeroes, load);
        max = _mm256_max_pd(load, sub);
        _mm256_storeu_pd(r + i + 12, max);
    }
    // tail case
    for(unsigned int i=cells/16 * 16; i < cells; i++) {
        r[i] = fabs(m1[i]);
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    int cells = mat1->rows * mat1->cols;
    double *m1 = mat1->data;
    double *m2 = mat2->data;
    double *r = result->data;

    #pragma omp parallel for 
    for(unsigned int i=0; i < cells/16 * 16; i+=16) {
        // Load array elements 0-3 into a temporary vector register
        __m256d load1 = _mm256_loadu_pd(m1 + i);
        __m256d load2 = _mm256_loadu_pd(m2 + i);
        _mm256_storeu_pd(r + i, _mm256_add_pd(load1, load2));
        load1 = _mm256_loadu_pd(m1 + i + 4);
        load2 = _mm256_loadu_pd(m2 + i + 4);
        _mm256_storeu_pd(r + i + 4, _mm256_add_pd(load1, load2));
        load1 = _mm256_loadu_pd(m1 + i + 8);
        load2 = _mm256_loadu_pd(m2 + i + 8);
        _mm256_storeu_pd(r + i + 8, _mm256_add_pd(load1, load2));
        load1 = _mm256_loadu_pd(m1 + i + 12);
        load2 = _mm256_loadu_pd(m2 + i + 12);
        _mm256_storeu_pd(r + i + 12, _mm256_add_pd(load1, load2));
    }

    for(unsigned int i=cells/16 * 16; i < cells; i++) {
        r[i] = m1[i] + m2[i];
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    return 0;
}

// Tranpose function for parallize 
void transpose(matrix *dest, matrix *src) {
    #pragma omp parallel for 
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            set(dest, j, i, get(src, i, j));
        }
    }
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    matrix *matmem;
    double *m1 = mat1->data;
    allocate_matrix(&matmem, mat2->cols, mat2->rows);
    transpose(matmem, mat2);
    double *m2 = matmem->data;
    int rrows = result->rows;
    int rcols = result->cols;
    int cols = mat1->cols;
    int round_cols = cols / 16 * 16;
    #pragma omp parallel for
    for (int i = 0; i < rrows; i++) {
        for (int j = 0; j < rcols; j++) {
            double count = 0.00;
            __m256d sum_vec = _mm256_set1_pd(0.00);
            for(unsigned int k = 0; k < round_cols; k += 16) {
                double *d1 = m1 + i * cols + k;
                double *d2 = m2 + j * cols + k;
                
                sum_vec = _mm256_fmadd_pd(_mm256_loadu_pd(d1), _mm256_loadu_pd(d2), sum_vec);
                sum_vec = _mm256_fmadd_pd(_mm256_loadu_pd(d1 + 4), _mm256_loadu_pd(d2 + 4), sum_vec);
                sum_vec = _mm256_fmadd_pd(_mm256_loadu_pd(d1 + 8), _mm256_loadu_pd(d2 + 8), sum_vec);
                sum_vec = _mm256_fmadd_pd(_mm256_loadu_pd(d1 + 12), _mm256_loadu_pd(d2 + 12), sum_vec);
            }

            for(unsigned int k = cols/16 * 16; k < cols / 4 * 4; k+=4) {
                double *d1 = m1 + i * cols + k;
                double *d2 = m2 + j * cols + k;
                sum_vec = _mm256_fmadd_pd(_mm256_loadu_pd(d1), _mm256_loadu_pd(d2), sum_vec);
            }
            double matmem_arr[4];
            _mm256_storeu_pd(matmem_arr, sum_vec);
            count += matmem_arr[0] + matmem_arr[1] + matmem_arr[2] + matmem_arr[3];
            
            for(unsigned int k = cols/4 * 4; k < cols; k++) {
                count += get(mat1, i, k) * get(matmem, j, k);
            }
            set(result, i, j, count);
        }
    }
    deallocate_matrix(matmem);
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO
    // initialize identity matrix
    matrix *matmem;
    matrix *matmem2;
    matrix *matmem3;
    allocate_matrix(&matmem, mat->rows, mat->cols);
    allocate_matrix(&matmem2, mat->rows, mat->cols);
    allocate_matrix(&matmem3, mat->rows, mat->cols);
    memcpy(matmem3->data, mat->data, sizeof(double) * result->rows * result->cols);
    
    fill_matrix(result, 0.00);
    for (int k = 0; k < mat->rows; k++) {
        set(result, k, k, 1.00);
    }
    if (pow == 0) {
        return 0;
    }
    while (pow > 1) {
        if (pow % 2 == 0) {
            memcpy(matmem2->data, matmem3->data, sizeof(double) * result->rows * result->cols);
            mul_matrix(matmem3, matmem2, matmem2);
            pow = pow/2;
        } else {
            memcpy(matmem->data, result->data, sizeof(double) * result->rows * result->cols);
            memcpy(matmem2->data, matmem3->data, sizeof(double) * result->rows * result->cols);
            __m256d t1 = _mm256_loadu_pd(matmem->data);
            __m256d t2 = _mm256_loadu_pd(matmem2->data);
            mul_matrix(result, matmem2, matmem);
            mul_matrix(matmem3, matmem2, matmem2);
            pow = (pow - 1)/2;
        }
    }
    memcpy(matmem->data, result->data, sizeof(double) * result->rows * result->cols);
    mul_matrix(result, matmem, matmem3);
    deallocate_matrix(matmem3);
    deallocate_matrix(matmem2);
    deallocate_matrix(matmem);
    return 0;
    
    /*
    fill_matrix(result, 0);
    matrix *temp; 
    allocate_matrix(&temp, mat->rows, mat->cols);
    add_matrix(temp, temp, mat);

    #pragma omp parallel for 
    for (int i=0; i<mat->rows; i++) {
        result->data[i * mat->rows + i] = 1;
    }

    while (pow) {
        if (pow & 1) {
            mul_matrix(result, result, temp);
        }
        pow = pow >> 1;
        mul_matrix(temp, temp, temp);
    }
    deallocate_matrix(temp);
    return 0;
    */
}
