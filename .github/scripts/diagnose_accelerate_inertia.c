// Standalone Apple API reproducer; deliberately does not include cpo3 code.
// clang -std=c11 -framework Accelerate diagnose_accelerate_inertia.c -o inertia
// Prints the runtime's behavior; cpo3's regression tests gate our workaround.
#include <Accelerate/Accelerate.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>

static void report(const char *operation, SparseOpaqueFactorization_Double factor,
                   double value) {
    assert(factor.status == SparseStatusOK);
    int positive = 0, zero = 0, negative = 0;
    int status = SparseGetInertia(factor, &positive, &zero, &negative);
    assert(status == 0);
    double rhs = value;
    DenseVector_Double vector = { .count = 1, .data = &rhs };
    SparseSolve(factor, vector);
    assert(fabs(rhs - 1.0) < 1e-12);
    printf("%s: A=[%g], inertia=(%d,%d,%d), expected=(%d,0,%d), solution=%g\n",
           operation, value, positive, zero, negative,
           value > 0.0, value < 0.0, rhs);
}

int main(void) {
    long columns[] = {0, 1};
    int rows[] = {0};
    double value = 1.0;
    SparseMatrix_Double matrix = {
        .structure = {
            .rowCount = 1, .columnCount = 1, .columnStarts = columns,
            .rowIndices = rows, .blockSize = 1,
            .attributes = { .kind = SparseSymmetric, .triangle = SparseUpperTriangle }
        },
        .data = &value
    };
    // Defaults throughout: Apple owns symbolic analysis, numeric storage,
    // scaling and workspaces. There is no Rust FFI or cpo3 wrapper involved.
    SparseOpaqueFactorization_Double factor = SparseFactor(SparseFactorizationLDLTTPP, matrix);
    report("SparseFactor", factor, value);
    value = 2.0;
    SparseRefactor(matrix, &factor);
    report("SparseRefactor #1", factor, value);
    value = -3.0;
    SparseRefactor(matrix, &factor);
    report("SparseRefactor #2", factor, value);
    value = 4.0;
    SparseRefactor(matrix, &factor);
    report("SparseRefactor #3", factor, value);
    SparseCleanup(factor);
    factor = SparseFactor(SparseFactorizationLDLTTPP, matrix);
    report("fresh SparseFactor", factor, value);
    SparseCleanup(factor);
    return 0;
}
