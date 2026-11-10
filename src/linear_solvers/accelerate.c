// Keep Accelerate's inline overloads and bitfield ABI behind a small C boundary.
#include <Accelerate/Accelerate.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    SparseMatrixStructure matrix;
    SparseOpaqueSymbolicFactorization symbolic;
    SparseOpaqueFactorization_Double numeric;
    bool factored;
    void *factor_workspace;
    void *solve_workspace;
    size_t solve_workspace_size;
} Cpo3Accelerate;

bool cpo3_accelerate_available(void) {
    if (__builtin_available(macOS 13.0, *)) return true;
    return false;
}

void cpo3_accelerate_free(Cpo3Accelerate *factor) {
    if (!factor) return;
    if (factor->factored) SparseCleanup(factor->numeric);
    SparseCleanup(factor->symbolic);
    free(factor->factor_workspace);
    free(factor->solve_workspace);
    free(factor);
}

Cpo3Accelerate *cpo3_accelerate_new(int n, long *columns, int *rows, int *status) {
    Cpo3Accelerate *factor = calloc(1, sizeof(*factor));
    if (!factor) { *status = SparseInternalError; return NULL; }
    factor->matrix = (SparseMatrixStructure) {
        .rowCount = n, .columnCount = n, .columnStarts = columns,
        .rowIndices = rows, .blockSize = 1,
        .attributes = { .kind = SparseSymmetric, .triangle = SparseUpperTriangle }
    };
    factor->symbolic = SparseFactor(SparseFactorizationLDLTTPP, factor->matrix);
    *status = factor->symbolic.status;
    if (*status != SparseStatusOK) { cpo3_accelerate_free(factor); return NULL; }
    factor->factor_workspace = malloc(factor->symbolic.workspaceSize_Double);
    if (!factor->factor_workspace && factor->symbolic.workspaceSize_Double) {
        *status = SparseInternalError;
        cpo3_accelerate_free(factor);
        return NULL;
    }
    return factor;
}

int cpo3_accelerate_factor(Cpo3Accelerate *factor, double *values) {
    SparseMatrix_Double matrix = { .structure = factor->matrix, .data = values };
    // Ruiz and the outer refinement already handle scaling and accuracy.
    SparseNumericFactorOptions options = {
        .scalingMethod = SparseScalingUser, .pivotTolerance = 0.01, .zeroTolerance = 0.0
    };
    if (factor->factored) {
        SparseRefactor(matrix, &factor->numeric, options, factor->factor_workspace);
    } else {
        factor->numeric = SparseFactor(factor->symbolic, matrix, options);
        factor->factored = true;
    }
    return factor->numeric.status;
}

int cpo3_accelerate_inertia(Cpo3Accelerate *factor, int *positive, int *zero, int *negative) {
    if (__builtin_available(macOS 13.0, *)) {
        return SparseGetInertia(factor->numeric, positive, zero, negative);
    }
    return -1;
}

int cpo3_accelerate_solve(Cpo3Accelerate *factor, int columns, double *rhs) {
    size_t size = factor->numeric.solveWorkspaceRequiredStatic
        + (size_t)columns * factor->numeric.solveWorkspaceRequiredPerRHS;
    if (size > factor->solve_workspace_size) {
        void *workspace = realloc(factor->solve_workspace, size);
        if (!workspace) return SparseInternalError;
        factor->solve_workspace = workspace;
        factor->solve_workspace_size = size;
    }
    DenseMatrix_Double matrix = {
        .rowCount = factor->matrix.rowCount, .columnCount = columns,
        .columnStride = factor->matrix.rowCount, .data = rhs
    };
    SparseSolve(factor->numeric, matrix, factor->solve_workspace);
    return SparseStatusOK;
}
