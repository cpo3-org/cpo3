// Keep CHOLMOD's version-dependent structs behind a small C boundary.
#include <cholmod.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if CHOLMOD_MAIN_VERSION != 5 || CHOLMOD_SUB_VERSION < 2 || CHOLMOD_SUB_VERSION > 3
#error "The runtime adapter supports CHOLMOD 5.2.x and 5.3.x layouts only"
#endif

// Must match Api in cholmod.rs. All symbols come from the runtime library;
// this shim deliberately has no direct references to CHOLMOD functions.
typedef struct {
    int (*start)(cholmod_common *);
    int (*finish)(cholmod_common *);
    cholmod_factor *(*analyze)(cholmod_sparse *, cholmod_common *);
    int (*factorize)(cholmod_sparse *, cholmod_factor *, cholmod_common *);
    int (*solve2)(int, cholmod_factor *, cholmod_dense *, cholmod_sparse *,
                  cholmod_dense **, cholmod_sparse **, cholmod_dense **,
                  cholmod_dense **, cholmod_common *);
    int (*free_dense)(cholmod_dense **, cholmod_common *);
    int (*free_factor)(cholmod_factor **, cholmod_common *);
} Cpo3CholmodApi;

typedef struct {
    const Cpo3CholmodApi *api;
    cholmod_common common;
    cholmod_sparse matrix;
    cholmod_factor *factor;
    cholmod_dense *solution;
    cholmod_dense *workspace;
    cholmod_dense *extra_workspace;
} Cpo3Cholmod;

void cpo3_cholmod_free(Cpo3Cholmod *solver) {
    if (!solver) return;
    solver->api->free_dense(&solver->solution, &solver->common);
    solver->api->free_dense(&solver->workspace, &solver->common);
    solver->api->free_dense(&solver->extra_workspace, &solver->common);
    solver->api->free_factor(&solver->factor, &solver->common);
    solver->api->finish(&solver->common);
    free(solver);
}

Cpo3Cholmod *cpo3_cholmod_new(const Cpo3CholmodApi *api, size_t n,
                            int64_t *columns, int64_t *rows, double *values,
                            int threads, int *status) {
    Cpo3Cholmod *solver = calloc(1, sizeof(*solver));
    if (!solver) { *status = CHOLMOD_OUT_OF_MEMORY; return NULL; }
    solver->api = api;
    if (!api->start(&solver->common)) {
        *status = CHOLMOD_INVALID;
        free(solver);
        return NULL;
    }
    cholmod_common *common = &solver->common;
    common->print = 0;
    common->error_handler = NULL;
    // KKT matrices are indefinite: supernodal LL' cannot be used here.
    common->supernodal = CHOLMOD_SIMPLICIAL;
    common->final_ll = 0;
    common->final_pack = 0;
    common->dbound = 0;
    common->nmethods = 1;
    common->method[0].ordering = CHOLMOD_AMD;
    if (threads > 0) common->nthreads_max = threads;
    solver->matrix = (cholmod_sparse) {
        .nrow = n, .ncol = n, .nzmax = (size_t)columns[n],
        .p = columns, .i = rows, .x = values,
        .stype = 1, .itype = CHOLMOD_LONG, .xtype = CHOLMOD_REAL,
        .dtype = CHOLMOD_DOUBLE, .sorted = 1, .packed = 1
    };
    solver->factor = api->analyze(&solver->matrix, common);
    *status = common->status;
    if (!solver->factor) { cpo3_cholmod_free(solver); return NULL; }
    return solver;
}

int cpo3_cholmod_factor(Cpo3Cholmod *solver, size_t *positive, size_t *negative) {
    if (!solver->api->factorize(&solver->matrix, solver->factor, &solver->common)) {
        return solver->common.status;
    }
    if (solver->common.status != CHOLMOD_OK) return solver->common.status;
    if (solver->factor->minor != solver->factor->n) return CHOLMOD_NOT_POSDEF;
    // Simplicial LDL' stores D at the first entry of each column of L.
    const int64_t *columns = solver->factor->p;
    const double *values = solver->factor->x;
    *positive = 0;
    *negative = 0;
    for (size_t column = 0; column < solver->factor->n; ++column) {
        double pivot = values[columns[column]];
        if (!isfinite(pivot)) return CHOLMOD_INVALID;
        if (pivot == 0.0) return CHOLMOD_NOT_POSDEF;
        *positive += pivot > 0.0;
        *negative += pivot < 0.0;
    }
    return CHOLMOD_OK;
}

int cpo3_cholmod_solve(Cpo3Cholmod *solver, size_t columns, double *rhs) {
    size_t n = solver->matrix.nrow;
    cholmod_dense matrix = {
        .nrow = n, .ncol = columns, .nzmax = n * columns, .d = n,
        .x = rhs, .xtype = CHOLMOD_REAL, .dtype = CHOLMOD_DOUBLE
    };
    if (!solver->api->solve2(CHOLMOD_A, solver->factor, &matrix, NULL,
                            &solver->solution, NULL, &solver->workspace,
                            &solver->extra_workspace, &solver->common)) {
        return solver->common.status;
    }
    for (size_t column = 0; column < columns; ++column) {
        memcpy(rhs + column * n,
               (double *)solver->solution->x + column * solver->solution->d,
               n * sizeof(double));
    }
    return CHOLMOD_OK;
}
