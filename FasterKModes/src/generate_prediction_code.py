import os
import sys
KAIGYO = "\n"
MIN_DEPTH = 1
MAX_DEPTH = 16
MIN_TREE_UNROLL = 1
MAX_TREE_UNROLL = 16
NEW_PATH = "./pred"

os.makedirs(NEW_PATH, exist_ok=True)


# # Create Predict Function for Scikit-Learn DecisionTree and ExtraTree, 
# single output (binary classification and single-target classification)
for max_depth in range(MIN_DEPTH, MAX_DEPTH+1):
        t = 1
        tmp_indices_trees = [f"tmp_idx{i}" for i in range(t)]
        tmp_slide_trees   = [f"slide_t{i}" for i in range(t)]

        base = f"""
void predict(
        double *responses, double *events, 
        int *features, double *thresholds, double *values, 
        int n_rows, int n_cols, int max_depth, int n_outputs, int num_threads
    ){{
    double tmp_sum; 
    int n_idx;
    int l_idx;
    
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for(int n=0; n<n_rows; n++){{
        n_idx= n*n_cols;
        double* events_head = &events[n_idx];
        l_idx = 0;
{KAIGYO.join([f'        l_idx = 2*l_idx + 1 + (events_head[features[l_idx]] > thresholds[l_idx]);' for dep in range(max_depth) for idx, sld in zip(tmp_indices_trees, tmp_slide_trees)])}
        responses[n] = values[l_idx];
    }}
}}
    """
        with open(f"./{NEW_PATH}/sk_predict_single_output_single_tree_depth{max_depth:0=2}_tree{t:0=2}.c", "w") as f:
            f.write("#include <omp.h>\n")
            f.write("#include <stdio.h>\n")
            f.write("#include <math.h>\n")
            f.write("#include <stdlib.h>\n")
            f.write(base+"\n")

# # Create Predict Function for CatBoost
for max_depth in range(MIN_DEPTH, MAX_DEPTH+1):
    shift_values = [pow(2,i) for i in range(0, max_depth)]
    base = f"""
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

void predict(
        double *responses, double *events, 
        int *features, double *thresholds, double *values, 
        int n_rows, int n_cols, int max_depth, int n_trees, int n_outputs, int num_threads
    ){{

    uint16_t leaf_idx;
    int shift_val[{max_depth}] = {{}};
    for(int t=0; t<n_trees; t++){{
        int64_t f_start_idx = t * max_depth;
        int64_t t_start_idx = t * max_depth;
        int64_t v_start_idx = t * n_outputs * max_depth;

        int *f_ptr = &features[f_start_idx];
        double *t_ptr = &thresholds[t_start_idx];
        double *v_ptr = &values[v_start_idx];
        for(int i=0; i<n_rows; i++){{
            leaf_idx = 0;
            double *row = &events[i*n_cols];
            for(int d=0; d<max_depth; d++){{
                leaf_idx |= (row[f_ptr[d]]>=t_ptr[d]) * shift_val[d];
            }}
            responses[i] += values[leaf_idx + t*16];
        }}
    }}
}}
    """
    with open(f"./{NEW_PATH}/sk_predict_single_output_single_tree_depth{max_depth:0=2}_tree{t:0=2}.c", "w") as f:
        f.write("#include <omp.h>\n")
        f.write("#include <stdio.h>\n")
        f.write("#include <math.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write(base+"\n")





# # Create Predict Function for Scikit-Learn DecisionTree and ExtraTree, multiple output (multiclass classification, class probabilities)
for max_depth in range(MIN_DEPTH, MAX_DEPTH+1):
    t = 1
    tmp_indices_trees = [f"tmp_idx{i}" for i in range(t)]
    tmp_slide_trees   = [f"slide_t{i}" for i in range(t)]

    base = f"""
void predict(
        double *responses, double *events, 
        int *features, double *thresholds, double *values, 
        int n_rows, int n_cols, int max_depth, int n_outputs, int num_threads
    ){{
    double tmp_sum; 
    int n_idx;
    int l_idx;
    int r_idx;
    
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for(int n=0; n<n_rows; n++){{
        n_idx=n*n_cols;
        r_idx=n*n_outputs;
        double* events_head = &events[n_idx];
        l_idx = 0;
{KAIGYO.join([f'        l_idx = 2*l_idx + 1 + (events_head[features[l_idx]] > thresholds[l_idx]);' for dep in range(max_depth) for idx, sld in zip(tmp_indices_trees, tmp_slide_trees)])}
        l_idx = l_idx * n_outputs;
        for(int o=0; o<n_outputs; o++){{
            responses[r_idx+o] = values[l_idx+o];
        }}
    }}
}}
    """
    with open(f"./{NEW_PATH}/sk_predict_multiple_output_single_tree_depth{max_depth:0=2}_tree{t:0=2}.c", "w") as f:
        f.write("#include <omp.h>\n")
        f.write("#include <stdio.h>\n")
        f.write("#include <math.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write(base+"\n")










# # Create Predict Function for Scikit-Learn RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, single output
for max_depth in range(MIN_DEPTH, MAX_DEPTH+1):
    for t in range(MIN_TREE_UNROLL, MAX_TREE_UNROLL+1):
            tmp_indices_trees = [f"l_idx{i}" for i in range(t)]

            base = f"""
void predict(
        double *responses, double *events, 
        int *features, double *thresholds, double *values, 
        int n_rows, int n_cols, int max_depth, int n_trees, int n_outputs, int num_threads
    ){{
    double tmp_sum; 
    int n_idx;
    int {', '.join(tmp_indices_trees)};
    double* events_head;

    omp_set_num_threads(num_threads);
    #pragma omp parallel for private(n_idx, events_head, tmp_sum, {', '.join(tmp_indices_trees)})
    for(int n=0; n<n_rows; n++){{
        n_idx= n*n_cols;
        events_head = &events[n_idx];
        tmp_sum=0;
        for(int t=0; t<n_trees; t+={t}){{
{KAIGYO.join([f"            {idx} = t + {i};" for i, idx in enumerate(tmp_indices_trees)])}
{KAIGYO.join([f'            {idx} = 2*{idx} + n_trees + (events_head[features[{idx}]] > thresholds[{idx}]);' for dep in range(max_depth) for idx in tmp_indices_trees])}
            tmp_sum += {' + '.join([f'values[{l_idx}]' for l_idx in tmp_indices_trees])};
        }}
        responses[n] = tmp_sum;
    }}
}}
        """
            with open(f"./{NEW_PATH}/sk_predict_single_output_ensemble_trees_outer-sample_depth{max_depth:0=2}_tree{t:0=2}.c", "w") as f:
                f.write("#include <omp.h>\n")
                f.write("#include <stdio.h>\n")
                f.write("#include <math.h>\n")
                f.write("#include <stdlib.h>\n")
                f.write(base+"\n")


# # Create Predict Function for Scikit-Learn RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, single output
t = 1
for max_depth in range(MIN_DEPTH, MAX_DEPTH+1):
    tmp_indices_trees = [f"tmp_idx{i}" for i in range(t)]
    tmp_slide_trees   = [f"slide_t{i}" for i in range(t)]

    indent = "            "
    idx_trans_base = f"n_idx = 2*n_idx + 1 + (events_head[f_ptr[n_idx]]>t_ptr[n_idx]);"
    idx_trans = [idx_trans_base] * max_depth
    idx_trans = f"\n{indent}".join(idx_trans)

    base = f"""
void predict(
    double *responses, double *events, 
    int *features, double *thresholds, double *values, 
    int n_rows, int n_cols, int max_depth, int n_trees, int n_outputs, int num_threads){{
    
    double *events_head;
    int *f_ptr;
    double *t_ptr;
    double *v_ptr;
    int t_idx, n_nodes, n_idx;
    double tmp_res[n_rows];

    n_nodes = pow(2, max_depth+1)-1;
    omp_set_num_threads(num_threads);
    #pragma omp parallel for private(events_head, t_idx, f_ptr, t_ptr, v_ptr, n_idx)
    for(int t=0; t<n_trees; t++){{

        t_idx = t*n_nodes;
        f_ptr = &features[t_idx];
        t_ptr = &thresholds[t_idx];
        v_ptr = &values[t_idx];
        for(int n=0; n<n_rows; n++){{
            events_head = &events[n*n_cols];
            n_idx = 0;
            {idx_trans}
            #pragma omp atomic
            responses[n] += v_ptr[n_idx];
        }}
    }}
}}"""
    with open(f"./{NEW_PATH}/sk_predict_single_output_ensemble_trees_outer-tree_depth{max_depth:0=2}_tree{t:0=2}.c", "w") as f:
        f.write("#include <omp.h>\n")
        f.write("#include <stdio.h>\n")
        f.write("#include <math.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write(base+"\n")





# # Create Predict Function for Scikit-Learn RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, multiple output (multiclass classification, class probabilities)
for max_depth in range(MIN_DEPTH, MAX_DEPTH+1):
    for t in range(MIN_TREE_UNROLL, MAX_TREE_UNROLL+1):
        tmp_indices_trees = [f"l_idx{i}" for i in range(t)]
        base = f"""
void predict(
        double *responses, double *events, 
        int *features, double *thresholds, double *values, 
        int n_rows, int n_cols, int max_depth, int n_trees, int n_outputs, int num_threads
    ){{
    double tmp_sum; 
    int n_idx;
    int r_idx;
    int {', '.join(tmp_indices_trees)};
    double tmp_sums[n_outputs];

    omp_set_num_threads(num_threads);
    #pragma omp parallel for private(n_idx, r_idx, {', '.join(tmp_indices_trees)}, tmp_sums)
    for(int n=0; n<n_rows; n++){{
        n_idx= n*n_cols;
        r_idx= n*n_outputs;
        
        double* events_head = &events[n_idx];
        for(int o=0; o<n_outputs; o++){{
            tmp_sums[o] = 0;
        }}
        for(int t=0; t<n_trees; t+={t}){{
{KAIGYO.join([f"                {idx} = t + {i};" for i, idx in enumerate(tmp_indices_trees)])}
{KAIGYO.join([f'                {idx} = 2*{idx} + n_trees + (events_head[features[{idx}]] > thresholds[{idx}]);' for dep in range(max_depth) for idx in tmp_indices_trees])}
{KAIGYO.join([f"                {idx} = {idx} * n_outputs;" for i, idx in enumerate(tmp_indices_trees)])}
        for(int o=0; o<n_outputs; o++){{
            tmp_sums[o] += {' + '.join([f'values[{l_idx}+o]' for l_idx in tmp_indices_trees])};
        }}
        }}
        for(int o=0; o<n_outputs; o++){{
            responses[r_idx+o] = tmp_sums[o];
        }}
    }}
}}
        """
        with open(f"./{NEW_PATH}/sk_predict_multiple_output_ensemble_trees_outer-sample_depth{max_depth:0=2}_tree{t:0=2}.c", "w") as f:
            f.write("#include <omp.h>\n")
            f.write("#include <stdio.h>\n")
            f.write("#include <math.h>\n")
            f.write("#include <stdlib.h>\n")
            f.write(base+"\n")



t = 1
for max_depth in range(MIN_DEPTH, MAX_DEPTH+1):
    tmp_indices_trees = [f"tmp_idx{i}" for i in range(t)]
    tmp_slide_trees   = [f"slide_t{i}" for i in range(t)]

    indent = "            "
    idx_trans_base = f"l_idx = 2*l_idx + 1 + (events_head[f_ptr[l_idx]]>t_ptr[l_idx]);"
    idx_trans = [idx_trans_base] * max_depth
    idx_trans = f"\n{indent}".join(idx_trans)

    base = f"""
void predict(
    double *responses, double *events, 
    int *features, double *thresholds, double *values, 
    int n_rows, int n_cols, int max_depth, int n_trees, int n_outputs, int num_threads){{
    
    double *events_head;
    int *f_ptr;
    double *t_ptr;
    double *v_ptr;
    int t_idx, n_nodes, l_idx, n_idx;

    n_nodes = pow(2, max_depth+1)-1;
    omp_set_num_threads(num_threads);
    //#pragma omp parallel for private(events_head, t_idx, f_ptr, t_ptr, v_ptr, l_idx, n_idx)
    for(int t=0; t<n_trees; t++){{

        t_idx = t*n_nodes;
        f_ptr = &features[t_idx];
        t_ptr = &thresholds[t_idx];
        v_ptr = &values[t_idx*n_outputs];
        for(int n=0; n<n_rows; n++){{
            n_idx = n * n_outputs;
            events_head = &events[n*n_cols];
            l_idx = 0;
            {idx_trans}
            l_idx *= n_outputs;
            for(int o=0; o<n_outputs; o++){{
                //#pragma omp atomic
                responses[n_idx+o] += v_ptr[l_idx+o];
            }}
        }}
    }}
}}"""
    with open(f"./{NEW_PATH}/sk_predict_multiple_output_ensemble_trees_outer-tree_depth{max_depth:0=2}_tree{t:0=2}.c", "w") as f:
        f.write("#include <omp.h>\n")
        f.write("#include <stdio.h>\n")
        f.write("#include <math.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write(base+"\n")









# Create Predict Function for CatBoost, single output
for max_depth in range(MIN_DEPTH, MAX_DEPTH+1):
    for t in range(MIN_TREE_UNROLL, MAX_TREE_UNROLL+1):
        tmp_indices_trees = [f"tmp_idx{i}" for i in range(t)]
        tmp_slide_trees   = [f"slide_t{i}" for i in range(t)]

        base = f"""
void predict(
        double *responses, double *events, 
        int *features, double *thresholds, double *values, 
        int n_rows, int n_cols, int max_depth, int n_trees, int n_outputs, int num_threads
    ){{
    double tmp_sum; 
    int n_idx;
    int {', '.join(tmp_indices_trees)};
    int slide=pow(2, max_depth);
    int {', '.join(tmp_slide_trees)};
    
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for(int n=0; n<n_rows; n++){{
        n_idx= n*n_cols;
        double* events_head = &events[n_idx];
        tmp_sum = 0;
        for(int t=0; t<n_trees; t+={t}){{
        
{KAIGYO.join([f'            {s} = (t+{i})*max_depth;' for i, s in enumerate(tmp_slide_trees)])}

{KAIGYO.join([f'            {s} = 0;' for s in tmp_indices_trees])}

{KAIGYO.join([f'            {idx} |= (events_head[features[{dep}+{sld}]] > thresholds[{dep}+{sld}]) << {dep};' for dep in range(max_depth) for idx, sld in zip(tmp_indices_trees, tmp_slide_trees)])}

            tmp_sum += {' + '.join([f'values[{idx}+(t+{t})*slide]' for t, idx in enumerate(tmp_indices_trees)])};
        }}        
        responses[n] += tmp_sum;
    }}
}}
    """
        with open(f"./{NEW_PATH}/cb_predict_single_output_ensemble_trees_outer-sample_depth{max_depth:0=2}_tree{t:0=2}.c", "w") as f:
            f.write("#include <omp.h>\n")
            f.write("#include <stdio.h>\n")
            f.write("#include <math.h>\n")
            f.write("#include <stdlib.h>\n")
            f.write(base)

for max_depth in range(MIN_DEPTH, MAX_DEPTH+1):
    for t in range(MIN_TREE_UNROLL, MAX_TREE_UNROLL+1):
        tmp_indices_trees = [f"tmp_idx{i}" for i in range(t)]
        tmp_slide_trees   = [f"slide_t{i}" for i in range(t)]

        base = f"""
void predict(
        double *responses, double *events, 
        int *features, double *thresholds, double *values, 
        int n_rows, int n_cols, int max_depth, int n_trees, int n_outputs, int num_threads
    )_bra
    int n_idx, r_idx;
    int {', '.join(tmp_indices_trees)};
    int slide=pow(2, max_depth) * n_outputs;
    int {', '.join(tmp_slide_trees)};
    double tmp_sums[n_outputs];
    
    omp_set_num_threads(num_threads);
    #pragma omp parallel for private(n_idx, r_idx, tmp_sums)
    for(int n=0; n<n_rows; n++)_bra
        n_idx= n*n_cols;
        r_idx= n*n_outputs;
        double* events_head = &events[n_idx];
        for(int o=0; o<n_outputs; o++)_bra
            tmp_sums[o] = 0;
        _cket

        for(int t=0; t<n_trees; t+={t})_bra
        
{KAIGYO.join([f'            {s} = (t+{i})*max_depth;' for i, s in enumerate(tmp_slide_trees)])}

{KAIGYO.join([f'            {s} = 0;' for s in tmp_indices_trees])}

{KAIGYO.join([f'            {idx} |= (events_head[features[{dep}+{sld}]] > thresholds[{dep}+{sld}]) << {dep};' for dep in range(max_depth) for idx, sld in zip(tmp_indices_trees, tmp_slide_trees)])}
{KAIGYO.join([f'            {idx} = {idx} * n_outputs;' for idx in tmp_indices_trees])}

            for(int o=0; o<n_outputs; o++)_bra
                tmp_sums[o] += {' + '.join([f'values[{idx}+o+(t+{t})*slide]' for t, idx in enumerate(tmp_indices_trees)])};
            _cket
        _cket        
        for(int o=0; o<n_outputs; o++)_bra
            responses[r_idx+o] += tmp_sums[o];
        _cket
    _cket
_cket
    """
        with open(f"./{NEW_PATH}/cb_predict_multiple_output_ensemble_trees_outer-sample_depth{max_depth:0=2}_tree{t:0=2}.c", "w") as f:
            f.write("#include <omp.h>\n")
            f.write("#include <stdio.h>\n")
            f.write("#include <math.h>\n")
            f.write("#include <stdlib.h>\n")
            f.write(base.replace("_bra", "{").replace("_cket", "}")+"\n")


# Create Predict Function for CatBoost, single output
t=1
for max_depth in range(MIN_DEPTH, MAX_DEPTH+1):
    base = f"""
void predict(
        double *responses, double *events, 
        int *features, double *thresholds, double *values, 
        int n_rows, int n_cols, int max_depth, int n_trees, int n_outputs, int num_threads
    ){{

    double* val_ptr;
    int slide_v=pow(2, max_depth);
    int n_idx, l_idx;
    int slide_t;

    omp_set_num_threads(num_threads);
    #pragma omp parallel for private(val_ptr, slide_t, n_idx, l_idx) reduction(+:responses[:n_rows])
    for(int t=0; t<n_trees; t++){{
        val_ptr = &values[slide_v*t];
        slide_t = t * max_depth;
        for(int n=0; n<n_rows; n++){{
            n_idx = n*n_cols;
            double* events_head = &events[n_idx];

            l_idx = 0;
{KAIGYO.join([f'            l_idx |= (events_head[features[{dep}+slide_t]] > thresholds[{dep}+slide_t]) << {dep};' for dep in range(max_depth)])}

            responses[n] += val_ptr[l_idx];
        }}        
    }}
}}
    """
    with open(f"./{NEW_PATH}/cb_predict_single_output_ensemble_trees_outer-tree_depth{max_depth:0=2}_tree{t:0=2}.c", "w") as f:
        f.write("#include <omp.h>\n")
        f.write("#include <stdio.h>\n")
        f.write("#include <math.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write(base)

for max_depth in range(MIN_DEPTH, MAX_DEPTH+1):
    base = f"""
void predict(
        double *responses, double *events, 
        int *features, double *thresholds, double *values, 
        int n_rows, int n_cols, int max_depth, int n_trees, int n_outputs, int num_threads
    ){{

    double* val_ptr;
    int slide_v=pow(2, max_depth) * n_outputs;
    int n_idx, l_idx, r_idx;
    int slide_t;

    omp_set_num_threads(num_threads);
    #pragma omp parallel for private(val_ptr, slide_t, n_idx, l_idx, r_idx) reduction(+:responses[:n_rows*n_outputs])
    for(int t=0; t<n_trees; t++){{
        val_ptr = &values[slide_v*t];
        slide_t = t * max_depth;
        for(int n=0; n<n_rows; n++){{
            n_idx = n*n_cols;
            double* events_head = &events[n_idx];

            l_idx = 0;
{KAIGYO.join([f'            l_idx |= (events_head[features[{dep}+slide_t]] > thresholds[{dep}+slide_t]) << {dep};' for dep in range(max_depth)])}
            l_idx *= n_outputs;

            r_idx = n * n_outputs;
            for(int o=0; o<n_outputs; o++){{
                responses[r_idx+o] += val_ptr[l_idx+o];
            }}
        }}        
    }}
}}
    """
    with open(f"./{NEW_PATH}/cb_predict_multiple_output_ensemble_trees_outer-tree_depth{max_depth:0=2}_tree{t:0=2}.c", "w") as f:
        f.write("#include <omp.h>\n")
        f.write("#include <stdio.h>\n")
        f.write("#include <math.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write(base)
sys.exit("DONE.")
