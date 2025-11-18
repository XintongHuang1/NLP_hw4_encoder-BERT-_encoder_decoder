"""
Error analysis script for comparing two T5 models
Usage: python find_errors.py --best_model exp4_lr3e4 --baseline_model exp1_lr5e5
"""
import argparse
from utils import load_queries_and_records


def analyze_errors(model_name, gt_sql_path, gt_record_path, model_sql_path, model_record_path):
    """Analyze errors for a single model"""
    
    # Load gold & model SQL + records + error messages
    gt_qs, gt_recs, gt_errs = load_queries_and_records(gt_sql_path, gt_record_path)
    md_qs, md_recs, md_errs = load_queries_and_records(model_sql_path, model_record_path)

    assert len(gt_qs) == len(md_qs) == len(gt_recs) == len(md_recs)
    n = len(gt_qs)

    sql_error_indices = []
    record_mismatch_indices = []
    perfect_match_indices = []

    # Fine-grained categories for SQL execution errors
    sql_error_types = {
        "malformed_query_structure": [],     # SQL syntax/parsing errors (e.g., near "900": syntax error)
        "schema_reference_error": [],        # Invalid column/table references (e.g., no such column)
        "column_ambiguity": [],              # Ambiguous column references requiring qualification
        "execution_failure": [],             # Other runtime execution errors
    }

    # Fine-grained categories for record mismatches
    record_error_types = {
        "missing_results": [],               # Model returns empty set when results expected
        "spurious_results": [],              # Model returns results when empty set expected
        "incorrect_result_set": []           # Both non-empty but different result sets
    }

    # Main loop: classify into coarse and fine-grained error types
    for i in range(n):
        err_msg = md_errs[i]

        # A. SQL execution errors (non-empty error message)
        if err_msg is not None and err_msg != "":
            sql_error_indices.append(i)

            msg_lower = err_msg.lower()

            # SQL execution error subtypes
            if "syntax error" in msg_lower:
                sql_error_types["malformed_query_structure"].append(i)
            elif "no such column" in msg_lower or "no such table" in msg_lower:
                sql_error_types["schema_reference_error"].append(i)
            elif "ambiguous column name" in msg_lower:
                sql_error_types["column_ambiguity"].append(i)
            else:
                sql_error_types["execution_failure"].append(i)

            continue  # skip record comparison if SQL execution failed

        # B. SQL executed successfully: compare records
        if set(gt_recs[i]) != set(md_recs[i]):
            record_mismatch_indices.append(i)

            # Record mismatch subtypes
            gold_len = len(gt_recs[i])
            pred_len = len(md_recs[i])

            if pred_len == 0 and gold_len > 0:
                record_error_types["missing_results"].append(i)
            elif pred_len > 0 and gold_len == 0:
                record_error_types["spurious_results"].append(i)
            else:
                record_error_types["incorrect_result_set"].append(i)
        else:
            perfect_match_indices.append(i)

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Total samples: {n}")
    print(f"SQL execution errors: {len(sql_error_indices)} ({len(sql_error_indices)/n*100:.1f}%)")
    print(f"Record mismatches: {len(record_mismatch_indices)} ({len(record_mismatch_indices)/n*100:.1f}%)")
    print(f"Perfect matches: {len(perfect_match_indices)} ({len(perfect_match_indices)/n*100:.1f}%)")
    print()

    # SQL execution error subtype statistics
    print("=== SQL Execution Error Subtypes ===")
    print(f"  Malformed Query Structure: {len(sql_error_types['malformed_query_structure'])}/{n} ({len(sql_error_types['malformed_query_structure'])/n*100:.1f}%)")
    print(f"  Schema Reference Error: {len(sql_error_types['schema_reference_error'])}/{n} ({len(sql_error_types['schema_reference_error'])/n*100:.1f}%)")
    print(f"  Column Ambiguity: {len(sql_error_types['column_ambiguity'])}/{n} ({len(sql_error_types['column_ambiguity'])/n*100:.1f}%)")
    print(f"  Execution Failure: {len(sql_error_types['execution_failure'])}/{n} ({len(sql_error_types['execution_failure'])/n*100:.1f}%)")
    print()

    print("  Example indices for malformed_query_structure:", sql_error_types["malformed_query_structure"][:10])
    print("  Example indices for schema_reference_error:", sql_error_types["schema_reference_error"][:10])
    print("  Example indices for column_ambiguity:", sql_error_types["column_ambiguity"][:10])
    print("  Example indices for execution_failure:", sql_error_types["execution_failure"][:10])
    print()

    # Record mismatch subtype statistics
    print("=== Record Mismatch Subtypes ===")
    print(f"  Missing Results: {len(record_error_types['missing_results'])}/{n} ({len(record_error_types['missing_results'])/n*100:.1f}%)")
    print(f"  Spurious Results: {len(record_error_types['spurious_results'])}/{n} ({len(record_error_types['spurious_results'])/n*100:.1f}%)")
    print(f"  Incorrect Result Set: {len(record_error_types['incorrect_result_set'])}/{n} ({len(record_error_types['incorrect_result_set'])/n*100:.1f}%)")
    print()

    print("  Example indices for missing_results:", record_error_types["missing_results"][:10])
    print("  Example indices for spurious_results:", record_error_types["spurious_results"][:10])
    print("  Example indices for incorrect_result_set:", record_error_types["incorrect_result_set"][:10])
    print()

    return {
        'total': n,
        'sql_errors': sql_error_indices,
        'record_mismatches': record_mismatch_indices,
        'perfect_matches': perfect_match_indices,
        'sql_error_types': sql_error_types,
        'record_error_types': record_error_types,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze errors for T5 models")
    parser.add_argument('--best_model', type=str, default='exp4_lr3e4',
                        help='Name of best model experiment')
    parser.add_argument('--baseline_model', type=str, default='exp1_lr5e5',
                        help='Name of baseline model experiment')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'test'],
                        help='Which split to analyze')
    args = parser.parse_args()

    gt_sql_path = f"data/{args.split}.sql"
    gt_record_path = f"records/ground_truth_{args.split}.pkl"

    # Analyze best model
    print("\n" + "="*60)
    print("BEST MODEL ANALYSIS")
    print("="*60)
    best_model_sql = f"results/t5_ft_{args.best_model}_{args.split}.sql"
    best_model_pkl = f"records/t5_ft_{args.best_model}_{args.split}.pkl"
    best_results = analyze_errors(args.best_model, gt_sql_path, gt_record_path, 
                                   best_model_sql, best_model_pkl)

    # Analyze baseline model
    print("\n" + "="*60)
    print("BASELINE MODEL ANALYSIS")
    print("="*60)
    baseline_model_sql = f"results/t5_ft_{args.baseline_model}_{args.split}.sql"
    baseline_model_pkl = f"records/t5_ft_{args.baseline_model}_{args.split}.pkl"
    baseline_results = analyze_errors(args.baseline_model, gt_sql_path, gt_record_path,
                                      baseline_model_sql, baseline_model_pkl)

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Best model ({args.best_model}):")
    print(f"  Perfect matches: {len(best_results['perfect_matches'])}/{best_results['total']} "
          f"({len(best_results['perfect_matches'])/best_results['total']*100:.1f}%)")
    print(f"  SQL errors: {len(best_results['sql_errors'])}/{best_results['total']} "
          f"({len(best_results['sql_errors'])/best_results['total']*100:.1f}%)")
    print(f"  Record mismatches: {len(best_results['record_mismatches'])}/{best_results['total']} "
          f"({len(best_results['record_mismatches'])/best_results['total']*100:.1f}%)")
    print()
    print(f"Baseline model ({args.baseline_model}):")
    print(f"  Perfect matches: {len(baseline_results['perfect_matches'])}/{baseline_results['total']} "
          f"({len(baseline_results['perfect_matches'])/baseline_results['total']*100:.1f}%)")
    print(f"  SQL errors: {len(baseline_results['sql_errors'])}/{baseline_results['total']} "
          f"({len(baseline_results['sql_errors'])/baseline_results['total']*100:.1f}%)")
    print(f"  Record mismatches: {len(baseline_results['record_mismatches'])}/{baseline_results['total']} "
          f"({len(baseline_results['record_mismatches'])/baseline_results['total']*100:.1f}%)")


if __name__ == "__main__":
    main()

