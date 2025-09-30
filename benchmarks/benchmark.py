import h5py
import pandas as pd
import duckdb
import json
import time

def main():
    # Prebuilt database with index set up.
    DATABASE_PATH = 'test.db'
    DATASET_PATH = '../../PDX/benchmarks/datasets/downloaded/agnews-mxbai-1024-euclidean.hdf5'
    DATASET_TABLE_NAME = 'mxbai'
    DATASET_DIMS = 1024

    THREAD_COUNT = 0 # 0 = auto
    ENABLE_VSS = True
    QUERY_K = 10

    NUMBER_OF_PROGRESS_UPDATES = 10
    PROFILE_WORKAROUND_OUTPUT_FILE = 'temp_profile_output'
    OUTPUT_FILE = f"results_{DATASET_TABLE_NAME}_k{QUERY_K}_t{THREAD_COUNT}_vss{ENABLE_VSS}.csv"


    with duckdb.connect(DATABASE_PATH) as conn:
        if (ENABLE_VSS):
            conn.execute("install vss;")
            conn.execute("load vss;")
        if (THREAD_COUNT != 0):
            conn.execute(f"PRAGMA threads={THREAD_COUNT};")
        conn.execute("PRAGMA explain_output=optimized_only;")
        conn.execute("SET enable_profiling=json;")
        conn.execute("SET profiling_mode=detailed;")
        # The profiling output is used as a workaround to capture the top-level profiler metrics.
        conn.execute(f"SET profiling_output={PROFILE_WORKAROUND_OUTPUT_FILE};")

        with h5py.File(DATASET_PATH, 'r') as f:
            # Warm up query to force the lazy index to be loaded.
            warmup_query = f"SELECT * FROM {DATASET_TABLE_NAME} ORDER BY array_distance(vec,{[0.0] * DATASET_DIMS}::FLOAT[{DATASET_DIMS}]) LIMIT {QUERY_K};"
            conn.execute(warmup_query).fetchall()

            results = []

            queries = f['test']
            queries_len = len(queries)

            for query_idx, query_vec in enumerate(queries):
                if (query_idx % (queries_len / NUMBER_OF_PROGRESS_UPDATES) == 0):
                    print(f"Query {query_idx} of {queries_len}")

                query = f"SELECT * FROM {DATASET_TABLE_NAME} ORDER BY array_distance(vec,{query_vec.tolist()}::FLOAT[{DATASET_DIMS}]) LIMIT {QUERY_K};"

                # Python E2E query execution time
                start_time = time.monotonic()
                conn.execute(query).fetchall()
                end_time = time.monotonic()

                # Read in the full profiling output contained in the workaround file.
                with open(PROFILE_WORKAROUND_OUTPUT_FILE, 'r') as prof_file:
                    result_json = json.loads(prof_file.read())

                scan_operator = get_final_operator(result_json)
                if scan_operator['operator_name'] != 'SEQ_SCAN ' and scan_operator['operator_name'] != 'HNSW_INDEX_SCAN ':
                    raise Exception(f"Expected SEQ_SCAN or HNSW_INDEX_SCAN operator, got {scan_operator['operator_name']}")

                results.append({
                    'query_idx': query_idx,
                    'e2e_duration': end_time - start_time,
                    'index_scan_duration': scan_operator['operator_timing'],
                    'latency': result_json['latency'],
                    'cpu_time': result_json['cpu_time'],
                })

            avg_e2e_duration = sum(r['e2e_duration'] for r in results) / len(results)
            avg_cpu_time = sum(r['cpu_time'] for r in results) / len(results)
            avg_latency = sum(r['latency'] for r in results) / len(results)
            avg_index_scan_duration = sum(r['index_scan_duration'] for r in results) / len(results)
            percentage_of_e2e = (avg_index_scan_duration / avg_e2e_duration) * 100 if avg_e2e_duration != 0 else 0
            percentage_of_latencys = (avg_index_scan_duration / avg_latency) * 100 if avg_latency != 0 else 0

            print(f"Average index_scan_duration: {avg_index_scan_duration:.6f} seconds")
            print(f"Average cpu_time: {avg_cpu_time:.6f} seconds")
            print(f"Average latency: {avg_latency:.6f} seconds")
            print(f"Average e2e_duration: {avg_e2e_duration:.6f} seconds")
            print(f"Index scan as percentage of total duration: {percentage_of_e2e:.2f}%")
            print(f"Index scan as percentage of total latency: {percentage_of_latencys:.2f}%")

            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

def get_final_operator(result_json):
    while len(result_json['children']) > 0:
        result_json = result_json['children'][0]
    return result_json

if __name__ == '__main__':
    main()
