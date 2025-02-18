import os
import time
import numpy as np
import tvm
import tvm.runtime as rt
from tvm.contrib import graph_executor
import csv
import psutil

CHANGE_DETECTION_HISTORY = 5
model_filepath = './change_detection_large_images_1_10_32_32_graph_cc_q8_A53.so'
input_name = 'images'
csv_file = 'measurements_CD.csv'
input_shape = (1, 10, 32, 32)
repeats = 10000

input_data = np.random.rand(*input_shape).astype(np.float32)

os.environ['TVM_NUM_THREADS'] = "4"
dev = tvm.cpu(0)

loaded_lib = rt.load_module(model_filepath)
module = graph_executor.GraphModule(loaded_lib['default'](dev))
map_inputs = {input_name: input_data}

all_times = []  # List to store runtimes
throughputs = []  # List to store throughput values in Mbps
memory_samples = []
before_vectors = np.random.rand(CHANGE_DETECTION_HISTORY,128)
for i in range(repeats):
    process = psutil.Process(os.getpid())
    start = time.time()
    module.set_input(**map_inputs)
    module.run()
    out = module.get_output(0).numpy().squeeze()
    
    # Normalize the vectors to unit vectors (important for cosine similarity)
    single_vector_norm = out / np.linalg.norm(out)
    three_vectors_norm = before_vectors / np.linalg.norm(before_vectors, axis=1, keepdims=True)

    # Compute cosine similarity
    cosine_similarities = np.dot(three_vectors_norm, single_vector_norm)

    # Convert cosine similarity to cosine distance (1 - similarity)
    cosine_distances = 1 - cosine_similarities

    # Compute mean cosine distance
    mean_distance = np.mean(cosine_distances)
    runtime_s = time.time() - start

    # Take memory samples
    mem_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    memory_samples.append(mem_usage)
    
    # Calculate the runtime in milliseconds
    runtime_ms = runtime_s * 1000  # Convert runtime to milliseconds
    all_times.append(runtime_ms)

    # Calculate throughput in Mbps
    # Number of elements in the input array
    input_size = input_data.size
    # Input size in bits (each will be int16)
    input_bits = input_size * 16
    # Throughput in bits per second
    throughput_bps = input_bits / runtime_s
    # Convert throughput to Mbps
    throughput_mbps = throughput_bps / 1e6
    throughputs.append(throughput_mbps)

    print(
        f"Iteration {i + 1}: Runtime: {runtime_ms:.3f} ms | "
        f"Throughput: {throughput_mbps:.3f} Mbps | "
        f"RAM usage: {mem_usage:.3f} MB"
    )

# Print final statistics
print(f"Median runtime: {np.median(all_times):.3f} ms")
print(f"Mean runtime: {np.mean(all_times):.3f} ms")
print(f"Min runtime: {min(all_times):.3f} ms")
print(f"Max runtime: {max(all_times):.3f} ms")
print(f"Median throughput: {np.median(throughputs):.3f} Mbps")
print(f"Mean throughput: {np.mean(throughputs):.3f} Mbps")
print(f"Min throughput: {min(throughputs):.3f} Mbps")
print(f"Max throughput: {max(throughputs):.3f} Mbps")
print(f"Median RAM: {np.median(memory_samples):.3f} MB")
print(f"Mean RAM: {np.mean(memory_samples):.3f} MB")
print(f"Max RAM: {max(memory_samples):.3f} MB")
print(f"Min RAM: {min(memory_samples):.3f} MB")


# Write results to CSV
with open(csv_file, mode='w', newline='') as file: 
    writer = csv.writer(file)
    # Write headers
    writer.writerow(['Iteration', 'Runtime (ms)', 'Throughput (Mbps)', 'RAM usage (MB)'])
    # Write data for each iteration
    for i in range(repeats):
        writer.writerow([i + 1, all_times[i], throughputs[i], memory_samples[i]])

# Optionally, print the last output for inspection
print(f"Final output (last iteration): {out}")