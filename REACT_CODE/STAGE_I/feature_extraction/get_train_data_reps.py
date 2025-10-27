import os
import subprocess
import time
import GPUtil

# Configuration parameters
start_chunk = 1
end_chunk = 200
script_path = "/home/cwx/icons/icons/single_representation.py"
base_train_file = "/data2/cwx/icons/dataset/download/llava-v1.5-instruct/llava_665k_splits/chunk_{}.json"
model_path = "/data2/cwx/icons/checkpoints/llava_warm_up_lora/"
image_folder = "/data2/cwx/icons/dataset/download/llava-v1.5-instruct"
output_path = "/data2/cwx/icons/output_mean"
min_free_mem_GB = 17
max_gpus = 8

# Running tasks, recording current (chunk_id, process) on each GPU
running_procs = {}

def get_free_gpus(threshold_gb):
    """
    Get available GPUs with free memory above the threshold (in GB).
    """
    available = []
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        if gpu.memoryFree > threshold_gb * 1024:
            available.append(gpu.id)
    return available

def launch_task(gpu_id, chunk_id):
    """
    Launch a single chunk task on the specified GPU.
    """
    train_file = base_train_file.format(chunk_id)
    log_file = os.path.join(output_path, f"chunk_{chunk_id}.log")
    command = [
        "bash", "-c",
        f"CUDA_VISIBLE_DEVICES={gpu_id} python {script_path} "
        f"--train_file {train_file} "
        f"--model_path {model_path} "
        f"--image_folder {image_folder} "
        f"--output_path {output_path} "
        f"> {log_file} 2>&1"
    ]
    proc = subprocess.Popen(command)
    print(f"Launched chunk_{chunk_id} on GPU {gpu_id}")
    running_procs[gpu_id] = (chunk_id, proc)

# Main scheduling loop
next_chunk = start_chunk
while next_chunk <= end_chunk or running_procs:
    # Clean up completed tasks
    completed_gpus = []
    for gpu_id, (chunk_id, proc) in running_procs.items():
        if proc.poll() is not None:  # Subprocess has completed
            print(f"Finished chunk_{chunk_id} on GPU {gpu_id}")
            completed_gpus.append(gpu_id)

    for gpu_id in completed_gpus:
        del running_procs[gpu_id]

    # Get currently idle GPUs (not running any task and with enough memory)
    free_gpus = get_free_gpus(min_free_mem_GB)
    for gpu_id in free_gpus:
        if gpu_id in running_procs:
            continue
        if next_chunk > end_chunk:
            break
        launch_task(gpu_id, next_chunk)
        next_chunk += 1

    time.sleep(10) 
