# Hugging Face: https://huggingface.co/google/bert_uncased_L-2_H-128_A-2
# Paper: https://arxiv.org/pdf/1908.08962

from accelerate import Accelerator, ProfileKwargs
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import copy
import random
import numpy as np
import time
import os, time, socket, platform

model_name = "google/bert_uncased_L-2_H-128_A-2"
seq_length = 128
num_epochs = 2
batch_size = 2
total_examples = 32


# ------- Save the trace ------------------
def trace_handler(profile):
    # Print profiling results
    print("Sort by cpu_time_total:")
    print(profile.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    # Create directory if it does not exist
    os.makedirs("traces", exist_ok=True)
    profile.export_chrome_trace("traces/batch_size_" + str(batch_size) + ".json")

# -------- CPU RSS helpers (best effort) --------
def get_rss_bytes():
    """Current process RSS in bytes (best effort)."""
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        pass
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On Linux ru_maxrss is in KB, on macOS it is bytes.
        if platform.system().lower() == "linux":
            return int(usage.ru_maxrss * 1024)
        return int(usage.ru_maxrss)
    except Exception:
        return None

def format_bytes(n):
    if n is None:
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} B"


# Load google/bert_uncased_L-2_H-128_A-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_ids = torch.randint(0, tokenizer.vocab_size, (total_examples, seq_length))
attention_mask = torch.ones_like(input_ids)
labels = torch.randint(0, 2, (total_examples,))

dataset = TensorDataset(input_ids, attention_mask, labels)

# Define profiling kwargs for CPU activities
profile_kwargs = ProfileKwargs(activities=["cpu"], profile_memory=True, record_shapes=True, with_flops=True, on_trace_ready=trace_handler)

# Initialize the accelerator for CPU
accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])
device = accelerator.device

rank = accelerator.process_index
world = accelerator.num_processes
local_rank = accelerator.local_process_index
host = socket.gethostname()

# ---- track memory baseline ----
cpu_rss_start = get_rss_bytes()
cpu_rss_peak = cpu_rss_start

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# Define loss function and optimizer
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Prepare the model, optimizer, and data loader for CPU execution
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Move model to training mode
model.train()

# Training + profiling
total_seen_local = 0
t0 = time.time()

# Training loop
with accelerator.profile() as prof:
  for epoch in range(num_epochs):
      running_loss = 0.0
      for (input_ids_batch, attention_mask_batch, labels_batch) in dataloader:

          input_ids_batch = input_ids_batch.to(device)
          attention_mask_batch = attention_mask_batch.to(device)
          labels_batch = labels_batch.to(device)

          optimizer.zero_grad()
          outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch).logits
          loss = criterion(outputs, labels_batch)
          accelerator.backward(loss)
          optimizer.step()

          running_loss += loss.item()
          total_seen_local += input_ids_batch.size(0)

          # update cpu "peak" (best effort)
          rss = get_rss_bytes()
          if rss is not None and (cpu_rss_peak is None or rss > cpu_rss_peak):
              cpu_rss_peak = rss

      avg_loss = torch.tensor(running_loss / len(dataloader), device=device)
      avg_loss = accelerator.reduce(avg_loss, reduction="mean").item()
      if accelerator.is_main_process:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}", flush=True)

  accelerator.wait_for_everyone()
  if torch.cuda.is_available():
      torch.cuda.synchronize()
  t1 = time.time()

  # Global processed (no gather_object needed)
  global_seen = int(accelerator.reduce(torch.tensor(total_seen_local, device=device), reduction="sum").item())

  local_throughput = total_seen_local / (t1 - t0) if (t1 - t0) > 0 else float("nan")
  global_throughput = global_seen / (t1 - t0) if (t1 - t0) > 0 else float("nan")

  # ---- final memory snapshot ----
  cpu_rss_end = get_rss_bytes()

  gpu_name = None
  gpu_alloc = gpu_reserved = gpu_peak_alloc = gpu_peak_reserved = None
  if torch.cuda.is_available():
      try:
          gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
      except Exception:
          gpu_name = None
      gpu_alloc = torch.cuda.memory_allocated()
      gpu_reserved = torch.cuda.memory_reserved()
      gpu_peak_alloc = torch.cuda.max_memory_allocated()
      gpu_peak_reserved = torch.cuda.max_memory_reserved()

  if accelerator.is_main_process:
    print(
        f"\nDone. world_size={world} global_examples={global_seen} time={t1-t0:.2f}s "
        f"global_throughput={global_throughput:.1f} examples/s\n",
        flush=True
    )

  # ---- Print each rank's profiler + memory (serialized) ----
  accelerator.wait_for_everyone()
  for r in range(world):
      if rank == r:
          print("=" * 70, flush=True)
          print(
              f"[RANK {rank}/{world}] host={host} local_rank={local_rank} device={device} gpu={gpu_name}\n"
              f"  local_examples={total_seen_local} local_throughput={local_throughput:.1f} examples/s\n"
              f"  CPU RSS: start={format_bytes(cpu_rss_start)}  end={format_bytes(cpu_rss_end)}  peak~={format_bytes(cpu_rss_peak)}\n"
              f"  GPU mem: alloc={format_bytes(gpu_alloc)}  reserved={format_bytes(gpu_reserved)}  "
              f"peak_alloc={format_bytes(gpu_peak_alloc)}  peak_reserved={format_bytes(gpu_peak_reserved)}",
              flush=True
          )
          print("=" * 70, flush=True)

      accelerator.wait_for_everyone()
  accelerator.end_training()