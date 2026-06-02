# GPU-only vs GPU-CPU Heter Throughput

This folder contains the prompt/output throughput comparison requested on
2026-05-28.

Workloads:

- Prompt lengths: `2000, 4000, 8000`
- Output lengths: `200, 400, 600`
- Batch size: `1`

Policy:

- Prefix caching is not disabled.
- Each workload is measured with a fresh vLLM server and one measured request.
- The PDF uses English text only.
