# Mixed-precision filters for batched progress preservation

* Each bias corresponds to a filter (a kernel in Conv or a column of the weight matrix in Gemm)
* Modify `transform.py` to:
    * Use different quantization factors for different filters
    * For each filter, use different quantization factors for different weight values if
      values in the input vector have different quantization factors
* Update sanity checks for skipping non-quantized filters
