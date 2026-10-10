---
description: How to interpret K GraphFuse latency reductions, speedup ratios, derived serial rates, accuracy and benchmark win percentages without confusing their units or baselines.
---
# Metric definitions and percentage claims

A benchmark claim needs a named comparator, matching task, measurement scope and unit. This project presents **less compute time** and **higher derived serial compute rate** together. They describe the same latency result; they are not independent measurements or evidence of concurrent service throughput.

## Time, ratio and derived rate

Let \(T_C>0\) be the control's median resident compute time and \(T_K>0\) the K GraphFuse median for the same task. Define

\[
s=\frac{T_C}{T_K},\qquad
\text{time reduction}=100\left(1-\frac{T_K}{T_C}\right)\%,\qquad
\text{derived rate increase}=100(s-1)\%.
\]

The derived rate is proportional to the reciprocal of latency when identical calls execute serially under the timed contract. It does not measure transactions per second, requests under load, FLOPS, utilization or end-to-end customer capacity. Preparation and host transfers excluded from a resident timing may dominate a real service.

| Median speedup ratio | Less time | Higher derived serial compute rate | Rate as a percentage of baseline |
|---|---:|---:|---:|
| 1.10× | 9.09% | 10.00% | 110.00% |
| 1.21× | 17.36% | 21.00% | 121.00% |
| 2.00× | 50.00% | 100.00% | 200.00% |
| 2.75× | 63.64% | 175.00% | 275.00% |
| 3.73946× | 73.26% | 273.95% | 373.95% |

**1.21× means a 21% derived rate increase.** “121% of baseline” includes the original 100%; saying “121% better” would add it again. Similarly, “73% less time” and “274% higher derived rate” can both describe one result. Neither should be labeled simply “274% better,” which leaves the metric undefined.

The strongest selected precision case records 4.334326 ms for its FP64 CuPy control and 1.159077 ms for K GraphFuse: **73.26% less median compute time**, equivalent to **273.95% higher derived serial compute rate**. That is a constructed 100k-vertex, four-channel T4 case. It is not a native cuGraph comparison. The [precision results](precision-gpu.md) retain all six cases, absolute medians, p95, selection policy and accuracy limits.

## Case coverage is a different percentage

The native comparison records 59 strong speed-gate wins among 89 attempted cases with an accepted native control: **66.3% case coverage**. There are 85 comparable cases and four K accuracy failures, retained as non-wins. Requiring both observed reference-error measures to be no worse leaves 26/89, or **29.2%**.

Those percentages say how often a gate passed. They do not say a winning case ran 66.3% faster. The [native comparison](cugraph.md) provides the case rows, same-precision breakdown, strongest tested controls and losses. The separate six-case precision study is not added to that denominator.

## Accuracy and resources keep their own units

- An error reduction is \(100(1-e_K/e_C)\%\) when \(e_C>0\), for one named error measure against one named reference. If both errors are zero, report a tie; if only the control error is zero, a relative reduction is undefined.
- Passing an accuracy tolerance is an acceptance result. It does not mean zero numerical error or better downstream fraud detection, ranking or prediction.
- A lower sampled memory peak is a resource observation under a specified sampling method. It does not establish a lower continuous peak, lower energy, a cheaper machine or a smaller cloud bill.
- Absolute milliseconds, error norms, bytes and comparator precision remain visible alongside percentage summaries. Losses remain in the same evidence record.

## Confidence intervals and scope

The precision point estimate is the ratio of pooled medians. Its recorded bootstrap interval estimates paired ratios of **median block medians**, a different estimator; the interval need not contain the pooled point estimate. Both are from one run and follow profile exploration. The figures transform interval endpoints monotonically into time and derived-rate percentages; they do not create a new confidence statement or remove selection bias.

Use the same clearly labeled conversion for every case. Select a primary metric because it matches the user's decision, and retain the paired unit and absolute measurements. Choosing whichever percentage looks larger without its definition would conceal the baseline.

Read the [measurement methods](methodology.md), [research manuscript](../research/paper.md) and [limitations](../reference/limitations.md) before reusing a claim. Numeric figure inputs and source hashes are in [presentation_metrics.json](../assets/presentation_metrics.json); the [offline audit](reproduce.md) checks them against the recorded measurements.
