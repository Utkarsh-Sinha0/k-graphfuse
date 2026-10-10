# K GraphFuse detailed benchmark results

This report includes verified downloaded records only. Running and queued cases without measurements are not counted as results. A partial worker checkpoint is labeled separately from a complete suite.

Complete suites: **15**; partial checkpoints: **0**. Case records: **111**. Candidate configurations: **2190**.

Predeclared ranking/filter cases recorded: **104/104**. Accuracy-accepted K GPU comparisons against the fastest tested overall control: **33/85** pass the combined 10% median, confidence and p95 bar. Cases without an accepted K confirmation: **4**.

## Execution state

| Suite | State | Verified cases |
|---|---|---:|
| directed-fixed | complete | 12 |
| convergence | complete | 14 |
| precision-weights | complete | 15 |
| hard-graphs | complete | 15 |
| public-snap | complete | 12 |
| graphalytics | complete | 2 |
| scale-density | complete | 7 |
| locality | complete | 6 |
| dynamic-rebuild | complete | 2 |
| file-workflows | complete | 2 |
| matrix-filters | complete | 6 |
| sensitivity | complete | 5 |
| memory-pressure | complete | 4 |
| other-calculations | complete | 5 |
| independent-repeat | complete | 4 |

## Per-suite outcomes

| Suite | K confirmed cases | Overall 10% bar | GPU-only 10% bar | No accepted K confirmation |
|---|---:|---:|---:|---:|
| scale-density | 7 | 2/7 | 3/7 | 0 |
| public-snap | 10 | 5/10 | 5/10 | 2 |
| precision-weights | 13 | 3/13 | 3/13 | 2 |
| locality | 6 | 3/6 | 3/6 | 0 |
| independent-repeat | 4 | 1/4 | 1/4 | 0 |
| hard-graphs | 15 | 9/15 | 9/15 | 0 |
| graphalytics | 2 | 0/2 | 0/2 | 0 |
| file-workflows | 1 | 1/1 | 1/1 | 0 |
| dynamic-rebuild | 1 | 1/1 | 1/1 | 0 |
| directed-fixed | 12 | 4/12 | 8/12 | 0 |
| convergence | 14 | 4/14 | 4/14 | 0 |

## Confirmed comparisons

Five confirmation blocks of 31 samples per selected backend; pilot samples are excluded. Ratio above 1 means lower K GraphFuse compute time. Only accuracy-accepted configurations appear.

CPU timings appear where the predeclared CPU size limit permits them. The strongest tested overall control and strongest GPU control are separate CSV fields; missing CPU measurements do not establish a GPU win over CPU.

| Case | K median ms | Comparator | Median ms | Ratio | Paired block 95% interval | 10% gate |
|---|---:|---|---:|---:|---|---|
| Scale n=1000 degree=5 | 0.5159 | CuPy full polynomial control | 1.4030 | 2.720× | 2.329–3.102 | Pass |
| Scale n=1000 degree=5 | 0.5159 | Native cuGraph | 1.4208 | 2.754× | 2.389–2.831 | Pass |
| Scale n=1000 degree=5 | 0.5159 | SciPy adaptive/finite reference control | 0.3163 | 0.613× | 0.511–0.676 | Does not pass |
| Scale n=10000 degree=5 | 0.5830 | CuPy full polynomial control | 1.3777 | 2.363× | 2.339–2.426 | Pass |
| Scale n=10000 degree=5 | 0.5830 | Native cuGraph | 1.5150 | 2.598× | 2.552–2.710 | Pass |
| Scale n=10000 degree=5 | 0.5830 | SciPy adaptive/finite reference control | 1.9974 | 3.426× | 3.306–3.506 | Pass |
| Scale n=100000 degree=5 | 1.2757 | CuPy full polynomial control | 1.4872 | 1.166× | 1.146–1.186 | Pass |
| Scale n=100000 degree=5 | 1.2757 | Native cuGraph | 2.3395 | 1.834× | 1.818–1.843 | Pass |
| Scale n=100000 degree=5 | 1.2757 | SciPy adaptive/finite reference control | 20.7936 | 16.300× | 15.892–16.448 | Pass |
| Scale n=1000000 degree=5 | 8.2602 | CuPy full polynomial control | 7.2130 | 0.873× | 0.868–0.879 | Does not pass |
| Scale n=1000000 degree=5 | 8.2602 | Native cuGraph | 19.8654 | 2.405× | 2.401–2.409 | Pass |
| Scale n=5000000 degree=5 | 121.7658 | CuPy full polynomial control | 121.3675 | 0.997× | 0.934–1.071 | Does not pass |
| Scale n=5000000 degree=5 | 121.7658 | Native cuGraph | 175.4260 | 1.441× | 1.352–1.448 | Pass |
| Scale n=100000 degree=64 | 6.5584 | CuPy full polynomial control | 7.4029 | 1.129× | 1.082–1.255 | Pass |
| Scale n=100000 degree=64 | 6.5584 | Native cuGraph | 5.8961 | 0.899× | 0.877–0.990 | Does not pass |
| Scale n=100000 degree=64 | 6.5584 | SciPy adaptive/finite reference control | 189.3156 | 28.866× | 28.186–31.666 | Pass |
| Scale n=500000 degree=32 | 16.4219 | CuPy full polynomial control | 16.6372 | 1.013× | 0.974–1.034 | Does not pass |
| Scale n=500000 degree=32 | 16.4219 | Native cuGraph | 23.5375 | 1.433× | 1.392–1.461 | Pass |
| Public ca-GrQc ten-step global | 0.5558 | CuPy full polynomial control | 1.3923 | 2.505× | 2.431–2.710 | Pass |
| Public ca-GrQc ten-step global | 0.5558 | Native cuGraph | 1.8208 | 3.276× | 3.238–3.290 | Pass |
| Public ca-GrQc ten-step global | 0.5558 | SciPy adaptive/finite reference control | 1.1548 | 2.078× | 2.020–2.203 | Pass |
| Public ca-GrQc converged personalized | 1.5181 | CuPy full polynomial control | 12.0928 | 7.966× | 4.472–8.063 | Pass |
| Public ca-GrQc converged personalized | 1.5181 | Native cuGraph | 9.5269 | 6.275× | 3.652–6.733 | Pass |
| Public facebook_combined ten-step global | 0.6169 | CuPy full polynomial control | 1.3661 | 2.215× | 2.198–2.227 | Pass |
| Public facebook_combined ten-step global | 0.6169 | Native cuGraph | 1.6997 | 2.755× | 2.739–2.777 | Pass |
| Public facebook_combined ten-step global | 0.6169 | SciPy adaptive/finite reference control | 2.6888 | 4.359× | 4.320–4.391 | Pass |
| Public facebook_combined converged personalized | 2.1726 | CuPy Chebyshev control | 7.3339 | 3.376× | 3.359–6.309 | Pass |
| Public facebook_combined converged personalized | 2.1726 | Native cuGraph | 11.6784 | 5.375× | 5.257–9.981 | Pass |
| Public facebook_combined converged personalized | 2.1726 | SciPy adaptive/finite reference control | 20.7503 | 9.551× | 9.329–17.953 | Pass |
| Public web-Google ten-step global | 7.5477 | CuPy full polynomial control | 5.7225 | 0.758× | 0.756–0.860 | Does not pass |
| Public web-Google ten-step global | 7.5477 | Native cuGraph | 14.4534 | 1.915× | 1.913–2.168 | Pass |
| Public com-dblp.ungraph ten-step global | 2.0049 | CuPy full polynomial control | 1.8580 | 0.927× | 0.852–1.008 | Does not pass |
| Public com-dblp.ungraph ten-step global | 2.0049 | Native cuGraph | 5.7261 | 2.856× | 2.586–3.370 | Pass |
| Public com-dblp.ungraph converged personalized | 9.2457 | CuPy Chebyshev control | 10.8678 | 1.175× | 1.039–1.212 | Pass |
| Public com-dblp.ungraph converged personalized | 9.2457 | Native cuGraph | 61.0414 | 6.602× | 5.775–6.792 | Pass |
| Public com-youtube.ungraph ten-step global | 7.1244 | CuPy full polynomial control | 6.4017 | 0.899× | 0.879–0.964 | Does not pass |
| Public com-youtube.ungraph ten-step global | 7.1244 | Native cuGraph | 16.2332 | 2.279× | 2.260–2.380 | Pass |
| Public com-youtube.ungraph converged personalized | 36.1759 | CuPy Chebyshev control | 36.0132 | 0.996× | 0.994–0.996 | Does not pass |
| Public com-youtube.ungraph converged personalized | 36.1759 | Native cuGraph | 183.7571 | 5.080× | 5.074–5.084 | Pass |
| Public roadNet-CA ten-step global | 7.3904 | CuPy full polynomial control | 6.2787 | 0.850× | 0.848–0.853 | Does not pass |
| Public roadNet-CA ten-step global | 7.3904 | Native cuGraph | 20.9847 | 2.839× | 2.836–2.845 | Pass |
| Precision goal=0.0001 weights=ordinary | 4.2157 | CuPy full polynomial control | 9.0925 | 2.157× | 1.975–3.029 | Pass |
| Precision goal=0.0001 weights=ordinary | 4.2157 | Native cuGraph | 2.6370 | 0.626× | 0.535–0.874 | Does not pass |
| Precision goal=0.0001 weights=ordinary | 4.2157 | SciPy adaptive/finite reference control | 22.0978 | 5.242× | 4.833–7.258 | Pass |
| Precision goal=0.0001 weights=dominant-edge | 4.8617 | CuPy full polynomial control | 9.0750 | 1.867× | 1.544–2.664 | Pass |
| Precision goal=0.0001 weights=dominant-edge | 4.8617 | Native cuGraph | 3.5398 | 0.728× | 0.587–1.037 | Does not pass |
| Precision goal=0.0001 weights=dominant-edge | 4.8617 | SciPy adaptive/finite reference control | 31.2834 | 6.435× | 5.306–9.248 | Pass |
| Precision goal=1e-06 weights=ordinary | 3.6505 | CuPy full polynomial control | 12.3179 | 3.374× | 3.144–3.695 | Pass |
| Precision goal=1e-06 weights=ordinary | 3.6505 | Native cuGraph | 2.7099 | 0.742× | 0.717–0.808 | Does not pass |
| Precision goal=1e-06 weights=ordinary | 3.6505 | SciPy adaptive/finite reference control | 25.9075 | 7.097× | 6.609–7.886 | Pass |
| Precision goal=1e-06 weights=dominant-edge | 5.3004 | CuPy full polynomial control | 13.1038 | 2.472× | 2.036–3.398 | Pass |
| Precision goal=1e-06 weights=dominant-edge | 5.3004 | Native cuGraph | 3.9443 | 0.744× | 0.667–1.016 | Does not pass |
| Precision goal=1e-06 weights=dominant-edge | 5.3004 | SciPy adaptive/finite reference control | 39.1145 | 7.379× | 6.203–10.025 | Pass |
| Precision goal=1e-08 weights=ordinary | 6.0337 | CuPy full polynomial control | 16.1651 | 2.679× | 1.870–2.716 | Pass |
| Precision goal=1e-08 weights=ordinary | 6.0337 | Native cuGraph | 4.1008 | 0.680× | 0.605–0.715 | Does not pass |
| Precision goal=1e-08 weights=ordinary | 6.0337 | SciPy adaptive/finite reference control | 31.5202 | 5.224× | 3.610–5.365 | Pass |
| Precision goal=1e-08 weights=wide-range | 6.3092 | CuPy full polynomial control | 16.1536 | 2.560× | 1.904–2.633 | Pass |
| Precision goal=1e-08 weights=wide-range | 6.3092 | Native cuGraph | 14.2093 | 2.252× | 1.523–2.441 | Pass |
| Precision goal=1e-08 weights=wide-range | 6.3092 | SciPy adaptive/finite reference control | 121.7617 | 19.299× | 14.436–19.737 | Pass |
| Precision goal=1e-08 weights=dominant-edge | 6.1821 | CuPy full polynomial control | 16.1446 | 2.611× | 1.434–2.721 | Pass |
| Precision goal=1e-08 weights=dominant-edge | 6.1821 | Native cuGraph | 6.1352 | 0.992× | 0.541–1.419 | Does not pass |
| Precision goal=1e-08 weights=dominant-edge | 6.1821 | SciPy adaptive/finite reference control | 46.0471 | 7.448× | 4.037–7.814 | Pass |
| Precision goal=1e-10 weights=ordinary | 7.3441 | CuPy full polynomial control | 19.6902 | 2.681× | 1.945–2.697 | Pass |
| Precision goal=1e-10 weights=ordinary | 7.3441 | Native cuGraph | 4.5428 | 0.619× | 0.447–0.678 | Does not pass |
| Precision goal=1e-10 weights=ordinary | 7.3441 | SciPy adaptive/finite reference control | 36.3505 | 4.950× | 3.564–5.038 | Pass |
| Precision goal=1e-10 weights=wide-range | 7.4200 | CuPy full polynomial control | 19.7163 | 2.657× | 2.633–2.681 | Pass |
| Precision goal=1e-10 weights=wide-range | 7.4200 | Native cuGraph | 16.8203 | 2.267× | 2.250–2.293 | Pass |
| Precision goal=1e-10 weights=wide-range | 7.4200 | SciPy adaptive/finite reference control | 135.0841 | 18.205× | 17.875–19.033 | Pass |
| Precision goal=1e-10 weights=dominant-edge | 8.3749 | CuPy full polynomial control | 19.8772 | 2.373× | 1.872–2.654 | Pass |
| Precision goal=1e-10 weights=dominant-edge | 8.3749 | Native cuGraph | 7.1447 | 0.853× | 0.595–0.981 | Does not pass |
| Precision goal=1e-10 weights=dominant-edge | 8.3749 | SciPy adaptive/finite reference control | 52.1765 | 6.230× | 4.919–6.909 | Pass |
| Precision goal=1e-12 weights=ordinary | 9.1541 | CuPy full polynomial control | 23.0396 | 2.517× | 2.410–2.569 | Pass |
| Precision goal=1e-12 weights=ordinary | 9.1541 | Native cuGraph | 5.5592 | 0.607× | 0.534–0.813 | Does not pass |
| Precision goal=1e-12 weights=ordinary | 9.1541 | SciPy adaptive/finite reference control | 39.2327 | 4.286× | 4.057–4.409 | Pass |
| Precision goal=1e-12 weights=wide-range | 8.8833 | CuPy full polynomial control | 22.8992 | 2.578× | 2.513–2.616 | Pass |
| Precision goal=1e-12 weights=wide-range | 8.8833 | Native cuGraph | 16.8931 | 1.902× | 1.844–1.991 | Pass |
| Precision goal=1e-12 weights=wide-range | 8.8833 | SciPy adaptive/finite reference control | 154.5119 | 17.394× | 16.946–17.443 | Pass |
| Precision goal=1e-12 weights=dominant-edge | 8.9533 | CuPy full polynomial control | 23.1411 | 2.585× | 2.393–2.620 | Pass |
| Precision goal=1e-12 weights=dominant-edge | 8.9533 | Native cuGraph | 7.8154 | 0.873× | 0.733–1.129 | Does not pass |
| Precision goal=1e-12 weights=dominant-edge | 8.9533 | SciPy adaptive/finite reference control | 57.5319 | 6.426× | 5.983–6.511 | Pass |
| Localized finite propagation n=100000 shape=chain | 0.7742 | CuPy full polynomial control | 1.5556 | 2.009× | 1.970–2.068 | Pass |
| Localized finite propagation n=100000 shape=chain | 0.7742 | Native cuGraph | 3.7975 | 4.905× | 4.775–5.039 | Pass |
| Localized finite propagation n=100000 shape=chain | 0.7742 | SciPy adaptive/finite reference control | 8.7576 | 11.312× | 11.064–11.627 | Pass |
| Localized finite propagation n=100000 shape=hub | 0.6464 | CuPy full polynomial control | 1.5679 | 2.426× | 1.853–2.624 | Pass |
| Localized finite propagation n=100000 shape=hub | 0.6464 | Native cuGraph | 4.9427 | 7.647× | 6.662–9.151 | Pass |
| Localized finite propagation n=100000 shape=hub | 0.6464 | SciPy adaptive/finite reference control | 10.6575 | 16.488× | 12.712–17.755 | Pass |
| Localized finite propagation n=100000 shape=random | 1.2735 | CuPy full polynomial control | 1.4501 | 1.139× | 1.130–1.152 | Pass |
| Localized finite propagation n=100000 shape=random | 1.2735 | Native cuGraph | 2.6108 | 2.050× | 2.046–2.054 | Pass |
| Localized finite propagation n=100000 shape=random | 1.2735 | SciPy adaptive/finite reference control | 21.2848 | 16.713× | 16.363–16.919 | Pass |
| Localized finite propagation n=1000000 shape=chain | 2.9001 | CuPy full polynomial control | 2.5136 | 0.867× | 0.863–0.869 | Does not pass |
| Localized finite propagation n=1000000 shape=chain | 2.9001 | Native cuGraph | 11.6814 | 4.028× | 3.999–4.135 | Pass |
| Localized finite propagation n=1000000 shape=hub | 3.3249 | CuPy full polynomial control | 2.9603 | 0.890× | 0.888–0.891 | Does not pass |
| Localized finite propagation n=1000000 shape=hub | 3.3249 | Native cuGraph | 37.2560 | 11.205× | 11.184–11.265 | Pass |
| Localized finite propagation n=1000000 shape=random | 8.1050 | CuPy full polynomial control | 6.9728 | 0.860× | 0.848–0.892 | Does not pass |
| Localized finite propagation n=1000000 shape=random | 8.1050 | Native cuGraph | 21.3227 | 2.631× | 2.595–2.734 | Pass |
| Independent directed weighted repeat | 1.3025 | CuPy full polynomial control | 1.4125 | 1.084× | 1.065–1.165 | Does not pass |
| Independent directed weighted repeat | 1.3025 | Native cuGraph | 2.5962 | 1.993× | 1.965–2.113 | Pass |
| Independent directed weighted repeat | 1.3025 | SciPy adaptive/finite reference control | 20.9780 | 16.106× | 15.880–17.227 | Pass |
| Independent undirected personalized repeat | 8.4493 | CuPy Chebyshev control | 10.2967 | 1.219× | 1.082–1.556 | Pass |
| Independent undirected personalized repeat | 8.4493 | Native cuGraph | 7.6793 | 0.909× | 0.811–1.150 | Does not pass |
| Independent undirected personalized repeat | 8.4493 | SciPy adaptive/finite reference control | 111.7084 | 13.221× | 11.664–16.833 | Pass |
| Independent strict precision repeat | 7.4327 | CuPy full polynomial control | 19.8579 | 2.672× | 2.353–2.721 | Pass |
| Independent strict precision repeat | 7.4327 | Native cuGraph | 14.9806 | 2.015× | 1.763–2.175 | Pass |
| Independent strict precision repeat | 7.4327 | SciPy adaptive/finite reference control | 147.1037 | 19.791× | 17.448–19.943 | Pass |
| Independent large scale repeat | 8.2125 | CuPy full polynomial control | 7.0166 | 0.854× | 0.851–0.895 | Does not pass |
| Independent large scale repeat | 8.2125 | Native cuGraph | 19.7495 | 2.405× | 2.396–2.515 | Pass |
| Topology chain | 0.7508 | CuPy full polynomial control | 1.4828 | 1.975× | 1.880–2.315 | Pass |
| Topology chain | 0.7508 | Native cuGraph | 3.6935 | 4.919× | 4.725–5.299 | Pass |
| Topology chain | 0.7508 | SciPy adaptive/finite reference control | 8.6808 | 11.562× | 11.271–12.257 | Pass |
| Topology hub | 0.6601 | CuPy full polynomial control | 1.4690 | 2.225× | 1.820–2.557 | Pass |
| Topology hub | 0.6601 | Native cuGraph | 5.4327 | 8.230× | 6.717–8.671 | Pass |
| Topology hub | 0.6601 | SciPy adaptive/finite reference control | 10.3562 | 15.688× | 13.093–17.434 | Pass |
| Topology disconnected | 1.2606 | CuPy full polynomial control | 1.4544 | 1.154× | 1.147–1.185 | Pass |
| Topology disconnected | 1.2606 | Native cuGraph | 2.5853 | 2.051× | 2.033–2.088 | Pass |
| Topology disconnected | 1.2606 | SciPy adaptive/finite reference control | 19.5943 | 15.543× | 15.382–15.671 | Pass |
| Topology dangling | 1.2835 | CuPy full polynomial control | 1.3816 | 1.076× | 1.071–1.090 | Does not pass |
| Topology dangling | 1.2835 | Native cuGraph | 2.5714 | 2.003× | 1.997–2.015 | Pass |
| Topology dangling | 1.2835 | SciPy adaptive/finite reference control | 20.2916 | 15.810× | 15.575–15.907 | Pass |
| Topology isolated | 1.3287 | CuPy full polynomial control | 1.3663 | 1.028× | 1.023–1.033 | Does not pass |
| Topology isolated | 1.3287 | Native cuGraph | 2.5852 | 1.946× | 1.936–1.961 | Pass |
| Topology isolated | 1.3287 | SciPy adaptive/finite reference control | 20.0076 | 15.058× | 15.004–15.127 | Pass |
| Topology empty | 2.7196 | CuPy full polynomial control | 2.3095 | 0.849× | 0.699–0.952 | Does not pass |
| Topology empty | 2.7196 | Native cuGraph | 1.6347 | 0.601× | 0.494–0.672 | Does not pass |
| Topology empty | 2.7196 | SciPy adaptive/finite reference control | 9.2606 | 3.405× | 2.802–3.818 | Pass |
| Topology dense | 1.7754 | CuPy full polynomial control | 1.6787 | 0.946× | 0.945–0.960 | Does not pass |
| Topology dense | 1.7754 | Native cuGraph | 2.1288 | 1.199× | 1.141–1.250 | Pass |
| Topology dense | 1.7754 | SciPy adaptive/finite reference control | 60.3523 | 33.993× | 33.685–35.267 | Pass |
| Dead ends fraction=0.01 personal=False | 1.3134 | CuPy full polynomial control | 1.4155 | 1.078× | 1.066–1.096 | Does not pass |
| Dead ends fraction=0.01 personal=False | 1.3134 | Native cuGraph | 2.2889 | 1.743× | 1.728–1.750 | Pass |
| Dead ends fraction=0.01 personal=False | 1.3134 | SciPy adaptive/finite reference control | 20.2115 | 15.389× | 15.310–15.542 | Pass |
| Dead ends fraction=0.01 personal=True | 1.2798 | CuPy full polynomial control | 1.4266 | 1.115× | 1.095–1.380 | Pass |
| Dead ends fraction=0.01 personal=True | 1.2798 | Native cuGraph | 2.6107 | 2.040× | 2.028–2.048 | Pass |
| Dead ends fraction=0.01 personal=True | 1.2798 | SciPy adaptive/finite reference control | 20.2999 | 15.862× | 15.784–15.991 | Pass |
| Dead ends fraction=0.25 personal=False | 1.1913 | CuPy full polynomial control | 1.4282 | 1.199× | 1.172–1.215 | Pass |
| Dead ends fraction=0.25 personal=False | 1.1913 | Native cuGraph | 2.4305 | 2.040× | 2.034–2.050 | Pass |
| Dead ends fraction=0.25 personal=False | 1.1913 | SciPy adaptive/finite reference control | 19.1902 | 16.108× | 15.928–16.276 | Pass |
| Dead ends fraction=0.25 personal=True | 1.1757 | CuPy full polynomial control | 1.3963 | 1.188× | 1.175–1.207 | Pass |
| Dead ends fraction=0.25 personal=True | 1.1757 | Native cuGraph | 2.9186 | 2.482× | 2.467–2.515 | Pass |
| Dead ends fraction=0.25 personal=True | 1.1757 | SciPy adaptive/finite reference control | 19.5594 | 16.636× | 16.326–17.053 | Pass |
| Dead ends fraction=0.75 personal=False | 0.9487 | CuPy full polynomial control | 1.3870 | 1.462× | 1.452–1.479 | Pass |
| Dead ends fraction=0.75 personal=False | 0.9487 | Native cuGraph | 2.0773 | 2.190× | 2.185–2.195 | Pass |
| Dead ends fraction=0.75 personal=False | 0.9487 | SciPy adaptive/finite reference control | 16.7879 | 17.696× | 17.516–17.958 | Pass |
| Dead ends fraction=0.75 personal=True | 0.9286 | CuPy full polynomial control | 1.5349 | 1.653× | 1.615–1.680 | Pass |
| Dead ends fraction=0.75 personal=True | 0.9286 | Native cuGraph | 1.1451 | 1.233× | 1.227–1.236 | Pass |
| Dead ends fraction=0.75 personal=True | 0.9286 | SciPy adaptive/finite reference control | 16.7864 | 18.076× | 17.897–18.302 | Pass |
| Isolated fraction=0.1 | 1.1968 | CuPy full polynomial control | 1.4090 | 1.177× | 1.170–1.199 | Pass |
| Isolated fraction=0.1 | 1.1968 | Native cuGraph | 2.4942 | 2.084× | 2.073–2.092 | Pass |
| Isolated fraction=0.1 | 1.1968 | SciPy adaptive/finite reference control | 19.4560 | 16.256× | 16.087–16.404 | Pass |
| Isolated fraction=0.75 | 2.6008 | CuPy full polynomial control | 2.3310 | 0.896× | 0.744–1.017 | Does not pass |
| Isolated fraction=0.75 | 2.6008 | Native cuGraph | 2.7082 | 1.041× | 0.801–1.464 | Does not pass |
| Isolated fraction=0.75 | 2.6008 | SciPy adaptive/finite reference control | 11.5016 | 4.422× | 3.321–6.248 | Pass |
| Official Graphalytics datagen-7_5-fb PageRank | 56.0887 | CuPy full polynomial control | 55.9206 | 0.997× | 0.979–1.012 | Does not pass |
| Official Graphalytics datagen-7_5-fb PageRank | 56.0887 | Native cuGraph | 54.6670 | 0.975× | 0.961–0.998 | Does not pass |
| Official Graphalytics graph500-22 PageRank | 435.7409 | CuPy full polynomial control | 435.2873 | 0.999× | 0.999–0.999 | Does not pass |
| Official Graphalytics graph500-22 PageRank | 435.7409 | Native cuGraph | 160.1893 | 0.368× | 0.351–0.389 | Does not pass |
| File workflow warm stage context | 0.6516 | CuPy full polynomial control | 1.4226 | 2.183× | 2.093–2.307 | Pass |
| File workflow warm stage context | 0.6516 | Native cuGraph | 1.6437 | 2.523× | 2.481–2.564 | Pass |
| File workflow warm stage context | 0.6516 | SciPy adaptive/finite reference control | 3.9061 | 5.995× | 5.799–6.509 | Pass |
| Warm baseline before frequent rebuilds | 0.5978 | CuPy full polynomial control | 1.4033 | 2.348× | 2.301–2.398 | Pass |
| Warm baseline before frequent rebuilds | 0.5978 | Native cuGraph | 1.4670 | 2.454× | 2.432–2.473 | Pass |
| Warm baseline before frequent rebuilds | 0.5978 | SciPy adaptive/finite reference control | 2.0611 | 3.448× | 3.375–3.507 | Pass |
| Directed n=1000 weighted=False personalized=False | 0.5143 | CuPy full polynomial control | 1.5980 | 3.107× | 2.823–3.693 | Pass |
| Directed n=1000 weighted=False personalized=False | 0.5143 | Native cuGraph | 1.4086 | 2.739× | 2.716–2.765 | Pass |
| Directed n=1000 weighted=False personalized=False | 0.5143 | SciPy adaptive/finite reference control | 0.3166 | 0.616× | 0.593–0.710 | Does not pass |
| Directed n=1000 weighted=False personalized=True | 0.5079 | CuPy full polynomial control | 1.4555 | 2.866× | 2.771–3.253 | Pass |
| Directed n=1000 weighted=False personalized=True | 0.5079 | Native cuGraph | 1.5176 | 2.988× | 2.893–3.094 | Pass |
| Directed n=1000 weighted=False personalized=True | 0.5079 | SciPy adaptive/finite reference control | 0.3228 | 0.636× | 0.622–0.691 | Does not pass |
| Directed n=1000 weighted=True personalized=False | 0.5081 | CuPy full polynomial control | 1.4330 | 2.820× | 2.792–2.850 | Pass |
| Directed n=1000 weighted=True personalized=False | 0.5081 | Native cuGraph | 1.2833 | 2.526× | 2.490–2.598 | Pass |
| Directed n=1000 weighted=True personalized=False | 0.5081 | SciPy adaptive/finite reference control | 0.3251 | 0.640× | 0.623–0.656 | Does not pass |
| Directed n=1000 weighted=True personalized=True | 0.5409 | CuPy full polynomial control | 1.4793 | 2.735× | 2.300–2.791 | Pass |
| Directed n=1000 weighted=True personalized=True | 0.5409 | Native cuGraph | 1.5049 | 2.782× | 2.309–2.826 | Pass |
| Directed n=1000 weighted=True personalized=True | 0.5409 | SciPy adaptive/finite reference control | 0.3247 | 0.600× | 0.499–0.611 | Does not pass |
| Directed n=100000 weighted=False personalized=False | 1.2782 | CuPy full polynomial control | 1.5380 | 1.203× | 1.172–1.247 | Pass |
| Directed n=100000 weighted=False personalized=False | 1.2782 | Native cuGraph | 2.3413 | 1.832× | 1.798–1.949 | Pass |
| Directed n=100000 weighted=False personalized=False | 1.2782 | SciPy adaptive/finite reference control | 22.4546 | 17.568× | 16.560–19.136 | Pass |
| Directed n=100000 weighted=False personalized=True | 1.2859 | CuPy full polynomial control | 1.4825 | 1.153× | 1.132–1.169 | Pass |
| Directed n=100000 weighted=False personalized=True | 1.2859 | Native cuGraph | 2.6266 | 2.043× | 2.033–2.137 | Pass |
| Directed n=100000 weighted=False personalized=True | 1.2859 | SciPy adaptive/finite reference control | 25.2450 | 19.632× | 16.929–21.781 | Pass |
| Directed n=100000 weighted=True personalized=False | 1.2932 | CuPy full polynomial control | 1.4874 | 1.150× | 1.100–1.162 | Pass |
| Directed n=100000 weighted=True personalized=False | 1.2932 | Native cuGraph | 2.3311 | 1.803× | 1.712–1.822 | Pass |
| Directed n=100000 weighted=True personalized=False | 1.2932 | SciPy adaptive/finite reference control | 21.6660 | 16.754× | 15.346–18.552 | Pass |
| Directed n=100000 weighted=True personalized=True | 1.2794 | CuPy full polynomial control | 1.4740 | 1.152× | 1.104–1.169 | Pass |
| Directed n=100000 weighted=True personalized=True | 1.2794 | Native cuGraph | 2.6177 | 2.046× | 2.034–2.061 | Pass |
| Directed n=100000 weighted=True personalized=True | 1.2794 | SciPy adaptive/finite reference control | 22.7067 | 17.747× | 16.889–19.723 | Pass |
| Directed n=1000000 weighted=False personalized=False | 7.9490 | CuPy full polynomial control | 6.8976 | 0.868× | 0.840–0.889 | Does not pass |
| Directed n=1000000 weighted=False personalized=False | 7.9490 | Native cuGraph | 19.7185 | 2.481× | 2.406–2.522 | Pass |
| Directed n=1000000 weighted=False personalized=True | 8.1704 | CuPy full polynomial control | 6.8694 | 0.841× | 0.838–0.905 | Does not pass |
| Directed n=1000000 weighted=False personalized=True | 8.1704 | Native cuGraph | 21.2850 | 2.605× | 2.593–2.818 | Pass |
| Directed n=1000000 weighted=True personalized=False | 8.2303 | CuPy full polynomial control | 6.9494 | 0.844× | 0.839–0.887 | Does not pass |
| Directed n=1000000 weighted=True personalized=False | 8.2303 | Native cuGraph | 19.8379 | 2.410× | 2.397–2.534 | Pass |
| Directed n=1000000 weighted=True personalized=True | 8.1735 | CuPy full polynomial control | 6.8868 | 0.843× | 0.837–0.890 | Does not pass |
| Directed n=1000000 weighted=True personalized=True | 8.1735 | Native cuGraph | 21.5368 | 2.635× | 2.606–2.770 | Pass |
| Convergence beta=0.5 n=10000 shape=random | 1.4551 | CuPy full polynomial control | 3.8409 | 2.640× | 2.595–2.909 | Pass |
| Convergence beta=0.5 n=10000 shape=random | 1.4551 | Native cuGraph | 2.4608 | 1.691× | 1.631–1.805 | Pass |
| Convergence beta=0.5 n=10000 shape=random | 1.4551 | SciPy adaptive/finite reference control | 3.5404 | 2.433× | 2.314–2.777 | Pass |
| Convergence beta=0.85 n=10000 shape=random | 4.1400 | CuPy full polynomial control | 15.9095 | 3.843× | 3.506–5.581 | Pass |
| Convergence beta=0.85 n=10000 shape=random | 4.1400 | Native cuGraph | 2.9040 | 0.701× | 0.606–1.007 | Does not pass |
| Convergence beta=0.85 n=10000 shape=random | 4.1400 | SciPy adaptive/finite reference control | 5.5265 | 1.335× | 1.213–1.961 | Pass |
| Convergence beta=0.95 n=10000 shape=random | 8.4591 | CuPy full polynomial control | 50.3065 | 5.947× | 5.363–6.658 | Pass |
| Convergence beta=0.95 n=10000 shape=random | 8.4591 | Native cuGraph | 4.5191 | 0.534× | 0.363–0.594 | Does not pass |
| Convergence beta=0.95 n=10000 shape=random | 8.4591 | SciPy adaptive/finite reference control | 6.6627 | 0.788× | 0.712–0.857 | Does not pass |
| Convergence beta=0.5 n=10000 shape=undirected | 1.4288 | CuPy Chebyshev control | 3.4711 | 2.429× | 2.409–2.456 | Pass |
| Convergence beta=0.5 n=10000 shape=undirected | 1.4288 | Native cuGraph | 3.2695 | 2.288× | 2.248–2.303 | Pass |
| Convergence beta=0.5 n=10000 shape=undirected | 1.4288 | SciPy adaptive/finite reference control | 4.7052 | 3.293× | 3.254–3.330 | Pass |
| Convergence beta=0.85 n=10000 shape=undirected | 2.9477 | CuPy Chebyshev control | 7.8966 | 2.679× | 2.646–3.329 | Pass |
| Convergence beta=0.85 n=10000 shape=undirected | 2.9477 | Native cuGraph | 5.0949 | 1.728× | 1.594–2.173 | Pass |
| Convergence beta=0.85 n=10000 shape=undirected | 2.9477 | SciPy adaptive/finite reference control | 8.4257 | 2.858× | 2.812–3.545 | Pass |
| Convergence beta=0.95 n=10000 shape=undirected | 4.3028 | CuPy Chebyshev control | 14.1899 | 3.298× | 2.767–4.403 | Pass |
| Convergence beta=0.95 n=10000 shape=undirected | 4.3028 | Native cuGraph | 4.7147 | 1.096× | 0.899–1.586 | Does not pass |
| Convergence beta=0.95 n=10000 shape=undirected | 4.3028 | SciPy adaptive/finite reference control | 10.4966 | 2.439× | 2.052–3.226 | Pass |
| Convergence beta=0.5 n=100000 shape=random | 4.2916 | CuPy full polynomial control | 3.8962 | 0.908× | 0.904–1.350 | Does not pass |
| Convergence beta=0.5 n=100000 shape=random | 4.2916 | Native cuGraph | 4.6803 | 1.091× | 1.038–1.854 | Does not pass |
| Convergence beta=0.5 n=100000 shape=random | 4.2916 | SciPy adaptive/finite reference control | 39.8901 | 9.295× | 9.268–13.764 | Pass |
| Convergence beta=0.85 n=100000 shape=random | 10.1810 | CuPy full polynomial control | 16.1474 | 1.586× | 1.571–1.597 | Does not pass |
| Convergence beta=0.85 n=100000 shape=random | 10.1810 | Native cuGraph | 6.7386 | 0.662× | 0.515–0.811 | Does not pass |
| Convergence beta=0.85 n=100000 shape=random | 10.1810 | SciPy adaptive/finite reference control | 63.2815 | 6.216× | 6.170–6.250 | Pass |
| Convergence beta=0.95 n=100000 shape=random | 33.0805 | CuPy full polynomial control | 50.4891 | 1.526× | 1.514–1.550 | Pass |
| Convergence beta=0.95 n=100000 shape=random | 33.0805 | Native cuGraph | 5.8333 | 0.176× | 0.173–0.275 | Does not pass |
| Convergence beta=0.95 n=100000 shape=random | 33.0805 | SciPy adaptive/finite reference control | 73.6456 | 2.226× | 2.204–2.255 | Pass |
| Convergence beta=0.5 n=100000 shape=undirected | 4.2007 | CuPy full polynomial control | 4.0011 | 0.952× | 0.632–1.036 | Does not pass |
| Convergence beta=0.5 n=100000 shape=undirected | 4.2007 | Native cuGraph | 5.0406 | 1.200× | 0.807–1.558 | Does not pass |
| Convergence beta=0.5 n=100000 shape=undirected | 4.2007 | SciPy adaptive/finite reference control | 67.9622 | 16.179× | 10.626–17.150 | Pass |
| Convergence beta=0.85 n=100000 shape=undirected | 7.5455 | CuPy Chebyshev control | 9.3223 | 1.235× | 0.840–1.584 | Does not pass |
| Convergence beta=0.85 n=100000 shape=undirected | 7.5455 | Native cuGraph | 8.7263 | 1.156× | 0.829–1.342 | Does not pass |
| Convergence beta=0.85 n=100000 shape=undirected | 7.5455 | SciPy adaptive/finite reference control | 125.4162 | 16.621× | 11.807–18.258 | Pass |
| Convergence beta=0.95 n=100000 shape=undirected | 12.0473 | CuPy Chebyshev control | 15.0740 | 1.251× | 1.229–1.270 | Pass |
| Convergence beta=0.95 n=100000 shape=undirected | 12.0473 | Native cuGraph | 12.1184 | 1.006× | 0.976–1.017 | Does not pass |
| Convergence beta=0.95 n=100000 shape=undirected | 12.0473 | SciPy adaptive/finite reference control | 151.6086 | 12.584× | 12.073–13.005 | Pass |
| Slow restart beta=.99 directed | 35.1977 | CuPy full polynomial control | 255.1337 | 7.249× | 7.200–7.275 | Pass |
| Slow restart beta=.99 directed | 35.1977 | Native cuGraph | 154.0310 | 4.376× | 4.346–4.390 | Pass |
| Slow restart beta=.99 directed | 35.1977 | SciPy adaptive/finite reference control | 69.1736 | 1.965× | 1.935–1.991 | Pass |
| Slow restart beta=.99 undirected | 6.9420 | CuPy Chebyshev control | 32.1498 | 4.631× | 4.145–4.980 | Pass |
| Slow restart beta=.99 undirected | 6.9420 | Native cuGraph | 3.5543 | 0.512× | 0.461–0.763 | Does not pass |
| Slow restart beta=.99 undirected | 6.9420 | SciPy adaptive/finite reference control | 10.1923 | 1.468× | 1.327–1.597 | Does not pass |

## Cold rebuilding diagnostics

These measurements have the separate timing and validation boundaries recorded in their CSV and raw records. They do not inherit the 155-sample ranking confirmation protocol.

| suite | case | method | median_ms | p95_ms | sample_count |
|---|---|---|---|---|---|
| independent-repeat | Independent directed weighted repeat | CuPy full polynomial control | 66.7024 | 69.2125 | 11 |
| independent-repeat | Independent directed weighted repeat | K GraphFuse full polynomial | 166.4086 | 176.2718 | 11 |
| independent-repeat | Independent directed weighted repeat | Native cuGraph | 84.6212 | 93.7455 | 11 |
| hard-graphs | Topology chain | CuPy full polynomial control | 13.2769 | 14.7577 | 11 |
| hard-graphs | Topology chain | K GraphFuse full polynomial | 46.5028 | 48.6340 | 11 |
| hard-graphs | Topology chain | Native cuGraph | 20.2808 | 21.0459 | 11 |
| hard-graphs | Topology hub | CuPy full polynomial control | 25.8386 | 26.6640 | 11 |
| hard-graphs | Topology hub | K GraphFuse full polynomial | 83.1415 | 85.9905 | 11 |
| hard-graphs | Topology hub | Native cuGraph | 46.9511 | 48.8072 | 11 |
| hard-graphs | Topology disconnected | CuPy full polynomial control | 62.0112 | 65.6623 | 11 |
| hard-graphs | Topology disconnected | K GraphFuse full polynomial | 162.3440 | 164.7646 | 11 |
| hard-graphs | Topology disconnected | Native cuGraph | 83.8920 | 87.4501 | 11 |
| hard-graphs | Topology dangling | CuPy full polynomial control | 102.5102 | 106.3427 | 11 |
| hard-graphs | Topology dangling | K GraphFuse full polynomial | 174.3002 | 176.6275 | 11 |
| hard-graphs | Topology dangling | Native cuGraph | 95.0976 | 98.4788 | 11 |
| hard-graphs | Topology isolated | CuPy full polynomial control | 60.3469 | 62.2880 | 11 |
| hard-graphs | Topology isolated | K GraphFuse full polynomial | 155.9551 | 157.2868 | 11 |
| hard-graphs | Topology isolated | Native cuGraph | 81.6481 | 85.2852 | 11 |
| hard-graphs | Topology empty | CuPy full polynomial control | 11.9675 | 12.3513 | 11 |
| hard-graphs | Topology empty | K GraphFuse full polynomial | 27.6872 | 28.2166 | 11 |
| hard-graphs | Topology empty | Native cuGraph | 12.6206 | 13.1496 | 11 |
| hard-graphs | Topology dense | CuPy full polynomial control | 369.4742 | 373.8324 | 11 |
| hard-graphs | Topology dense | K GraphFuse full polynomial | 606.7392 | 624.2076 | 11 |
| hard-graphs | Topology dense | Native cuGraph | 348.4827 | 356.7686 | 11 |
| hard-graphs | Dead ends fraction=0.01 personal=False | CuPy full polynomial control | 95.3138 | 99.3123 | 11 |
| hard-graphs | Dead ends fraction=0.01 personal=False | K GraphFuse full polynomial | 187.7144 | 194.8155 | 11 |
| hard-graphs | Dead ends fraction=0.01 personal=False | Native cuGraph | 73.0328 | 74.4184 | 11 |
| hard-graphs | Dead ends fraction=0.01 personal=True | CuPy full polynomial control | 90.9297 | 94.6295 | 11 |
| hard-graphs | Dead ends fraction=0.01 personal=True | K GraphFuse full polynomial | 171.6899 | 189.1708 | 11 |
| hard-graphs | Dead ends fraction=0.01 personal=True | Native cuGraph | 73.7239 | 74.9476 | 11 |
| hard-graphs | Dead ends fraction=0.25 personal=False | CuPy full polynomial control | 83.2987 | 84.7914 | 11 |
| hard-graphs | Dead ends fraction=0.25 personal=False | K GraphFuse full polynomial | 166.8757 | 178.2751 | 11 |
| hard-graphs | Dead ends fraction=0.25 personal=False | Native cuGraph | 56.8624 | 57.9438 | 11 |
| hard-graphs | Dead ends fraction=0.25 personal=True | CuPy full polynomial control | 79.1658 | 81.3493 | 11 |
| hard-graphs | Dead ends fraction=0.25 personal=True | K GraphFuse full polynomial | 147.1070 | 148.9772 | 11 |
| hard-graphs | Dead ends fraction=0.25 personal=True | Native cuGraph | 60.3445 | 68.2574 | 11 |
| hard-graphs | Dead ends fraction=0.75 personal=False | CuPy full polynomial control | 41.5446 | 42.1609 | 11 |
| hard-graphs | Dead ends fraction=0.75 personal=False | K GraphFuse full polynomial | 81.1717 | 84.0946 | 11 |
| hard-graphs | Dead ends fraction=0.75 personal=False | Native cuGraph | 29.2986 | 31.7474 | 11 |
| hard-graphs | Dead ends fraction=0.75 personal=True | CuPy full polynomial control | 37.8817 | 45.1054 | 11 |
| hard-graphs | Dead ends fraction=0.75 personal=True | K GraphFuse full polynomial | 70.7127 | 73.2472 | 11 |
| hard-graphs | Dead ends fraction=0.75 personal=True | Native cuGraph | 29.0662 | 30.3584 | 11 |
| hard-graphs | Isolated fraction=0.1 | CuPy full polynomial control | 57.9390 | 59.1992 | 11 |
| hard-graphs | Isolated fraction=0.1 | K GraphFuse full polynomial | 127.0023 | 129.4407 | 11 |
| hard-graphs | Isolated fraction=0.1 | Native cuGraph | 66.2044 | 73.4523 | 11 |
| hard-graphs | Isolated fraction=0.75 | CuPy full polynomial control | 23.5142 | 24.0519 | 11 |
| hard-graphs | Isolated fraction=0.75 | K GraphFuse full polynomial | 39.5381 | 41.0155 | 11 |
| hard-graphs | Isolated fraction=0.75 | Native cuGraph | 20.3379 | 21.4440 | 11 |
| file-workflows | File workflow warm stage context | CuPy full polynomial control | 13.3765 | 13.9639 | 11 |
| file-workflows | File workflow warm stage context | K GraphFuse full polynomial | 40.3061 | 43.4706 | 11 |
| file-workflows | File workflow warm stage context | Native cuGraph | 19.5682 | 20.0048 | 11 |
| dynamic-rebuild | Warm baseline before frequent rebuilds | CuPy full polynomial control | 6.3873 | 6.8879 | 11 |
| dynamic-rebuild | Warm baseline before frequent rebuilds | K GraphFuse full polynomial | 24.4151 | 25.3469 | 11 |
| dynamic-rebuild | Warm baseline before frequent rebuilds | Native cuGraph | 11.5083 | 11.7415 | 11 |
| directed-fixed | Directed n=1000 weighted=False personalized=False | CuPy full polynomial control | 4.4876 | 4.7887 | 11 |
| directed-fixed | Directed n=1000 weighted=False personalized=False | K GraphFuse full polynomial | 13.0858 | 14.0219 | 11 |
| directed-fixed | Directed n=1000 weighted=False personalized=False | Native cuGraph | 6.7090 | 6.8016 | 11 |
| directed-fixed | Directed n=1000 weighted=False personalized=True | CuPy full polynomial control | 3.2759 | 3.5576 | 11 |
| directed-fixed | Directed n=1000 weighted=False personalized=True | K GraphFuse full polynomial | 10.3725 | 10.9759 | 11 |
| directed-fixed | Directed n=1000 weighted=False personalized=True | Native cuGraph | 6.3298 | 6.4467 | 11 |
| directed-fixed | Directed n=1000 weighted=True personalized=False | CuPy full polynomial control | 3.3886 | 3.9940 | 11 |
| directed-fixed | Directed n=1000 weighted=True personalized=False | K GraphFuse full polynomial | 10.0351 | 10.2256 | 11 |
| directed-fixed | Directed n=1000 weighted=True personalized=False | Native cuGraph | 5.9921 | 6.1760 | 11 |
| directed-fixed | Directed n=1000 weighted=True personalized=True | CuPy full polynomial control | 3.5064 | 3.6573 | 11 |
| directed-fixed | Directed n=1000 weighted=True personalized=True | K GraphFuse full polynomial | 10.5058 | 11.1355 | 11 |
| directed-fixed | Directed n=1000 weighted=True personalized=True | Native cuGraph | 6.2562 | 6.3947 | 11 |
| directed-fixed | Directed n=100000 weighted=False personalized=False | CuPy full polynomial control | 81.5444 | 92.7472 | 11 |
| directed-fixed | Directed n=100000 weighted=False personalized=False | K GraphFuse full polynomial | 177.4601 | 232.9348 | 11 |
| directed-fixed | Directed n=100000 weighted=False personalized=False | Native cuGraph | 96.0857 | 113.4468 | 11 |
| directed-fixed | Directed n=100000 weighted=False personalized=True | CuPy full polynomial control | 64.3342 | 97.5370 | 11 |
| directed-fixed | Directed n=100000 weighted=False personalized=True | K GraphFuse full polynomial | 165.3369 | 215.5929 | 11 |
| directed-fixed | Directed n=100000 weighted=False personalized=True | Native cuGraph | 83.4837 | 86.4744 | 11 |
| directed-fixed | Directed n=100000 weighted=True personalized=False | CuPy full polynomial control | 68.2181 | 71.5933 | 11 |
| directed-fixed | Directed n=100000 weighted=True personalized=False | K GraphFuse full polynomial | 184.2636 | 222.0143 | 11 |
| directed-fixed | Directed n=100000 weighted=True personalized=False | Native cuGraph | 94.1541 | 104.7527 | 11 |
| directed-fixed | Directed n=100000 weighted=True personalized=True | CuPy full polynomial control | 87.2452 | 100.4301 | 11 |
| directed-fixed | Directed n=100000 weighted=True personalized=True | K GraphFuse full polynomial | 175.5233 | 211.2541 | 11 |
| directed-fixed | Directed n=100000 weighted=True personalized=True | Native cuGraph | 84.9974 | 112.1653 | 11 |
| convergence | Convergence beta=0.5 n=10000 shape=random | CuPy full polynomial control | 8.7744 | 9.2962 | 11 |
| convergence | Convergence beta=0.5 n=10000 shape=random | K GraphFuse full polynomial | 29.4400 | 30.6825 | 11 |
| convergence | Convergence beta=0.5 n=10000 shape=random | Native cuGraph | 11.6737 | 12.4883 | 11 |
| convergence | Convergence beta=0.85 n=10000 shape=random | CuPy full polynomial control | 22.6793 | 37.3176 | 11 |
| convergence | Convergence beta=0.85 n=10000 shape=random | K GraphFuse full polynomial | 92.6764 | 96.1090 | 11 |
| convergence | Convergence beta=0.85 n=10000 shape=random | Native cuGraph | 13.3164 | 14.0131 | 11 |
| convergence | Convergence beta=0.95 n=10000 shape=random | CuPy full polynomial control | 59.2580 | 60.9804 | 11 |
| convergence | Convergence beta=0.95 n=10000 shape=random | K GraphFuse full polynomial | 650.1254 | 691.2326 | 11 |
| convergence | Convergence beta=0.95 n=10000 shape=random | Native cuGraph | 14.7106 | 15.5020 | 11 |
| convergence | Convergence beta=0.5 n=10000 shape=undirected | CuPy Chebyshev control | 995.6381 | 1045.2946 | 11 |
| convergence | Convergence beta=0.5 n=10000 shape=undirected | K GraphFuse Chebyshev | 998.9870 | 1041.0282 | 11 |
| convergence | Convergence beta=0.5 n=10000 shape=undirected | Native cuGraph | 21.2086 | 22.0364 | 11 |
| convergence | Convergence beta=0.85 n=10000 shape=undirected | CuPy Chebyshev control | 986.8676 | 1034.0215 | 11 |
| convergence | Convergence beta=0.85 n=10000 shape=undirected | K GraphFuse Chebyshev | 994.9634 | 1035.2621 | 11 |
| convergence | Convergence beta=0.85 n=10000 shape=undirected | Native cuGraph | 20.9164 | 21.2009 | 11 |
| convergence | Convergence beta=0.95 n=10000 shape=undirected | CuPy Chebyshev control | 1004.9263 | 1045.4949 | 11 |
| convergence | Convergence beta=0.95 n=10000 shape=undirected | K GraphFuse Chebyshev | 1009.2742 | 1038.8555 | 11 |
| convergence | Convergence beta=0.95 n=10000 shape=undirected | Native cuGraph | 23.4076 | 23.9759 | 11 |

## Complete workflow diagnostics

These measurements have the separate timing and validation boundaries recorded in their CSV and raw records. They do not inherit the 155-sample ranking confirmation protocol.

| suite | case | K_median_ms | native_median_ms | ratio |
|---|---|---|---|---|
| file-workflows | File import -> preparation -> ranking -> saved scores | 42.0341 | 23.5522 | 0.5603 |
| dynamic-rebuild | Changing directed weighted graph, rebuild each query | 21.4581 | 12.7374 | 0.5936 |

## Matrix and derivative diagnostics

These measurements have the separate timing and validation boundaries recorded in their CSV and raw records. They do not inherit the 155-sample ranking confirmation protocol.

| suite | case | K_median_ms | control_median_ms | ratio | control |
|---|---|---|---|---|---|
| sensitivity | Affine value and derivative n=10000 q=4 | 2.3013 | 2.3133 | 1.0052 | Optimized older complete-bound control float64 packed=True capture=True |
| sensitivity | Affine value and derivative n=100000 q=4 | 17.4600 | 17.6667 | 1.0118 | Optimized older complete-bound control float64 packed=True capture=True |
| sensitivity | Affine value and derivative n=1000000 q=4 | 168.7558 | 165.6143 | 0.9814 | Optimized older complete-bound control float64 packed=True capture=True |
| sensitivity | Affine value and derivative n=100000 q=16 | 42.2828 | 42.2513 | 0.9993 | Optimized older complete-bound control float64 packed=True capture=True |
| sensitivity | Affine value and derivative n=1000000 q=8 | 238.4440 | 249.9199 | 1.0481 | Optimized older complete-bound control float64 packed=True capture=False |
| memory-pressure | Feature workspace n=1000000 q=16 | 154.4627 | 147.8472 | 0.9572 | Optimized older complete-bound control float64 packed=True capture=True |
| memory-pressure | Feature workspace n=1000000 q=32 | 287.7726 | 289.1202 | 1.0047 | Optimized older complete-bound control float64 packed=True capture=False |
| memory-pressure | Feature workspace n=500000 q=64 | 406.4429 | 399.7816 | 0.9836 | Optimized older complete-bound control float64 packed=False capture=True |
| memory-pressure | Feature workspace n=250000 q=128 | 609.9126 | 608.4595 | 0.9976 | Optimized older complete-bound control float64 packed=True capture=False |
| matrix-filters | Polynomial filter n=10000 q=4 | 0.9849 | 0.9496 | 0.9642 | Optimized older complete-bound control float64 packed=True capture=True |
| matrix-filters | Polynomial filter n=100000 q=8 | 8.9919 | 9.0070 | 1.0017 | Optimized older complete-bound control float64 packed=True capture=False |
| matrix-filters | Polynomial filter n=1000000 q=4 | 86.7137 | 85.7371 | 0.9887 | Optimized older complete-bound control float64 packed=True capture=True |
| matrix-filters | Polynomial filter n=100000 q=16 | 16.5976 | 16.4781 | 0.9928 | Optimized older complete-bound control float64 packed=True capture=True |
| matrix-filters | Polynomial filter n=100000 q=4 | 10.7212 | 11.1294 | 1.0381 | Optimized older complete-bound control float64 packed=True capture=True |
| matrix-filters | Polynomial filter n=100000 q=4 | 1.4636 | 0.9664 | 0.6603 | CuPy/cuSPARSE retained-degree float32 |

## Native-only calculations

These measurements have the separate timing and validation boundaries recorded in their CSV and raw records. They do not inherit the 155-sample ranking confirmation protocol.

| case | median_ms | p95_ms | native_accepted | K_status |
|---|---|---|---|---|
| BFS / fewest hops | 2.6611 | 2.8026 | True | NOT_IMPLEMENTED |
| Shortest weighted routes | 9.6542 | 10.3178 | True | NOT_IMPLEMENTED |
| Connected groups | 14.1054 | 14.6291 | True | NOT_IMPLEMENTED |
| Triangle counting | 3.7285 | 4.2542 | True | NOT_IMPLEMENTED |
| Community detection | 62.1269 | 65.1766 | True | NOT_IMPLEMENTED |

## Interpretation limits

Cases without an accepted K confirmation are retained in case_coverage.csv. They are not quietly omitted from the coverage denominator or awarded a speedup. Source manifests, exact configurations and 29-test evidence were verified; portable raw records and frozen runners are in raw/.

The comparisons test specific semantics, inputs, tolerances, hardware and candidate configurations. They do not establish global SOTA or production savings. Source changes, captured/eager behavior, precision, preparation and final output costs remain explicit in the raw records.

Unsupported CuPy capture configurations are retained as exceptions; this does not prove every possible custom captured CUDA implementation is slower. K GraphFuse has no implementations of the native-only traversal/community/triangle algorithms. Failed setup attempts are not algorithm failures.

The released numerical core is unchanged. The sparse dangling-state adapter is experimental benchmark integration and depends on the personalization distribution. Other graph/source changes require a new prepared plan.

The synthetic generator named chain closes its final edge back to vertex zero: these measurements cover directed cycles, not open-ended paths. Dynamic rebuilding records 31 updates; the frozen raw description saying five is a labeling error.

CPU/local backends, complete rebuilding and file workflows have different timing boundaries from resident GPU computation. Sampled memory/power are not isolated algorithm peaks or integrated query energy. Customer quality and cloud bills remain unmeasured.
