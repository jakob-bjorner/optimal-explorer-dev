| task | Full Context speedup | Last speedup | Belief speedup |
|---|---|---|---|
| correlate2d_full_fill | 255x | 256x | 266x |
| lp_box | 21x | 21x | 22x |
| lp_mdp | 258x | 116x | 210x |
| lti_simulation | 2.7x | 1.5x | 2.8x |
| Communicability | 104x | 105x | 107x |
| Chebyshev_center | 9x | 9x | 8x |
| Battery_scheduling | 105x | 45x | 98x |
| l0_pruning | 1.0x | 1.0x | 1.0x |
Caption:
Full context (oracle memory performance, causes slower optimization, and not applicable in extreme horizon settings due to overlong contexts): keeping all prior ancestor implementations in context along with their speedups.
Last implementation: heuristic memory function used in TTT-discover.
Belief bottleneck: ABBEL style summary generation, has potential to learn arbitrary heuristic memory functions. 
The results presented represent only the sampling portion of TTT-discover representing the “without TTT” ablation of their paper, and were performed with a Qwen3-8b model hosted with VLLM

