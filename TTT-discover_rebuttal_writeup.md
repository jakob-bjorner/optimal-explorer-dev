

To further highlight the potential impact of our contributions, we present here, and will add to our appendix, an analysis of their applicability to the recent TTT-discover style automatic program improvement setting. In this setting, a model is asked to improve the performance of a particular measurable quantity, in our case the speed of a python function (problems, prompts, and measuring harness adapted from AlgoTune), through a structured tree search. Nodes are represented by prior solutions and their ancestors are the history of solutions. What makes this instantiation of test time training particularly applicable to belief bottlenecks is the need to train after each sampling process for improved program generation. This means the typical KV cache available for in context exploration becomes stale and must be recomputed, putting additional pressure against full context beyond the already existing context limits. In TTT-discover, the authors employed a heuristic memory function of taking the last ancestor’s implementation for context. To analyze the performance characteristics of this heuristic history function, and identify gaps in the model’s ability to create and condition on beliefs/summaries, we present performance characteristics across three settings: 
Full context (oracle memory performance, causes slower optimization, and not applicable in extreme horizon settings due to overlong contexts): keeping all prior ancestor implementations in context along with their speedups.
Last implementation: heuristic memory function used in TTT-discover.
Belief bottleneck: ABBEL style summary generation, has potential to learn arbitrary heuristic memory functions. 
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

We present results for all environments attempted. The model used is Qwen3-8b with huggingface’s recommended sampling parameters, operated through VLLM’s local model serving. Speedup differences under 10% should be considered insignificant.

The results presented represent only the sampling portion of TTT-discover representing the “without TTT” ablation of their paper, and were performed with a Qwen3-8b model hosted with VLLM. This was done for computational reasons, their training experiments for each problem cost ~$500, which is out of reach for us. Full context models operate at the frontier of performance, the last and belief history functions are in some environments on par with full context speedup, but in others they fall behind. While we are almost certain some environments exist where last speedup is superior to belief, none tested surpassed belief bottleneck based history. In such environments the knowledge of a heuristic function which performs well could help you shape the belief generating function through our paper’s proposed belief grading.

We note that these takeaways are problem and model dependent. In communications with the authors of TTT-discover they said that in their environments including the full history yielded little benefit.

Additionally instead of having 8 nodes sampled 64 times per step for 50 steps we limit node expansion to 4 nodes sampled 4 times for 10 steps. A further investigation into the application of belief grading for automated python program optimization presents an exciting avenue for future work, given our experimental results demonstrating faster convergence brought on by the belief shaping. Qualitatively, beliefs generated under Qwen3-8b resembled attempts at solving the problem. This is inline with others who report difficulty steering Qwen3 reasoning models to do tasks different from direct problem solving.
