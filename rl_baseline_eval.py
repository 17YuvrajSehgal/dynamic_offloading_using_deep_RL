# # rl_baseline_eval.py
#
# import os
# import sys
#
# import numpy as np
# import pandas as pd
# import torch
# from matplotlib import pyplot as plt
#
# from EnvConfig import EnvConfig
# from ac_agent import ActorCriticAgent
# from train_rl import make_env, print_gpu_info  # reuse helpers
#
# def smooth(vals, k: int = 10):
#     """Moving-average smoothing for nicer plots."""
#     vals = np.asarray(vals)
#     if len(vals) < k:
#         return vals
#     kernel = np.ones(k) / k
#     return np.convolve(vals, kernel, mode="valid")
#
# def evaluate_rl_baseline(
#     actor_path: str = "results/actor_offloading.pt",
#     csv_name: str = "rl-agent_metrics.csv",
#     T: int = EnvConfig.TOTAL_TIME_T,
# ):
#     """
#     Evaluate the trained RL actor as another 'baseline' and save metrics
#     in the same CSV format as the other *_metrics.csv files.
#     """
#     print("\n" + "="*60)
#     print("RL AGENT EVALUATION")
#     print("="*60)
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"[rl_baseline_eval] Using device: {device}")
#
#     if device == "cuda":
#         print(f"[rl_baseline_eval] GPU: {torch.cuda.get_device_name(0)}")
#         print(f"[rl_baseline_eval] Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
#
#     # Build environment and agent skeleton
#     env = make_env()
#     init_state = env.reset()
#     state_dim = init_state.shape[0]
#     print(f"[rl_baseline_eval] State dimension: {state_dim}")
#     print(f"[rl_baseline_eval] Evaluation timesteps: {T}")
#
#     agent = ActorCriticAgent(
#         state_dim=state_dim,
#         n_actions=3,
#         device=device,
#     )
#
#     # Load trained actor weights
#     if not os.path.exists(actor_path):
#         print(f"ERROR: Actor weights not found at {actor_path}")
#         print("Please train the model first using train_rl.py")
#         sys.exit(1)
#
#     state_dict = torch.load(actor_path, map_location=device)
#     agent.actor.load_state_dict(state_dict)
#     agent.actor.eval()
#     print(f"[rl_baseline_eval] ✅ Loaded actor weights from {actor_path}")
#     print(f"[rl_baseline_eval] Actor device: {next(agent.actor.parameters()).device}")
#
#     # Metric buffers
#     qoe_vals = []
#     lat_vals = []
#     eng_vals = []
#     batt_vals = []
#     offload_vals = []
#
#     print("\nStarting evaluation...")
#     state = env.reset()
#     ue = env.ue  # same UE object used internally
#
#     tasks_processed = 0
#
#     for t in range(T):
#         # Progress indicator every 200 timesteps
#         if (t + 1) % 200 == 0:
#             print(f"  Progress: {t+1}/{T} timesteps ({100*(t+1)/T:.1f}%)")
#             sys.stdout.flush()
#
#         action = agent.select_action(state)  # 0=local, 1=mec, 2=cloud
#
#         # Store battery BEFORE taking action (for QoE calculation)
#         battery_before = ue.battery_j
#
#         next_state, _, _, info = env.step(action)
#
#         # Check if this step processed a task
#         if 'latency' in info and info['latency'] > 0:
#             tasks_processed += 1
#
#         latency = float(info.get("latency", 0.0))
#         energy = float(info.get("energy", 0.0))
#         battery = float(info.get("battery", 0.0))
#         success = bool(info.get("success", False))
#         deadline = float(info.get("deadline", latency if latency > 0 else 1.0))
#
#         # ---- QoE computation: SAME as Paper Equation 18 ----
#         # Successful tasks: QoE = -E_consumed / B_n (current battery)
#         # Failed tasks: QoE = η (FAIL_PENALTY)
#         if battery_before <= 0.0:
#             # dead UE: penalty and large "virtual" latency
#             qoe = EnvConfig.FAIL_PENALTY
#             latency_eff = deadline * 10.0
#         else:
#             if latency > 0:  # Only compute QoE if a task was processed
#                 if success:
#                     # Use battery BEFORE consumption as B_n
#                     qoe = -(energy / battery_before)
#                     latency_eff = latency
#                 else:
#                     qoe = EnvConfig.FAIL_PENALTY
#                     latency_eff = latency
#             else:
#                 # No task processed in this step
#                 qoe = 0.0
#                 latency_eff = 0.0
#
#         # Offload ratio: 1 if MEC/Cloud, 0 if local
#         offload = 1.0 if action in (1, 2) else 0.0
#
#         qoe_vals.append(qoe)
#         lat_vals.append(latency_eff)
#         eng_vals.append(energy)
#         batt_vals.append(battery)
#         offload_vals.append(offload)
#
#         state = next_state
#
#         # Check if episode ended
#         if battery <= 0:
#             print(f"\nWARNING: UE battery depleted at timestep {t+1}")
#             break
#
#     print(f"\n✅ Evaluation complete!")
#     print(f"   Total timesteps: {len(qoe_vals)}")
#     print(f"   Tasks processed: {tasks_processed}")
#
#     # ---- Save to CSV in same format as baselines ----
#     os.makedirs("results", exist_ok=True)
#     df = pd.DataFrame(
#         {
#             "QoE": qoe_vals,
#             "Latency": lat_vals,
#             "Energy": eng_vals,
#             "Battery": batt_vals,
#             "OffloadRatio": offload_vals,
#         }
#     )
#
#     # ---- Create and save plot (no interactive display) ----
#     fig, axs = plt.subplots(2, 2, figsize=(10, 7))
#     axs[0, 0].plot(smooth(df["QoE"]))
#     axs[0, 0].set_title("RL Agent — QoE")
#
#     axs[0, 1].plot(df["Battery"])
#     axs[0, 1].set_title("Battery Level (J)")
#
#     axs[1, 0].plot(smooth(df["Latency"]))
#     axs[1, 0].set_title("Latency (s)")
#
#     axs[1, 1].plot(smooth(df["OffloadRatio"]))
#     axs[1, 1].set_title("Offload Ratio")
#
#     for ax in axs.ravel():
#         ax.grid(True, alpha=0.3)
#         ax.set_xlabel("Timestep")
#
#     fig.suptitle("RL Agent Performance", fontsize=12)
#     plt.tight_layout()
#
#     # Save plot to file instead of showing
#     plot_path = os.path.join("results", "rl_agent_performance.png")
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.close(fig)  # Close the figure to free memory
#     print(f"[rl_baseline_eval] ✅ Saved plot to {plot_path}")
#
#     csv_path = os.path.join("results", csv_name)
#     df.to_csv(csv_path, index=False)
#     print(f"[rl_baseline_eval] ✅ Saved RL metrics to {csv_path}")
#
#     # Print summary statistics
#     print("\n" + "="*60)
#     print("EVALUATION SUMMARY")
#     print("="*60)
#     print(f"Mean QoE: {np.mean(qoe_vals):.6f}")
#     print(f"Mean Latency: {np.mean(lat_vals):.6f} s")
#     print(f"Mean Energy: {np.mean(eng_vals):.3f} J")
#     print(f"Initial Battery: {batt_vals[0]:.2f} J")
#     print(f"Final Battery: {batt_vals[-1]:.2f} J")
#     print(f"Battery Consumed: {batt_vals[0] - batt_vals[-1]:.2f} J")
#     print(f"Mean Offload Ratio: {np.mean(offload_vals):.3f}")
#     print("="*60 + "\n")
#
#     if device == "cuda":
#         print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB\n")
#
# def run_all_with_rl(train_first: bool = False):
#     """
#     Full pipeline:
#       1) (optional) train RL actor
#       2) evaluate RL and save rl-agent_metrics.csv + RL-only plot
#       3) run all classical baselines and combined comparison plot
#     """
#     # Print GPU info at start
#     print_gpu_info()
#
#     if train_first:
#         from train_rl import train
#         print("[run_all_with_rl] Training RL agent...")
#         train()  # you can pass episodes=... here if you want
#
#     print("\n[run_all_with_rl] Evaluating RL agent...")
#     evaluate_rl_baseline()
#
#     print("\n[run_all_with_rl] Running baselines and combined plots...")
#     from main import run_all_baselines_and_plots
#     run_all_baselines_and_plots()
#
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train', action='store_true', help='Train RL agent before evaluation')
#     parser.add_argument('--eval-only', action='store_true', help='Only evaluate RL agent (no baselines)')
#     args = parser.parse_args()
#
#     if args.eval_only:
#         print_gpu_info()
#         evaluate_rl_baseline()
#     else:
#         run_all_with_rl(train_first=args.train)
