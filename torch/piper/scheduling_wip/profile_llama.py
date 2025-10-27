import torch
import argparse
from collections import defaultdict

from .llama_baseline import LLAMA_DEBUG, LLAMA_3B, Transformer
from .piper_analysis import estimate_model_size, profile_fx_fw_bw
from .piper_scheduling import grid_schedule_to_DAG_schedule, build_1f1b_schedule, print_schedule
from .piper_scheduling import DAG, DAGTask, Edge
from .interleaved_grid_schedules import pp2_interleaved_1f1b_grid_schedule, pp4_interleaved_1f1b_grid_schedule


def main():
    parser = argparse.ArgumentParser(description='Profile and schedule LLaMA model with pipeline parallelism')
    parser.add_argument('--model', choices=['LLAMA_DEBUG', 'LLAMA_3B'], default='LLAMA_DEBUG',
                        help='Model configuration to use (default: LLAMA_DEBUG)')
    parser.add_argument('--split_nodes', type=str, default='float_11',
                        help='Comma-separated list of split nodes (default: float_11)')
    parser.add_argument('--pp', type=int, choices=[2, 4], default=2,
                        help='Pipeline parallel degree (2 or 4). Default: 2')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for profiling (default: 8)')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Sequence length for profiling (default: 512)')
    parser.add_argument('--num_mbs', type=int, default=4,
                        help='Number of microbatches for scheduling (default: 4)')
    parser.add_argument('--schedule', choices=['1f1b', 'interleaved-1f1b'], default='1f1b',
                        help='Schedule type: 1f1b or interleaved-1f1b (default: 1f1b)')
    parser.add_argument('--comm_cost', type=float, default=0.0,
                        help='Communication cost in milliseconds for inter-stage transfers (default: 0.0)')
    
    args = parser.parse_args()
    
    # Set model configuration based on argument
    if args.model == 'LLAMA_DEBUG':
        llama_config = LLAMA_DEBUG
    elif args.model == 'LLAMA_3B':
        llama_config = LLAMA_3B
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Parse split nodes from comma-separated string
    split_nodes = [node.strip() for node in args.split_nodes.split(',') if node.strip()]
    
    seq_len = args.seq_len
    model = Transformer(llama_config, seq_len).to('cuda')

    # --- Trace model and get shapes ---
    batch_size = args.batch_size
    example_input = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len), device='cuda')

    model_size_stats = estimate_model_size(model, example_input)
    all_stats = profile_fx_fw_bw(model, example_input, split_nodes, num_iters=50)

    print("=== Estimate Memory Usage ===")
    for key, value in model_size_stats.items():
        print(f"{key}: {round(value, 1)} GB")
    
    for key, value in model_size_stats.items():
        model_size_stats[key] /= args.pp
    
    print("=== Profile ===")
    peak_mem = defaultdict(list)
    for graph_name, graph_stats in all_stats.items():
        print(f"{graph_name.upper()} GRAPH:")
        for stage_name, stage_stats in graph_stats.items():
            print(f"  {stage_name}:")
            for stat_key, stat_value in stage_stats.items():
                if stat_key == "cuda_time_ms":
                    print(f"    {stat_key}: {stat_value:.2f} ms")
                elif stat_key == "cuda_mem_peak_GB":
                    print(f"    {stat_key}: {stat_value:.2f} GB")
                    peak_mem[graph_name].append(stat_value)
                elif stat_key == "cuda_mem_delta_GB":
                    print(f"    {stat_key}: {stat_value:.2f} GB")
                else:
                    print(f"    {stat_key}: {stat_value}")
    
    for graph_name, graph_peak_mem in peak_mem.items():
        print(f"{graph_name.upper()} GRAPH:")
        print(f"  Peak memory: {max(graph_peak_mem)} GB")

    num_mbs = args.num_mbs
    # for num_mbs in (2, 4, 6, 8, 10, 12):
    comm_cost = args.comm_cost
    path, val = None, None

    # PP-2 1F1B
    if args.pp == 2 and args.schedule == "1f1b":
        grid_schedule = build_1f1b_schedule(num_mbs, args.pp)
        dag_schedule = grid_schedule_to_DAG_schedule(grid_schedule, [Edge(0, 1)], all_stats, comm_cost, num_mbs)
        # print_schedule(grid_schedule)
        path, val = dag_schedule.longest_path("f_0_0_0", f"u_0")

    # PP-2 Interleaved1F1B
    if args.pp == 2 and args.schedule == "interleaved-1f1b":
        grid_schedule = pp2_interleaved_1f1b_grid_schedule
        dag_schedule = grid_schedule_to_DAG_schedule(grid_schedule, [Edge(0, 1), Edge(1, 2), Edge(2, 3)], all_stats, comm_cost, num_mbs)
        # print_schedule(grid_schedule)
        path, val = dag_schedule.longest_path("f_0_0_0", "u_2")

    # PP-4 1F1B
    if args.pp == 4 and args.schedule == "1f1b":
        grid_schedule = build_1f1b_schedule(num_mbs, args.pp)
        dag_schedule = grid_schedule_to_DAG_schedule(grid_schedule, [Edge(0, 1), Edge(1, 2), Edge(2, 3)], all_stats, comm_cost, num_mbs)
        print_schedule(grid_schedule)
        path, val = dag_schedule.longest_path("f_0_0_0", f"u_0")

    # PP-4 Interleaved1F1B
    if args.pp == 4 and args.schedule == "interleaved-1f1b":
        grid_schedule = pp4_interleaved_1f1b_grid_schedule
        dag_schedule = grid_schedule_to_DAG_schedule(grid_schedule, [Edge(0, 1), Edge(1, 2), Edge(2, 3), Edge(3, 4), Edge(4, 5), Edge(5, 6), Edge(6, 7)], all_stats, comm_cost, num_mbs)
        # print_schedule(grid_schedule)
        path, val = dag_schedule.longest_path("f_0_0_0", f"u_4")

    act_mem = dag_schedule.activation_memory(model_size_stats)
    for key, value in act_mem.items():
        baseline_memory = model_size_stats["parameters_GB"] + model_size_stats["gradients_GB"] + model_size_stats["optimizer_GB"]
        print(f"{key} Max activation memory: {round(value, 1)} GB")
        print(f"{key} Max memory: {round(baseline_memory + value, 1)} GB")

    print(f"=========== {num_mbs} microbatches ===========")
    print(f"Schedule path: {path}")
    print(f"Schedule length: {val:.1f}ms")
    print(f"Schedule throughput: {batch_size * seq_len * num_mbs * 1000 / val:.1f} tokens/sec")

if __name__ == "__main__":
    main()