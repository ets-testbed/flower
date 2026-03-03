# Important Full Run Commands

## Run Directly (No Copy/Paste Needed)

From:

```powershell
cd C:\pycharm\flower\flower_research_extension
```

List available scenarios:

```powershell
py -m flower_research_extension.experiments.run_commands --list
```

Run one scenario directly:

```powershell
py -m flower_research_extension.experiments.run_commands --scenario mnist_iid
```

Run one scenario as config-check only:

```powershell
py -m flower_research_extension.experiments.run_commands --scenario mnist_iid --dry_run
```

Run all important scenarios:

```powershell
py -m flower_research_extension.experiments.run_commands --all
```

Run all as dry-run only:

```powershell
py -m flower_research_extension.experiments.run_commands --all --dry_run
```

Print commands without executing:

```powershell
py -m flower_research_extension.experiments.run_commands --all --print_only
```

## Scenario Names

- `mnist_iid`
- `mnist_dirichlet`
- `mnist_label_skew`
- `mnist_shard`
- `mnist_inner_dirichlet`
- `mnist_size`
- `mnist_distribution`
- `cifar10_iid`
- `svhn_dirichlet`
- `cifar100_iid`

## Full Command Behind Each Scenario

All scenarios run with these common args:
- `--dataset_root flower_research_extension/data`
- `--num_rounds 10`
- `--num_partitions 10`
- `--fraction_fit 0.25`
- `--min_fit_clients 3`
- `--min_evaluate_clients 3`
- `--batch_size 64`
- `--local_epochs 5`
- `--lr 0.01`
- `--momentum 0.9`
- `--seed 42`
- `--client_cpu 1`
- `--client_gpu 0.01`
- `--csv_log_dir flower_research_extension/results/logs`
- `--wandb_dir flower_research_extension/results/wandb`
- `--wandb_project flower-federated`
- `--wandb_run_name auto`
- `--disable_wandb`

Scenario-specific args:

- `mnist_iid`:
  - `--dataset mnist --model resnet18 --distribution iid`
- `mnist_dirichlet`:
  - `--dataset mnist --model resnet18 --distribution dirichlet --dirichlet_alpha 0.3`
- `mnist_label_skew`:
  - `--dataset mnist --model resnet18 --distribution label_skew --label_skew_classes 2`
- `mnist_shard`:
  - `--dataset mnist --model resnet18 --distribution shard --shard_num_shards_per_partition 2`
- `mnist_inner_dirichlet`:
  - `--dataset mnist --model resnet18 --distribution inner_dirichlet --inner_dirichlet_alpha 0.5 --size_partition_weights 1,1,1,1,1,1,1,1,1,1`
- `mnist_size`:
  - `--dataset mnist --model resnet18 --distribution size --size_partition_weights 1,2,3,4,5,6,7,8,9,10`
- `mnist_distribution`:
  - `--dataset mnist --model resnet18 --distribution distribution --distribution_matrix_json flower_research_extension/experiments/matrix_mnist_10x10.json`
  - matrix file is auto-created by runner if missing
- `cifar10_iid`:
  - `--dataset cifar10 --model resnet18 --distribution iid`
- `svhn_dirichlet`:
  - `--dataset svhn --model mobilenet_v2 --distribution dirichlet --dirichlet_alpha 0.3`
- `cifar100_iid`:
  - `--dataset cifar100 --model densenet121 --distribution iid`
