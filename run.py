from functions import *

def main():
  print("Type in one of Valid Datasets to Try: cora, citeseer, enzyme:")
  dataset_name = input()
  epoch = 0
  if dataset_name == "cora":
    epoch = 60
  elif dataset_name == "citeseer":
    epoch = 50
  elif dataset_name == "enzyme":
    epoch = 50
  else:
    print("Bad dataset input")
    exit(-1)
  
  generator = (torch.load("generatorGCN_%s_%d" % (dataset_name, epoch)))
  discriminator = (torch.load("discriminatorGCN_%s_%d" % (dataset_name, epoch)))
  node_data, edge_data, test_nodes, test_data = recall_dataset(dataset_name, .6, .2, .2, seed_hop=2)

  ## Set eval params
  number_trials = 100
  local_use = True
  max_seed_size = 10
  neighborhood_depth = 3

  emd = torch.zeros((number_trials, 3))
  mmd = torch.zeros((number_trials, 4))
  ## Iterate over multiple generated graphs and average 
  for trial in range(number_trials):
    benchmark, bench_root = generate_benchmark(node_data, edge_data, test_nodes, test_data, neighborhood_depth, local_task=local_use, max_seed_size=max_seed_size)
    generated = generate_graph(generator, discriminator, benchmark[0].shape[0], test_data, max_seed_size=max_seed_size, root_node=bench_root if local_use else None)
    deg_emd, clu_emd, cen_emd, b1, b2 = calc_emd(generated, benchmark)
    emd[trial, 0] = deg_emd
    emd[trial, 1] = clu_emd
    emd[trial, 2] = cen_emd
    mmd[trial, 0:3] = torch.tensor([compute_mmd([b1[i]], [b2[i]], kernel=gaussian_emd) for i in range(3)])
    mmd[trial, 3] = compute_mmd(b1, b2, kernel=gaussian_emd)
    print("Average EMD:", torch.mean(emd, dim=0))
    print("Average MMD:", torch.mean(mmd, dim=0))
    emd_std_dev = torch.std(emd, dim=0)
    mmd_std_dev = torch.std(mmd, dim=0)
    print(f"STD EMD: {emd_std_dev}")
    print(f"STD MMD: {mmd_std_dev}")


if __name__ == "__main__":
  main()