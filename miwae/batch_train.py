import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='animal',
                        help='dataset [default: handwritten0]')
    parser.add_argument('--gpu', type=int, default='0',
                        help='gpu id [default: 0]')
    args = parser.parse_args()
    
    #seed_list = [2022,1127,4217,1895,1016]
    seed_list = [2022,1127,4217]
    beta_list = [0,10,20,50,75]
    
    for seed in seed_list:
        for beta in beta_list:
            cmd = "CUDA_VISIBLE_DEVICES={} python train.py --seed {} --beta {} --dataset {}"\
                .format(args.gpu, seed, beta, args.dataset)
            print(cmd)
            os.system(cmd)