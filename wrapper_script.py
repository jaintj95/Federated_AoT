import simulated_averaging as sim
import argparse
from simulated_averaging import bool_string

def create_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.998, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--fraction', type=float or int, default=10,
                        help='how many fraction of poisoned data inserted')
    parser.add_argument('--local_train_period', type=int, default=1,
                        help='number of local training epochs')
    parser.add_argument('--num_nets', type=int, default=250,
                        help='number of totally available users')
    parser.add_argument('--part_nets_per_round', type=int, default=25,
                        help='number of participating clients per FL round')
    parser.add_argument('--fl_round', type=int, default=250,
                        help='total number of FL round to conduct')
    parser.add_argument('--fl_mode', type=str, default="fixed-pool",
                        help='fl mode: fixed-freq mode or fixed-pool mode')
    parser.add_argument('--attacker_pool_size', type=int, default=25,
                        help='size of attackers in the population, used when args.fl_mode == fixed-pool only')
    parser.add_argument('--defense_method', type=str, default="no-defense",
                        help='defenses: no-defense|norm-clipping|norm-clipping-adaptive|weak-dp|krum|multi-krum|rfa')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='device to set, can take the value of: cuda or cuda:x')
    parser.add_argument('--attack_method', type=str, default="blackbox",
                        help='describe the attack type: blackbox|pgd|graybox|')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use during the training process')
    parser.add_argument('--model', type=str, default='vgg16',
                        help='model to use during the training process')
    parser.add_argument('--eps', type=float, default=5e-5,
                        help='specify the l_inf epsilon budget')
    parser.add_argument('--norm_bound', type=float, default=3,
                        help='describe if there is defense method: no-defense|norm-clipping|weak-dp|')
    parser.add_argument('--adversarial_local_training_period', type=int, default=5,
                        help='specify how many epochs the adversary should train for')
    parser.add_argument('--poison_type', type=str, default='southwest',
                        help='data poisoning source: |ardis|fashion|(for EMNIST) || '
                             '|southwest|southwest+wow|southwest-da|greencar-neo|howto|(for CIFAR-10)')
    parser.add_argument('--rand_seed', type=int, default=7,
                        help='random seed utilize in the experiment for reproducibility.')
    parser.add_argument('--model_replacement', type=bool_string, default=False,
                        help='to scale or not to scale')
    parser.add_argument('--project_frequency', type=int, default=10,
                        help='project once every how many epochs')
    parser.add_argument('--adv_lr', type=float, default=0.02,
                        help='learning rate for adv in PGD setting')
    parser.add_argument('--prox_attack', type=bool_string, default=False,
                        help='use prox attack')
    parser.add_argument('--attack_case', type=str, default="edge-case",
                        help='attack case indicates whether the honest nodes see the attackers poisoned data points: '
                             'edge-case|normal-case|almost-edge-case')
    parser.add_argument('--stddev', type=float, default=0.158,
                        help='choose std_dev for weak-dp defense')
    return parser.parse_args()

def test_number_of_adversary_nodes():
    args = create_args()
    idx = 0
    results_global = {}
    while idx < 10:
        args.attacker_pool_size = (idx + 1) * 10
        results_local = sim.main_func(args)
        results_global[str(args.attacker_pool_size)] = results_local
    return results_global

def test_fraction_of_poisoned_data():
    pass