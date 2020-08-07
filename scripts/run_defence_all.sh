python3 simulated_averaging.py --fraction 0.1 \
--lr 0.02 \
--gamma 0.998 \
--num_nets 1000 \
--fl_round 250 \
--part_nets_per_round 100 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset cifar10 \
--model vgg16 \
--fl_mode fixed-pool \
--attacker_pool_size 100 \
--defense_method krum \
--attack_method blackbox \
--attack_case edge-case \
--model_replacement False \
--project_frequency 10 \
--stddev 0.025 \
--eps 2 \
--adv_lr 0.02 \
--prox_attack False \
--poison_type southwest \
--norm_bound 2 \
--device=cuda:1 \
> logs/southwest_vgg16_blackbox_krum_log 2>&1

python3 simulated_averaging.py --fraction 0.1 \
--lr 0.02 \
--gamma 0.998 \
--num_nets 1000 \
--fl_round 250 \
--part_nets_per_round 100 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset cifar10 \
--model vgg16 \
--fl_mode fixed-pool \
--attacker_pool_size 100 \
--defense_method multi-krum \
--attack_method blackbox \
--attack_case edge-case \
--model_replacement False \
--project_frequency 10 \
--stddev 0.025 \
--eps 2 \
--adv_lr 0.02 \
--prox_attack False \
--poison_type southwest \
--norm_bound 2 \
--device=cuda:1 \
> logs/southwest_vgg16_blackbox_multi_krum_log 2>&1

python3 simulated_averaging.py --fraction 0.1 \
--lr 0.02 \
--gamma 0.998 \
--num_nets 1000 \
--fl_round 250 \
--part_nets_per_round 100 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset cifar10 \
--model vgg16 \
--fl_mode fixed-pool \
--attacker_pool_size 100 \
--defense_method norm-clipping \
--attack_method blackbox \
--attack_case edge-case \
--model_replacement False \
--project_frequency 10 \
--stddev 0.025 \
--eps 2 \
--adv_lr 0.02 \
--prox_attack False \
--poison_type southwest \
--norm_bound 2 \
--device=cuda:1 \
> logs/southwest_vgg16_blackbox_norm_clipping_log 2>&1

python3 simulated_averaging.py --fraction 0.1 \
--lr 0.02 \
--gamma 0.998 \
--num_nets 1000 \
--fl_round 250 \
--part_nets_per_round 100 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset cifar10 \
--model vgg16 \
--fl_mode fixed-pool \
--attacker_pool_size 100 \
--defense_method weak-dp \
--attack_method blackbox \
--attack_case edge-case \
--model_replacement False \
--project_frequency 10 \
--stddev 0.025 \
--eps 2 \
--adv_lr 0.02 \
--prox_attack False \
--poison_type southwest \
--norm_bound 2 \
--device=cuda:1 \
> logs/southwest_vgg16_blackbox_weak_dp_log 2>&1

python3 simulated_averaging.py --fraction 0.1 \
--lr 0.02 \
--gamma 0.998 \
--num_nets 1000 \
--fl_round 250 \
--part_nets_per_round 100 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset cifar10 \
--model vgg16 \
--fl_mode fixed-pool \
--attacker_pool_size 100 \
--defense_method rfa \
--attack_method blackbox \
--attack_case edge-case \
--model_replacement False \
--project_frequency 10 \
--stddev 0.025 \
--eps 2 \
--adv_lr 0.02 \
--prox_attack False \
--poison_type southwest \
--norm_bound 2 \
--device=cuda:1 \
> logs/southwest_vgg16_blackbox_rfa_log 2>&1