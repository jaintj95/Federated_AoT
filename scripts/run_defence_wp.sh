python simulated_averaging.py --fraction 0.1 \
--lr 0.02 \
--gamma 0.998 \
--num_nets 200 \
--fl_round 200 \
--part_nets_per_round 10 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset cifar10 \
--model vgg16 \
--fl_mode fixed-freq \
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
> logs/southwest_vgg16_blackbox_no_defense_log 2>&1
