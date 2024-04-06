common_config = {
    'data_dir': '/home/dev/Documents/new_crnn/crnn-pytorch/bangla_real',
    'img_width': 100,
    'img_height': 32,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
}

train_config = {
    'epochs': 200,
    'train_batch_size': 32,
    'eval_batch_size': 2048,
    'lr': 0.0005,
    'show_interval': 10,
    'valid_interval': 500,
    'save_interval': 500,
    'cpu_workers': 12,
    'reload_checkpoint': '/home/dev/Documents/new_crnn/crnn-pytorch/checkpoints/augmented_dataset/crnn_blur_real__051500_loss0.746896220906769_acc0.8670849971477467.pt',
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': '/home/dev/Documents/new_crnn/crnn-pytorch/checkpoints/bangla_real'
}
train_config.update(common_config)

evaluate_config = {
    'eval_batch_size': 512,
    'cpu_workers': 12,
    'reload_checkpoint': '/home/dev/Documents/new_crnn/crnn-pytorch/checkpoints/us_lpr/best.pt',
    'decode_method': 'beam_search',
    'beam_size': 10,
}
evaluate_config.update(common_config)
