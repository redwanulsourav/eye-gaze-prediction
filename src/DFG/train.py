def train_discriminator(
    discrim_model,
    gen_model,
    latent,
    ground_truth_frames
):
    gen_frames = gen_model(latent)
    label = discrim_model(gen_frames)
    
