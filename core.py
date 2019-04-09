from preprocess import format_example
from loss import lossless_triplet_loss

in_dims = (N_MINS, n_feat)
out_dims = N_FACTORS

# Network definition
with tf.device(tf_device):

    # Create the 3 inputs
    anchor_in = Input(shape=in_dims)
    pos_in = Input(shape=in_dims)
    neg_in = Input(shape=in_dims)

    # Share base network with the 3 inputs
    base_network = create_base_network(in_dims, out_dims)
    anchor_out = base_network(anchor_in)
    pos_out = base_network(pos_in)
    neg_out = base_network(neg_in)
    merged_vector = concatenate([anchor_out, pos_out, neg_out], axis=-1)

    # Define the trainable model
    model = Model(inputs=[anchor_in, pos_in, neg_in], outputs=merged_vector)
    model.compile(optimizer=Adam(),
                  loss=lossless_triplet_loss)

# Training the model
model.fit(train_data, y_hat, batch_size=256, epochs=10)