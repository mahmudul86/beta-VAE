import tensorflow as tf
import numpy as np










#loading the weigts and biases of the generator network i.e. the encoder and decoder newtwork parameters. 
#for example EW_1 refers encoder weight matrix between the input layer and the first hidded layer. 
EW_1=np.loadtxt('./temporary/Encoder_W1.dat').astype(np.float32)
Eb_1=np.loadtxt('./temporary/Encoder_b1.dat').astype(np.float32)
EW_2=np.loadtxt('./temporary/Encoder_W2.dat').astype(np.float32)
Eb_2=np.loadtxt('./temporary/Encoder_b2.dat').astype(np.float32)
EW_3=np.loadtxt('./temporary/Encoder_W3.dat').astype(np.float32)
Eb_3=np.loadtxt('./temporary/Encoder_b3.dat').astype(np.float32)

DW_4=np.loadtxt('./temporary/Decoder_W4.dat').astype(np.float32)
Db_4=np.loadtxt('./temporary/Decoder_b4.dat').astype(np.float32)
DW_5=np.loadtxt('./temporary/Decoder_W5.dat').astype(np.float32)
Db_5=np.loadtxt('./temporary/Decoder_b5.dat').astype(np.float32)
DW_6=np.loadtxt('./temporary/Decoder_W6.dat').astype(np.float32)
Db_6=np.loadtxt('./temporary/Decoder_b6.dat').astype(np.float32)

# #this function generates samples in the input space using just the trained decoder with two dimensional Gaussian as input
# #n_samples refers to the number of samples we want to generate
# #data_dim stands for the dimension of each sample

def decoder_output_samples(n_samples,data_dim):
    X_samples=np.zeros((n_samples,data_dim), dtype=float)
    for i in range(n_samples):
        if (i+1) % 10000 == 0:
            print('number of decoder samples generated so far:', i+1)
        z=tf.random.normal((1,2))
        hd_1=tf.nn.relu(tf.matmul(z, DW_4)+Db_4)
        hd_2=tf.nn.relu(tf.matmul(hd_1, DW_5)+Db_5)
        Decoder_output=tf.matmul(hd_2, DW_6)+Db_6
        mean_vector= Decoder_output[:,:data_dim]
        mean_de=np.zeros((data_dim), dtype=float)
        for j in range(data_dim):
            mean_de[j]=mean_vector[0,j]
        std_de = tf.exp(Decoder_output[:, data_dim:])
        sigma_de=np.zeros((data_dim, data_dim),dtype=float)
        for j in range(data_dim):
            for k in range(data_dim):
                if j==k:
                    sigma_de[j,k]=np.square(std_de[0,j])
        X_samples[i,:]=np.random.multivariate_normal(mean_de, sigma_de)
    
    X_samples=tf.reshape(X_samples,[n_samples,data_dim])

    return X_samples









# #this function generates samples in the input space using the trained encoder-decoder network parameters
# #ground truth samples are given at the encoder input
# #data_size refers to the number of data points we have in the dataset (i.e. training and testing data combined)
# #n_samples stands for the number of samples we want to generate
# #data_dim is the dimension of each sample
# #n_dim is the dimension of the latent space

def encoder_decoder_samples(data_size,n_samples,data_dim,n_dim):
    load=np.loadtxt('./temporary/non_dominant_data.dat').astype(np.float32)
    Encoder_input=tf.reshape(load, [data_size,data_dim])
    X_samples=np.zeros((n_samples,data_dim), dtype=float)
    for i in range(n_samples):
        if (i+1) % 10000 == 0:
            print('number of encoder_decoder samples generated so far:', i+1)
        #encoder_part
        input_i=Encoder_input[i,:]
        X_in=np.zeros((1,data_dim), dtype=float)
        for j in range(data_dim):
            X_in[0,j]=input_i[j]
        he_1=tf.nn.relu(tf.matmul(X_in, EW_1)+Eb_1)
        he_2=tf.nn.relu(tf.matmul(he_1, EW_2)+Eb_2)
        Encoder_output=tf.matmul(he_2, EW_3)+Eb_3
        mean_en= Encoder_output[:,:n_dim]
        std_en = tf.exp(Encoder_output[:, n_dim:])
        z=np.zeros((1,n_dim), dtype=float)
        for j in range(n_dim):
            z[0,j]=mean_en[0,j]+std_en[0,j]*np.random.normal(0,1)
        #decoder_part
        hd_1=tf.nn.relu(tf.matmul(z, DW_4)+Db_4)
        hd_2=tf.nn.relu(tf.matmul(hd_1, DW_5)+Db_5)
        Decoder_output=tf.matmul(hd_2, DW_6)+Db_6
        mean_vector= Decoder_output[:,:data_dim]
        mean_de=np.zeros((data_dim), dtype=float)
        for j in range(data_dim):
            mean_de[j]=mean_vector[0,j]
        std_de = tf.exp(Decoder_output[:, data_dim:])
        sigma_de=np.zeros((data_dim, data_dim),dtype=float)
        for j in range(data_dim):
            for k in range(data_dim):
                if j==k:
                    sigma_de[j,k]=np.square(std_de[0,j])
        X_samples[i,:]=np.random.multivariate_normal(mean_de, sigma_de)
    
    X_samples=tf.reshape(X_samples,[n_samples,data_dim])
      
    return X_samples


# deco_data=decoder_output_samples(40000, 10)
# en_deco_data=encoder_decoder_samples(320000,40000,10,2)

# np.savetxt('./vae_training_records/training_06_record/training_06_data/deco_data.dat'.format(), deco_data)
# np.savetxt('./vae_training_records/training_06_record/training_06_data/en_deco_data.dat'.format(), en_deco_data)

def number_of_active_units(data_size,n_samples,data_dim,n_dim):
    load=np.loadtxt('./temporary/dominant_model_data.dat').astype(np.float32)
    Encoder_input=tf.reshape(load, [data_size,data_dim])
    mean_samples=np.zeros((n_samples,n_dim), dtype=float)
    std_samples=np.zeros((n_samples,n_dim), dtype=float)
    for i in range(n_samples):
        input_i=Encoder_input[i,:]
        X_in=np.zeros((1,data_dim), dtype=float)
        for j in range(data_dim):
            X_in[0,j]=input_i[j]
        he_1=tf.nn.relu(tf.matmul(X_in, EW_1)+Eb_1)
        he_2=tf.nn.relu(tf.matmul(he_1, EW_2)+Eb_2)
        Encoder_output=tf.matmul(he_2, EW_3)+Eb_3
        mean_en= Encoder_output[:,:n_dim]
        mean_samples[i,:]=mean_en
        std_en = tf.exp(Encoder_output[:, n_dim:])
        std_samples[i,:]=std_en
    var_mean_vector=np.zeros((1,n_dim), dtype=float)
    var_std_vector=np.zeros((1,n_dim), dtype=float)

    for j in range(n_dim):
        var_mean_vector[0,j]=np.var(mean_samples[:,j])
        var_std_vector[0,j]=np.var(std_samples[:,j])
        

    return var_mean_vector, var_std_vector  

g,t=number_of_active_units(1000000,5000,10,10)  
print(g,t)


def number_of_active_units_latent_to_data(n_samples,data_dim,n_dim):
    mean_matrix=np.zeros((n_samples,data_dim))
    for i in range(n_samples):
        a=[np.random.normal(0,1),0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
        z=tf.reshape(a,[1,9])
        hd_1=tf.nn.relu(tf.matmul(z, DW_4)+Db_4)
        hd_2=tf.nn.relu(tf.matmul(hd_1, DW_5)+Db_5)
        Decoder_output=tf.matmul(hd_2, DW_6)+Db_6
        mean_vector= Decoder_output[:,:data_dim]
        mean_matrix[i,:]=mean_vector
    var_mean_vector=np.zeros((1,data_dim), dtype=float)

    for j in range(data_dim):
        var_mean_vector[0,j]=np.var(mean_matrix[:,j])
        

    return var_mean_vector,mean_matrix
        
# g,t=number_of_active_units_latent_to_data(5,10,9)

# print(t)
# np.savetxt('./temporary/active_dimensions_vector.dat'.format(), g)







# #this is the executing command of the entire file after_training.py
# p=discriminator_performance_calculation(10)
# print(p)


