import fenics, numpy, scipy, sympy, pickle, os, sys, time, math, torch
from matplotlib import pyplot as plt
from scipy import sparse as sp
import ES

varepsilon = 0.001
inveps = 1.0/varepsilon

class RB_Data():
    '''
    store the reduced basis of u and p
    the eigenvalues D are in increasing order
    Paremeters: u_data: tuple   length = 4
                        (B, N, D, projB) the reduced basis data of u
                        2D array, int, 1D array, 2D array   shape = (N_full_u, N_u), 1, (sample_size,), (N_full_u, N_u)
                        reduced basis and RB dimension of u, the eigenvalues and the projection matrix
                p_data: tuple   length = 4
                        (B, N, D, projB) the reduced basis data of p
                Mc: 2D array   shape=(N_p, N_u)
                    the basis matrix of (p, u)
                p_tensor: list of 2D array   shape = N_p*(2,2)
                          list of the basis macroscopic p
    Attributes: u:   U()
                     U().Basis: shape = (N_full_u, N_u), the reduced basis of u
                     U().N: the RB dimension of u
                     U().D: the eigenvalue of u
                     U().proj: the projector from full order to RB of u
                p:   U()
                     U().Basis: shape = (N_full_p, N_p), the reduced basis of p
                     U().N: the RB dimension of p
                     U().D: the eigenvalue of p
                     U().proj: the projector from full order to RB of p
    '''
    class U():
        def __init__(self, B, N, D, projB) -> None:
            self.Basis = B
            self.N = N
            self.D = D
            self.proj = projB
    def __init__(self, u_data, p_data, Mc, pt) -> None:
        self.u = self.U(u_data[0], u_data[1], u_data[2], u_data[3])
        self.p = self.U(p_data[0], p_data[1], p_data[2], p_data[3])
        self.Mc = Mc
        self.p_tensor = pt

    def project_u_rb(self, v):
        '''
        reture the RB coefficients of the v
        Parameters: v: 1D or 2D array   shape = (N_full_u,) or (n, N_full_u)
                       full order coefficients array for the unknown u
        Returns:    v_rb: 1D or 2D array   shape = (N_u,) or (n, N_u)
                          reduced basis coefficients array of v
        '''
        if v.ndim == 1:
            if v.shape[0] != self.u.proj.shape[0]:
                raise Exception('Projection fail! Row number is inconsistent! Dimension is %d but needs %d' %(v.shape[0], self.u.proj.shape[0]))
        elif v.shape[1] != self.u.proj.shape[0]:
            raise Exception('Projection fail! Row number is inconsistent! Dimension is %d but needs %d' %(v.shape[1], self.u.proj.shape[0]))
        return numpy.dot(v, self.u.proj)*1.0e5
    
    def return_u_full_order(self, v):
        '''
        reture the full order coefficients of the v
        Parameters: v: 1D or 2D array   shape = (N_u,) or (n, N_u)
                       reduced basis coefficient array for the unknown u
        Returns:    v_f: 1D or 2D array   shape = (N_full_u,) or (n, N_full_u)
                          full order coefficients array of v
        ''''''
        if v.ndim == 1:
            if v.shape[0] != self.u.proj.shape[1]:
                raise Exception('Projection fail! Row number is inconsistent! Dimension is %d but needs %d' %(v.shape[0], self.u.proj.shape[1]))
        elif v.shape[1] != self.u.proj.shape[1]:
            raise Exception('Row number is inequal to the reduced order! Dimension is %d but needs %d' %(v.shape[1], self.u.proj.shape[1]))
        '''
        return numpy.dot(v, self.u.Basis.transpose())*1.0e-5

    def project_p_rb(self, v):
        '''
        reture the RB coefficients of the v
        Parameters: v: 1D or 2D array   shape = (N_full_p,) or (n, N_full_p)
                       full order coefficients array for the flux p
        Returns:    v_rb: 1D or 2D array   shape = (N_p,) or (n, N_p)
                          reduced basis coefficients array of v
        ''''''
        if v.ndim == 1:
            if v.shape[0] != self.p.proj.shape[0]:
                raise Exception('Projection fail! Row number is inconsistent! Dimension is %d but needs %d' %(v.shape[0], self.p.proj.shape[0]))
        elif v.shape[1] != self.p.proj.shape[0]:
            raise Exception('Projection fail! Row number is inconsistent! Dimension is %d but needs %d' %(v.shape[1], self.p.proj.shape[0]))
        '''
        return numpy.dot(v, self.p.proj)*1.0e1
    
    def return_p_full_order(self, v):
        '''
        reture the full order coefficients of the v
        Parameters: v: 1D or 2D array   shape = (N_p,) or (n, N_p)
                       reduced basis coefficient array for the flux p
        Returns:    v_f: 1D or 2D array   shape = (N_full_p,) or (n, N_full_p)
                          full order coefficients array of v
        '''
        if v.ndim == 1:
            if v.shape[0] != self.p.proj.shape[1]:
                raise Exception('Projection fail! Row number is inconsistent! Dimension is %d but needs %d' %(v.shape[0], self.p.proj.shape[1]))
        elif v.shape[1] != self.p.proj.shape[1]:
            raise Exception('Row number is inequal to the reduced order! Dimension is %d but needs %d' %(v.shape[1], self.p.proj.shape[1]))
        return numpy.dot(v, self.p.Basis.transpose())*1.0e-1

class Training_Data():
    '''
    store the NN training data
    Paremeters: tp: 2D array   shape=(sample_size, N_para)
                    the training points (sample of parameter)
                tlq: 2D array   shape=(sample_size, N_u)
                     the training labels of u (rb coefficients) 
    Attributes: points   shape = (sample_size, N_para) 
                labels   shape = (sample_size, N_u)
    '''
    def __init__(self, para, label) -> None:
        self.points = para
        self.labels = label

class POD_vector():
    def __init__(self, u_rb, p_rb, mu) -> None:
        self.u_rb = u_rb
        self.p_rb = p_rb
        self.mu = mu

class NN_aio():
    '''
    the calss for the Neural Network that map mu -> P the parameter directly to the derived quantity
    Attributes: scale_data: Scale_Data()
                            the scale data to standardize the training_data and inverse_standardize the output
                n_hidden: Int
                          the number of nodes of hidden layers
                n_input: Int
                         the number of nodes of input layers
                n_output: Int
                          the number of nodes of hidden layers
                nn: NN.Net()
                    the neural network
    '''

    class Scale_Data():
        '''
        the class for standardizing
        Parameters:  training_data: Training_Data()
        Varaibles:  points_scale: 2D tensor   shape=(2, N_para)
                                  points_scale[0,:]   mean
                                  points_scale[1,:]   var
                    labels_scale: 2D tensor   shape=(2, N_u)
                                  labels_scale[0,:]   mean
                                  labels_scale[1,:]   var
        functions:   standardize()   
                     inv_standardize()
        '''
        def __init__(self, training_data) -> None:
            eps = 1.e-8
            self.points_scale = torch.from_numpy(numpy.mean(training_data.points, axis = 0)).float()
            self.points_scale = torch.vstack((self.points_scale, torch.from_numpy(numpy.var(training_data.points, axis = 0) + eps).float()))
            self.labels_scale = torch.from_numpy(numpy.mean(training_data.labels, axis = 0)).float()
            self.labels_scale = torch.vstack((self.labels_scale, torch.from_numpy(numpy.var(training_data.labels, axis = 0) + eps).float()))

        def standardize(self, x, scale):
            '''
            standardize the given data x by scale data
            each dimension are standardized autonomously
            Parameters: x: 2D tensor   shape = (n, dim) 
                        input data
                        scale: 2D tensor   shape = (2, dim)
                               2 choices: points_scale, labels_scale
            Returns:    x2: 2D tensor   shape = (n, dim) 
                            output standardized data
            '''
            x_mean, x_var = scale[0,:], scale[1,:]
            return (x-x_mean)/torch.sqrt(x_var)
        def inv_standardize(self, x, scale):
            '''
            inverse standardize the given data x
            each dimension are inverse standardized autonomously
            Parameters: x: 2D tensor   shape = (n, dim) 
                        input data to be inverse_standardized
                        scale: 2D tensor   shape = (2, dim)
                               2 choices: points_scale, labels_scale
            Returns:    x2: 2D tensor   shape = (n, dim) 
                            output unstandardized data
            '''
            x_mean, x_var = scale[0,:], scale[1,:]
            return x*torch.sqrt(x_var)+x_mean
    def __init__(self, n_hidden=100) -> None:
        self.scale_data = None
        self.n_hidden = n_hidden

    def get_scale(self, training_data):
        self.scale_data = self.Scale_Data(training_data)
        self.n_input = training_data.points.shape[1]
        self.n_output = training_data.labels.shape[1]
        self.nn = NN_aio.Net(self.n_input, self.n_hidden, self.n_output)

    class Res(torch.nn.Module):
        # ResNet block
        def __init__(self, n_input, n_output) -> None:
            assert n_input==n_output
            super(NN_aio.Res, self).__init__()
            self.layer = torch.nn.Sequential(
                torch.nn.Linear(n_input, n_output),
                torch.nn.ELU(inplace=True),
                torch.nn.Linear(n_output, n_output)
            )
            self.elu = torch.nn.ELU(inplace=True)
        def forward(self, input):
            output = self.layer(input) + input
            return self.elu(output)
    class Net(torch.nn.Module):
        def __init__(self, n_input, n_hidden_0, n_output) -> None:
            super(NN_aio.Net, self).__init__()
            self.layers = torch.nn.Sequential(
                #torch.nn.BatchNorm1d(n_input),
                torch.nn.Linear(n_input, n_hidden_0),
                torch.nn.ELU(inplace=True),
                #torch.nn.Dropout(drop_prob_0),
                NN_aio.Res(n_hidden_0, n_hidden_0),
                NN_aio.Res(n_hidden_0, n_hidden_0),
                NN_aio.Res(n_hidden_0, n_hidden_0),
                NN_aio.Res(n_hidden_0, n_hidden_0),
                NN_aio.Res(n_hidden_0, n_hidden_0),
                NN_aio.Res(n_hidden_0, n_hidden_0),
                NN_aio.Res(n_hidden_0, n_hidden_0),
                NN_aio.Res(n_hidden_0, n_hidden_0),
                NN_aio.Res(n_hidden_0, n_hidden_0),
                NN_aio.Res(n_hidden_0, n_hidden_0),
                torch.nn.Linear(n_hidden_0, n_output)
            )
        def forward(self, input):
            return self.layers(input)
    
    def train(self, training_data):
        learningrate = 0.005
        derate = 0.8
        lrate = 2e-5
        num_workers = 4
        BATCH_SIZE = 64
        EPOCH = 6000
        x_train = self.scale_data.standardize(torch.from_numpy(
            training_data.points).float(), self.scale_data.points_scale)
        #y_train_q = self.scale_data.standardize(torch.from_numpy(
        #   training_data.labels).float(), self.scale_data.labels_scale)
        y_train = torch.from_numpy(training_data.labels).float()
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        train_size = int(0.8 * len(dataset))
        valid_size = len(dataset) - train_size
        train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, valid_size], generator=torch.manual_seed(0))
        dataloader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=num_workers
        )
        validloader = torch.utils.data.DataLoader(
            dataset=valid_data, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=num_workers
        )
        def loss_f(a,b):
            mse = torch.nn.MSELoss(reduction='sum')
            d = a.shape[0]
            return mse(a,b)/d
        optimizer = torch.optim.Adam(self.nn.parameters(), lr = learningrate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=derate, patience=300, verbose=False, 
            threshold=0.0001, threshold_mode='rel', cooldown=500, min_lr=0, eps=lrate
        )
        early_stopping = ES.EarlyStopping(patience=120)
        print(self.nn)
        sys.stdout.flush()
        start = time.time()
        losses = []
        for epoch in range(EPOCH):
            self.nn.train()
            train_l_sum, batch_step = 0, 0
            for batch_x, batch_y in dataloader:
                b_x = batch_x
                b_y = batch_y
                optimizer.zero_grad()
                b_pred = self.nn(b_x)
                loss = loss_f(b_pred, b_y)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                train_l_sum += loss.item()
                batch_step += 1
            train_l_sum /= batch_step
            losses.append(train_l_sum)
            if epoch%10 == 9 or epoch == 0 or epoch == 4:
                print('Epoch %d/%d - Loss: %.8f' % (epoch+1, EPOCH, train_l_sum))
                sys.stdout.flush()
            self.nn.eval()
            valid_l_sum, batch_step = 0, 0
            for v_x, v_y in iter(validloader):
                v_p = self.nn(v_x)
                loss = loss_f(v_p, v_y)
                valid_l_sum += loss.item()
                batch_step += 1
            valid_l_sum /= batch_step
            early_stopping(valid_l_sum, self.nn)
            if epoch%10 == 9 or epoch == 0 or epoch == 4:
                print('-------%d-- Val Loss: %.8f' % (epoch+1, valid_l_sum))
                sys.stdout.flush()
            if early_stopping.early_stop:
                print("Early stopping at Epoch %d/%d - Loss: %.8f" % (epoch+1, EPOCH, train_l_sum))
                break
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'NNaio_loss' + '.pickle'), 'wb') as file:
            pickle.dump(losses, file)
        plt.plot(losses, label='loss_aio',lw=0.5)
        plt.legend(loc='best')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.savefig('loss_aio.png')
        plt.close()
        end = time.time() - start
        print('The training time: %.5fs' % end)

        if not os.path.exists(os.path.join(os.path.abspath('.'), 'offline_data')):
            os.mkdir(os.path.join(os.path.abspath('.'), 'offline_data'))
        torch.save(self.nn.state_dict(), os.path.join(os.path.abspath('.'), 'offline_data', 'NNaio_para.pkl'))

    def load_para(self):
        self.nn = NN_aio.Net(self.n_input, self.n_hidden, self.n_output)
        self.nn.load_state_dict(torch.load(os.path.join(os.path.abspath('.'), 'offline_data', 'NNaio_para.pkl')))
        self.nn.eval()
    def predict(self, x):
        '''
        the prediction functions
        Parameters: x: 1D or 2D array   shape = (N_para,) (n, N_para)
                        the point to evaluate the prediction
        Returns:    y: 1D or 2D array   shape = (N_u,) (n, N_u)
                        the predicted label for regression of u at point x
        '''
        self.nn.eval()
        xi = x
        if xi.ndim == 1:
            x = x.reshape(1,-1)
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            x = self.scale_data.standardize(x, self.scale_data.points_scale)
            y = self.nn(x)
            #y = self.scale_data.inv_standardize(y, self.scale_data.q_labels_scale)
        y = y.numpy()
        if xi.ndim == 1:
            y = y.flatten()
        return y
    def Jacobian(self, x):
        '''
        get the Jacobian at x
        Parameters: x: 1D array   shape = (N_para,)
                        the point to evaluate the prediction and Jacobian
        Returns:    y: 1D array   shape = (N_u,)
                        the predicted label for regression of u at point x
                    J: 2D array   shape = (N_u, N_para)
                        the Jacobian matrix for regression of u at point x
        '''
        self.nn.eval()
        n_x = self.scale_data.points_scale.shape[1]
        n_y = self.scale_data.labels_scale.shape[1]
        J = torch.empty(n_y, n_x)
        x = torch.from_numpy(x.reshape(1,-1)).float()
        x.requires_grad_(True)
        xm = self.scale_data.standardize(x, self.scale_data.points_scale)
        y = self.nn(xm)
        #y = self.scale_data.inv_standardize(ym, self.scale_data.q_labels_scale)
        for i in range(n_y):
            y[0,i].backward(retain_graph=True)
            t = x.grad
            J[i,:] = t[0,:]
            x.grad.data.zero_()
        y = y.detach().numpy().flatten()
        return y, J.numpy()

class NN_U():
    '''
    the calss for the Neural Network map mu -> u the parameter to the unknown quantity
    also serve as the first tier of network
    Attributes: scale_data: Scale_Data()
                            the scale data to standardize the training_data and inverse_standardize the output
                n_hidden: Int
                          the number of nodes of hidden layers
                n_input: Int
                         the number of nodes of input layers
                n_output: Int
                          the number of nodes of hidden layers
                nn: NN.Net()
                    the neural network
    '''

    class Scale_Data():
        '''
        the class for standardizing
        Parameters:  training_data: Training_Data()
        Varaibles:  points_scale: 2D tensor   shape=(2, N_para)
                                  points_scale[0,:]   mean
                                  points_scale[1,:]   var
                    labels_scale: 2D tensor   shape=(2, N_u)
                                  labels_scale[0,:]   mean
                                  labels_scale[1,:]   var
        functions:   standardize()   
                     inv_standardize()
        '''
        def __init__(self, training_data) -> None:
            eps = 1.e-8
            self.points_scale = torch.from_numpy(numpy.mean(training_data.points, axis = 0)).float()
            self.points_scale = torch.vstack((self.points_scale, torch.from_numpy(numpy.var(training_data.points, axis = 0) + eps).float()))
            self.labels_scale = torch.from_numpy(numpy.mean(training_data.labels, axis = 0)).float()
            self.labels_scale = torch.vstack((self.labels_scale, torch.from_numpy(numpy.var(training_data.labels, axis = 0) + eps).float()))

        def standardize(self, x, scale):
            '''
            standardize the given data x by scale data
            each dimension are standardized autonomously
            Parameters: x: 2D tensor   shape = (n, dim) 
                        input data
                        scale: 2D tensor   shape = (2, dim)
                               2 choices: points_scale, labels_scale
            Returns:    x2: 2D tensor   shape = (n, dim) 
                            output standardized data
            '''
            x_mean, x_var = scale[0,:], scale[1,:]
            return (x-x_mean)/torch.sqrt(x_var)
        def inv_standardize(self, x, scale):
            '''
            inverse standardize the given data x
            each dimension are inverse standardized autonomously
            Parameters: x: 2D tensor   shape = (n, dim) 
                        input data to be inverse_standardized
                        scale: 2D tensor   shape = (2, dim)
                               2 choices: points_scale, labels_scale
            Returns:    x2: 2D tensor   shape = (n, dim) 
                            output unstandardized data
            '''
            x_mean, x_var = scale[0,:], scale[1,:]
            return x*torch.sqrt(x_var)+x_mean
    def __init__(self, n_hidden=100) -> None:
        self.scale_data = None
        self.n_hidden = n_hidden

    def get_scale(self, training_data):
        self.scale_data = self.Scale_Data(training_data)
        self.n_input = training_data.points.shape[1]
        self.n_output = training_data.labels.shape[1]
        self.nn = NN_U.Net(self.n_input, self.n_hidden, self.n_output)

    class Res(torch.nn.Module):
        def __init__(self, n_input, n_output) -> None:
            assert n_input==n_output
            super(NN_U.Res, self).__init__()
            self.layer = torch.nn.Sequential(
                torch.nn.Linear(n_input, n_output),
                torch.nn.ELU(inplace=True),
                torch.nn.Linear(n_output, n_output)
            )
            self.elu = torch.nn.ELU(inplace=True)
        def forward(self, input):
            output = self.layer(input) + input
            return self.elu(output)
    class Net(torch.nn.Module):
        def __init__(self, n_input, n_hidden_0, n_output) -> None:
            super(NN_U.Net, self).__init__()
            self.layers = torch.nn.Sequential(
                #torch.nn.BatchNorm1d(n_input),
                torch.nn.Linear(n_input, n_hidden_0),
                torch.nn.ELU(inplace=True),
                #torch.nn.Dropout(drop_prob_0),
                NN_U.Res(n_hidden_0, n_hidden_0),
                NN_U.Res(n_hidden_0, n_hidden_0),
                NN_U.Res(n_hidden_0, n_hidden_0),
                NN_U.Res(n_hidden_0, n_hidden_0),
                NN_U.Res(n_hidden_0, n_hidden_0),
                NN_U.Res(n_hidden_0, n_hidden_0),
                NN_U.Res(n_hidden_0, n_hidden_0),
                NN_U.Res(n_hidden_0, n_hidden_0),
                torch.nn.Linear(n_hidden_0, n_output)
            )
        def forward(self, input):
            return self.layers(input)
    
    def train(self, training_data, retrain=False):
        # if retrain, then the training function will use a smaller learning rate
        # please load the pretrain parameter if retrain
        if retrain:
            learningrate = 0.001
            derate = 0.95
            lrate = 2e-5
        else:
            learningrate = 0.005
            derate = 0.8
            lrate = 2e-5
        num_workers = 4
        BATCH_SIZE = 64
        EPOCH = 6000
        x_train = self.scale_data.standardize(torch.from_numpy(
            training_data.points).float(), self.scale_data.points_scale)
        #y_train_q = self.scale_data.standardize(torch.from_numpy(
        #   training_data.labels).float(), self.scale_data.labels_scale)
        y_train = torch.from_numpy(training_data.labels).float()
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        train_size = int(0.8 * len(dataset))
        valid_size = len(dataset) - train_size
        train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, valid_size], generator=torch.manual_seed(0))
        dataloader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=num_workers
        )
        validloader = torch.utils.data.DataLoader(
            dataset=valid_data, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=num_workers
        )
        def loss_f(a,b):
            mse = torch.nn.MSELoss(reduction='sum')
            d = a.shape[0]
            return mse(a,b)/d
        optimizer = torch.optim.Adam(self.nn.parameters(), lr = learningrate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=derate, patience=300, verbose=False, 
            threshold=0.0001, threshold_mode='rel', cooldown=500, min_lr=0, eps=lrate
        )
        early_stopping = ES.EarlyStopping(patience=120)
        print(self.nn)
        sys.stdout.flush()
        start = time.time()
        losses = []
        for epoch in range(EPOCH):
            self.nn.train()
            train_l_sum, batch_step = 0, 0
            for batch_x, batch_y in dataloader:
                b_x = batch_x
                b_y = batch_y
                optimizer.zero_grad()
                b_pred = self.nn(b_x)
                loss = loss_f(b_pred, b_y)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                train_l_sum += loss.item()
                batch_step += 1
            train_l_sum /= batch_step
            losses.append(train_l_sum)
            if epoch%10 == 9 or epoch == 0 or epoch == 4:
                print('Epoch %d/%d - Loss: %.8f' % (epoch+1, EPOCH, train_l_sum))
                sys.stdout.flush()
            self.nn.eval()
            valid_l_sum, batch_step = 0, 0
            for v_x, v_y in iter(validloader):
                v_p = self.nn(v_x)
                loss = loss_f(v_p, v_y)
                valid_l_sum += loss.item()
                batch_step += 1
            valid_l_sum /= batch_step
            early_stopping(valid_l_sum, self.nn)
            if epoch%10 == 9 or epoch == 0 or epoch == 4:
                print('-------%d-- Val Loss: %.8f' % (epoch+1, valid_l_sum))
                sys.stdout.flush()
            if early_stopping.early_stop:
                print("Early stopping at Epoch %d/%d - Loss: %.8f" % (epoch+1, EPOCH, train_l_sum))
                break
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'NNu_loss' + '.pickle'), 'wb') as file:
            pickle.dump(losses, file)
        plt.plot(losses, label='loss_u',lw=0.5)
        plt.legend(loc='best')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.savefig('loss_u.png')
        plt.close()
        end = time.time() - start
        print('The training time: %.5fs' % end)

        if not os.path.exists(os.path.join(os.path.abspath('.'), 'offline_data')):
            os.mkdir(os.path.join(os.path.abspath('.'), 'offline_data'))
        torch.save(self.nn.state_dict(), os.path.join(os.path.abspath('.'), 'offline_data', 'NNu_para.pkl'))

    def load_para(self):
        self.nn = NN_U.Net(self.n_input, self.n_hidden, self.n_output)
        self.nn.load_state_dict(torch.load(os.path.join(os.path.abspath('.'), 'offline_data', 'NNu_para.pkl')))
        self.nn.eval()
    def predict(self, x):
        '''
        the prediction functions
        Parameters: x: 1D or 2D array   shape = (N_para,) (n, N_para)
                        the point to evaluate the prediction
        Returns:    y: 1D or 2D array   shape = (N_u,) (n, N_u)
                        the predicted label for regression of u at point x
        '''
        self.nn.eval()
        xi = x
        if xi.ndim == 1:
            x = x.reshape(1,-1)
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            x = self.scale_data.standardize(x, self.scale_data.points_scale)
            y = self.nn(x)
            #y = self.scale_data.inv_standardize(y, self.scale_data.q_labels_scale)
        y = y.numpy()
        if xi.ndim == 1:
            y = y.flatten()
        return y
    def Jacobian(self, x):
        '''
        get the Jacobian at x
        Parameters: x: 1D array   shape = (N_para,)
                        the point to evaluate the prediction and Jacobian
        Returns:    y: 1D array   shape = (N_u,)
                        the predicted label for regression of u at point x
                    J: 2D array   shape = (N_u, N_para)
                        the Jacobian matrix for regression of u at point x
        '''
        self.nn.eval()
        n_x = self.scale_data.points_scale.shape[1]
        n_y = self.scale_data.labels_scale.shape[1]
        J = torch.empty(n_y, n_x)
        x = torch.from_numpy(x.reshape(1,-1)).float()
        x.requires_grad_(True)
        xm = self.scale_data.standardize(x, self.scale_data.points_scale)
        y = self.nn(xm)
        #y = self.scale_data.inv_standardize(ym, self.scale_data.q_labels_scale)
        for i in range(n_y):
            y[0,i].backward(retain_graph=True)
            t = x.grad
            J[i,:] = t[0,:]
            x.grad.data.zero_()
        y = y.detach().numpy().flatten()
        return y, J.numpy()

    def load_from_f(self, nnp, nnf): 
        # load parameter from nnf(TTDN), requires that nnf should be constituted by nnu and nnp 
        udict = self.nn.state_dict().copy()
        pdict = nnp.nn.state_dict().copy()
        fdict = nnf.nn.state_dict().copy()
        ulist = list(udict.keys())
        plist = list(pdict.keys())
        flist = list(fdict.keys())
        Nu = len(ulist)
        Np = len(plist)
        Nf = len(flist)
        assert Nf == Nu+Np
        for i in range(Nu):
            udict[ulist[i]] = fdict[flist[i]]
        self.nn.load_state_dict(udict)

class NN_P():
    '''
    the calss for the Neural Network that map (u, mu) -> P to the derivative 
    work as an approximated constitutive law, i.e., the second tier of network
    Attributes: scale_data: Scale_Data()
                            the scale data to standardize the training_data and inverse_standardize the output
                n_hidden: Int
                          the number of nodes of hidden layers
                n_input: Int
                         the number of nodes of input layers
                n_output: Int
                          the number of nodes of hidden layers
                nn: NN.Net()
                    the neural network
    '''

    class Scale_Data():
        '''
        the class for standardizing
        Parameters:  training_data: Training_Data()
        Varaibles:  points_scale: 2D tensor   shape=(2, N_para+Nu)
                                  points_scale[0,:]   mean
                                  points_scale[1,:]   var
                    labels_scale: 2D tensor   shape=(2, N_p)
                                  labels_scale[0,:]   mean
                                  labels_scale[1,:]   var
        functions:   standardize()   
                     inv_standardize()
        '''
        def __init__(self, training_data) -> None:
            eps = 1.e-8
            self.points_scale = torch.from_numpy(numpy.mean(training_data.points, axis = 0)).float()
            self.points_scale = torch.vstack((self.points_scale, torch.from_numpy(numpy.var(training_data.points, axis = 0) + eps).float()))
            self.labels_scale = torch.from_numpy(numpy.mean(training_data.labels, axis = 0)).float()
            self.labels_scale = torch.vstack((self.labels_scale, torch.from_numpy(numpy.var(training_data.labels, axis = 0) + eps).float()))

        def standardize(self, x, scale):
            '''
            standardize the given data x by scale data
            each dimension are standardized autonomously
            Parameters: x: 2D tensor   shape = (n, dim) 
                        input data
                        scale: 2D tensor   shape = (2, dim)
                               2 choices: points_scale, labels_scale
            Returns:    x2: 2D tensor   shape = (n, dim) 
                            output standardized data
            '''
            x_mean, x_var = scale[0,:], scale[1,:]
            return (x-x_mean)/torch.sqrt(x_var)
        def inv_standardize(self, x, scale):
            '''
            inverse standardize the given data x
            each dimension are inverse standardized autonomously
            Parameters: x: 2D tensor   shape = (n, dim) 
                        input data to be inverse_standardized
                        scale: 2D tensor   shape = (2, dim)
                               2 choices: points_scale, labels_scale
            Returns:    x2: 2D tensor   shape = (n, dim) 
                            output unstandardized data
            '''
            x_mean, x_var = scale[0,:], scale[1,:]
            return x*torch.sqrt(x_var)+x_mean
    def __init__(self, n_hidden=100) -> None:
        self.scale_data = None
        self.n_hidden = n_hidden

    def get_scale(self, training_data):
        self.scale_data = self.Scale_Data(training_data)
        self.n_input = training_data.points.shape[1]
        self.n_output = training_data.labels.shape[1]
        self.nn = NN_P.Net(self.n_input, self.n_hidden, self.n_output)

    class Res(torch.nn.Module):
        def __init__(self, n_input, n_output) -> None:
            assert n_input==n_output
            super(NN_P.Res, self).__init__()
            self.layer = torch.nn.Sequential(
                torch.nn.Linear(n_input, n_output),
                torch.nn.ELU(inplace=True),
                torch.nn.Linear(n_output, n_output)
            )
            self.elu = torch.nn.ELU(inplace=True)
        def forward(self, input):
            output = self.layer(input) + input
            return self.elu(output)
    class Net(torch.nn.Module):
        def __init__(self, n_input, n_hidden_0, n_output) -> None:
            super(NN_P.Net, self).__init__()
            self.layers = torch.nn.Sequential(
                #torch.nn.BatchNorm1d(n_input),
                torch.nn.Linear(n_input, n_hidden_0),
                torch.nn.ELU(inplace=True),
                #torch.nn.Dropout(drop_prob_0),
                NN_P.Res(n_hidden_0, n_hidden_0),
                NN_P.Res(n_hidden_0, n_hidden_0),
                NN_P.Res(n_hidden_0, n_hidden_0),
                NN_P.Res(n_hidden_0, n_hidden_0),
                #NN_P.Res(n_hidden_0, n_hidden_0),
                #NN_P.Res(n_hidden_0, n_hidden_0),
                #NN_P.Res(n_hidden_0, n_hidden_0),
                #NN_P.Res(n_hidden_0, n_hidden_0),
                #NN_P.Res(n_hidden_0, n_hidden_0),
                #NN_P.Res(n_hidden_0, n_hidden_0),
                torch.nn.Linear(n_hidden_0, n_output)
            )
        def forward(self, input):
            return self.layers(input)
    
    def train(self, training_data, retrain=False, *args):
        if retrain:
            learningrate = 0.00002
            derate = 0.95
            lrate = 2e-6
        else:
            learningrate = 0.005
            derate = 0.8
            lrate = 2e-5
        num_workers = 4
        BATCH_SIZE = 64*4
        EPOCH = 6000
        x_train = self.scale_data.standardize(torch.from_numpy(
            training_data.points).float(), self.scale_data.points_scale)
        #y_train_q = self.scale_data.standardize(torch.from_numpy(
        #   training_data.labels).float(), self.scale_data.labels_scale)
        y_train = torch.from_numpy(training_data.labels).float()
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        train_size = int(0.8 * len(dataset))
        valid_size = len(dataset) - train_size
        train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, valid_size], generator=torch.manual_seed(0))
        validloader = torch.utils.data.DataLoader(
            dataset=valid_data, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=num_workers
        )
        if args:
            ds = [train_data]
            for tdp in args:
                x_tdp = self.scale_data.standardize(torch.from_numpy(
                    tdp.points).float(), self.scale_data.points_scale)
                y_tdp = torch.from_numpy(tdp.labels).float()
                dataset_tdp = torch.utils.data.TensorDataset(x_tdp, y_tdp)
                ds.append(dataset_tdp)
            dataset_c = torch.utils.data.ConcatDataset(ds)
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset_c, batch_size=BATCH_SIZE, 
                shuffle=True, num_workers=num_workers
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset=train_data, batch_size=BATCH_SIZE, 
                shuffle=True, num_workers=num_workers
            )
        def loss_f(a,b):
            mse = torch.nn.MSELoss(reduction='sum')
            d = a.shape[0]
            return mse(a,b)/d
        optimizer = torch.optim.Adam(self.nn.parameters(), lr = learningrate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=derate, patience=300, verbose=False, 
            threshold=0.0001, threshold_mode='rel', cooldown=500, min_lr=0, eps=lrate
        )
        early_stopping = ES.EarlyStopping(patience=300)
        print(self.nn)
        sys.stdout.flush()
        start = time.time()
        losses = []
        for epoch in range(EPOCH):
            if epoch == 0 and retrain:
                self.nn.eval()
                valid_l_sum, batch_step = 0, 0
                for v_x, v_y in iter(validloader):
                    v_p = self.nn(v_x)
                    loss = loss_f(v_p, v_y)
                    valid_l_sum += loss.item()
                    batch_step += 1
                valid_l_sum /= batch_step
                early_stopping(valid_l_sum, self.nn)
                print('-------%d-- Val Loss: %.8f' % (epoch+1, valid_l_sum))
                sys.stdout.flush()
            self.nn.train()
            train_l_sum, batch_step = 0, 0
            for batch_x, batch_y in dataloader:
                b_x = batch_x
                b_y = batch_y
                optimizer.zero_grad()
                b_pred = self.nn(b_x)
                loss = loss_f(b_pred, b_y)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                train_l_sum += loss.item()
                batch_step += 1
            train_l_sum /= batch_step
            losses.append(train_l_sum)
            if epoch%10 == 9 or epoch == 0 or epoch == 4:
                print('Epoch %d/%d - Loss: %.8f' % (epoch+1, EPOCH, train_l_sum))
                sys.stdout.flush()
            self.nn.eval()
            valid_l_sum, batch_step = 0, 0
            for v_x, v_y in iter(validloader):
                v_p = self.nn(v_x)
                loss = loss_f(v_p, v_y)
                valid_l_sum += loss.item()
                batch_step += 1
            valid_l_sum /= batch_step
            early_stopping(valid_l_sum, self.nn)
            if epoch%10 == 9 or epoch == 0 or epoch == 4:
                print('-------%d-- Val Loss: %.8f' % (epoch+1, valid_l_sum))
                sys.stdout.flush()
            if early_stopping.early_stop:
                print("Early stopping at Epoch %d/%d - Loss: %.8f" % (epoch+1, EPOCH, train_l_sum))
                break
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'NNp_loss' + '.pickle'), 'wb') as file:
            pickle.dump(losses, file)
        plt.plot(losses, label='loss_p',lw=0.5)
        plt.legend(loc='best')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.savefig('loss_p.png')
        plt.close()
        end = time.time() - start
        print('The training time: %.5fs' % end)

        if not os.path.exists(os.path.join(os.path.abspath('.'), 'offline_data')):
            os.mkdir(os.path.join(os.path.abspath('.'), 'offline_data'))
        torch.save(self.nn.state_dict(), os.path.join(os.path.abspath('.'), 'offline_data', 'NNp_para.pkl'))

    def load_para(self):
        self.nn = NN_P.Net(self.n_input, self.n_hidden, self.n_output)
        self.nn.load_state_dict(torch.load(os.path.join(os.path.abspath('.'), 'offline_data', 'NNp_para.pkl')))
        self.nn.eval()
    def predict(self, x):
        '''
        the prediction functions
        Parameters: x: 1D or 2D array   shape = (N_para+N_u,) (n, N_para+N_u)
                        the point to evaluate the prediction
        Returns:    y: 1D or 2D array   shape = (N_p,) (n, N_p)
                        the predicted label for regression of p at point x
        '''
        self.nn.eval()
        xi = x
        if xi.ndim == 1:
            x = x.reshape(1,-1)
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            x = self.scale_data.standardize(x, self.scale_data.points_scale)
            y = self.nn(x)
            #y = self.scale_data.inv_standardize(y, self.scale_data.q_labels_scale)
        y = y.numpy()
        if xi.ndim == 1:
            y = y.flatten()
        return y
    def Jacobian(self, x):
        '''
        get the Jacobian at x
        Parameters: x: 1D array   shape = (N_para+N_u,)
                        the point to evaluate the prediction and Jacobian
        Returns:    y: 1D array   shape = (N_p,)
                        the predicted label for regression of p at point x
                    J: 2D array   shape = (N_p, N_para+N_u)
                        the Jacobian matrix for regression of p at point x
        '''
        self.nn.eval()
        n_x = self.scale_data.points_scale.shape[1]
        n_y = self.scale_data.labels_scale.shape[1]
        J = torch.empty(n_y, n_x)
        x = torch.from_numpy(x.reshape(1,-1)).float()
        x.requires_grad_(True)
        xm = self.scale_data.standardize(x, self.scale_data.points_scale)
        y = self.nn(xm)
        #y = self.scale_data.inv_standardize(ym, self.scale_data.q_labels_scale)
        for i in range(n_y):
            y[0,i].backward(retain_graph=True)
            t = x.grad
            J[i,:] = t[0,:]
            x.grad.data.zero_()
        y = y.detach().numpy().flatten()
        return y, J.numpy()

    def load_from_f(self, nnu, nnf):
        # load parameter from nnf(TTDN), requires that nnf should be constituted by nnu and nnp
        udict = nnu.nn.state_dict().copy()
        pdict = self.nn.state_dict().copy()
        fdict = nnf.nn.state_dict().copy()
        ulist = list(udict.keys())
        plist = list(pdict.keys())
        flist = list(fdict.keys())
        Nu = len(ulist)
        Np = len(plist)
        Nf = len(flist)
        assert Nf == Nu+Np
        for i in range(Np):
            pdict[plist[i]] = fdict[flist[i+Nu]]
        self.nn.load_state_dict(pdict)

class NN_F():
    '''
    the calss for the TTDN Network, constituted by NN_U and NN_P
    Parameters: nnu: NN()
                     neural network of u
                nnp: NN()
                     neural network of p
    Attributes: scale_data: Scale_Data()
                            the scale data to standardize the training_data and inverse_standardize the output
                n_hidden: Int
                          the number of nodes of hidden layers
                n_input: Int
                         the number of nodes of input layers
                n_output: Int
                          the number of nodes of hidden layers
                nn: NN.Net()
                    the neural network
    '''
    class Scale_Data():
        '''
        the class for standardizing
        Parameters:  training_data: Training_Data()
        Varaibles:  points_scale: 2D tensor   shape=(2, N_para)
                                  points_scale[0,:]   mean
                                  points_scale[1,:]   var
                    labels_scale: 2D tensor   shape=(2, N_u)
                                  labels_scale[0,:]   mean
                                  labels_scale[1,:]   var
        functions:   standardize()   
                     inv_standardize()
        '''
        def __init__(self, training_data) -> None:
            eps = 1.e-8
            self.points_scale = torch.from_numpy(numpy.mean(training_data.points, axis = 0)).float()
            self.points_scale = torch.vstack((self.points_scale, torch.from_numpy(numpy.var(training_data.points, axis = 0) + eps).float()))
            self.labels_scale = torch.from_numpy(numpy.mean(training_data.labels, axis = 0)).float()
            self.labels_scale = torch.vstack((self.labels_scale, torch.from_numpy(numpy.var(training_data.labels, axis = 0) + eps).float()))

        def standardize(self, x, scale):
            '''
            standardize the given data x by scale data
            each dimension are standardized autonomously
            Parameters: x: 2D tensor   shape = (n, dim) 
                        input data
                        scale: 2D tensor   shape = (2, dim)
                               2 choices: points_scale, labels_scale
            Returns:    x2: 2D tensor   shape = (n, dim) 
                            output standardized data
            '''
            x_mean, x_var = scale[0,:], scale[1,:]
            return (x-x_mean)/torch.sqrt(x_var)
        def inv_standardize(self, x, scale):
            '''
            inverse standardize the given data x
            each dimension are inverse standardized autonomously
            Parameters: x: 2D tensor   shape = (n, dim) 
                        input data to be inverse_standardized
                        scale: 2D tensor   shape = (2, dim)
                               2 choices: points_scale, labels_scale
            Returns:    x2: 2D tensor   shape = (n, dim) 
                            output unstandardized data
            '''
            x_mean, x_var = scale[0,:], scale[1,:]
            return x*torch.sqrt(x_var)+x_mean
    def __init__(self, nnu, nnp) -> None:
        self.n_input = nnu.n_input
        self.n_output = nnp.n_output
        self.u_hidden = nnu.n_hidden
        self.p_hidden = nnp.n_hidden
        self.u_middle = nnu.n_output
        self.p_middle = nnp.n_input
        self.scale_data = nnu.scale_data
        self.scalep = nnp.scale_data
        self.nn = self.Net(self.n_input, self.n_output, self.u_hidden, self.p_hidden, self.u_middle, self.p_middle, self.scalep, self.scale_data)
    class Res(torch.nn.Module):
        def __init__(self, n_input, n_output) -> None:
            assert n_input==n_output
            super(NN_F.Res, self).__init__()
            self.layer = torch.nn.Sequential(
                torch.nn.Linear(n_input, n_output),
                torch.nn.ELU(inplace=True),
                torch.nn.Linear(n_output, n_output)
            )
            self.elu = torch.nn.ELU(inplace=True)
        def forward(self, input):
            output = self.layer(input) + input
            return self.elu(output)
    class Netu(torch.nn.Module):
        def __init__(self, n_input, n_hidden_0, n_output) -> None:
            super(NN_F.Netu, self).__init__()
            self.layers = torch.nn.Sequential(
                #torch.nn.BatchNorm1d(n_input),
                torch.nn.Linear(n_input, n_hidden_0),
                torch.nn.ELU(inplace=True),
                #torch.nn.Dropout(drop_prob_0),
                NN_F.Res(n_hidden_0, n_hidden_0),
                NN_F.Res(n_hidden_0, n_hidden_0),
                NN_F.Res(n_hidden_0, n_hidden_0),
                NN_F.Res(n_hidden_0, n_hidden_0),
                NN_F.Res(n_hidden_0, n_hidden_0),
                NN_F.Res(n_hidden_0, n_hidden_0),
                NN_F.Res(n_hidden_0, n_hidden_0),
                NN_F.Res(n_hidden_0, n_hidden_0),
                torch.nn.Linear(n_hidden_0, n_output)
            )
        def forward(self, input):
            return self.layers(input)
    class Netp(torch.nn.Module):
        def __init__(self, n_input, n_hidden_0, n_output) -> None:
            super(NN_F.Netp, self).__init__()
            self.layers = torch.nn.Sequential(
                #torch.nn.BatchNorm1d(n_input),
                torch.nn.Linear(n_input, n_hidden_0),
                torch.nn.ELU(inplace=True),
                #torch.nn.Dropout(drop_prob_0),
                NN_F.Res(n_hidden_0, n_hidden_0),
                NN_F.Res(n_hidden_0, n_hidden_0),
                NN_F.Res(n_hidden_0, n_hidden_0),
                NN_F.Res(n_hidden_0, n_hidden_0),
                #NN_F.Res(n_hidden_0, n_hidden_0),
                #NN_F.Res(n_hidden_0, n_hidden_0),
                #NN_F.Res(n_hidden_0, n_hidden_0),
                #NN_F.Res(n_hidden_0, n_hidden_0),
                #NN_F.Res(n_hidden_0, n_hidden_0),
                #NN_F.Res(n_hidden_0, n_hidden_0),
                torch.nn.Linear(n_hidden_0, n_output)
            )
        def forward(self, input):
            return self.layers(input)
    class Net(torch.nn.Module):
        def __init__(self, n_input, n_output, u_hidden, p_hidden, u_middle, p_middle, scalep, scale_data) -> None:
            super(NN_F.Net, self).__init__()
            self.layeru = torch.nn.Sequential(
                NN_F.Netu(n_input, u_hidden, u_middle)
            )
            self.layerp = torch.nn.Sequential(
                NN_F.Netp(p_middle, p_hidden, n_output)
            )
            self.scalep = scalep
            self.scale_data = scale_data
        def forward(self, input, *args):
            u_output = self.layeru(input)
            muori = self.scale_data.inv_standardize(input, self.scale_data.points_scale)
            middle = torch.column_stack((muori, u_output))
            middle = self.scalep.standardize(middle, self.scalep.points_scale)
            if args:
                for tdp in args:
                    BATCH1, BATCH2 = input.shape[0], tdp.shape[0]
                    predin = torch.cat((middle, tdp), dim=0)
                    p_output = self.layerp(predin)
                    p_output1, p_output2 = p_output[:BATCH1,:], p_output[BATCH1:,:]
                return p_output1, u_output, p_output2
            else:
                return self.layerp(middle), u_output
            
    def train_unsupervised(self, td, rb_data):
        # only unsupervised learning, corresponding to the second option of the paper
        learningrate = 0.00001
        derate = 0.8
        lrate = 2.e-6
        num_workers = 4
        BATCH_SIZE = 512
        EPOCH = 3000
        Mc = torch.from_numpy(rb_data.Mc.transpose()).float()
        x_train = self.scale_data.standardize(torch.from_numpy(
            td).float(), self.scale_data.points_scale)
        dataset = torch.utils.data.TensorDataset(x_train, torch.empty(td.shape[0]))
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=num_workers
        )
        ci = 0
        for child in self.nn.children():
            ci += 1
            if ci == 2:
                for param in child.parameters():
                    param.requires_grad = False
        def loss_f(input):
            res = torch.mm(input, Mc)
            loss = (res**2).mean()*Mc.shape[1]
            return loss
        def loss_mse(a,b):
            mse = torch.nn.MSELoss(reduction='sum')
            d = a.shape[0]
            return mse(a,b)/d
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.nn.parameters()), lr = learningrate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=derate, patience=300, verbose=False, 
            threshold=0.0001, threshold_mode='rel', cooldown=500, min_lr=0, eps=lrate
        )
        early_stopping = ES.EarlyStopping(patience=200)
        print(self.nn)
        sys.stdout.flush()
        start = time.time()
        losses = []
        for epoch in range(EPOCH):
            if epoch == 0:
                numpy.random.seed(99)
                mul = get_training_parameter(int(td.shape[0]/5))
                val_mu = self.scale_data.standardize(torch.from_numpy(
                        mul).float(), self.scale_data.points_scale)
                self.nn.eval()
                v_p, _ = self.nn(val_mu)
                loss = loss_f(v_p)
                print('      %d/     - Loss: %.8f' % (epoch+1, loss))
                sys.stdout.flush()
                early_stopping(loss, self.nn)
            self.nn.train()
            train_l_sum, batch_step = 0, 0
            for batch_x, _ in dataloader:
                b_x = batch_x
                optimizer.zero_grad()
                b_pred, _ = self.nn(b_x)
                loss = loss_f(b_pred)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                train_l_sum += loss.item()
                batch_step += 1
            train_l_sum /= batch_step
            losses.append(train_l_sum)
            if epoch%10 == 9 or epoch == 0 or epoch == 1:
                print('Epoch %d/%d - Loss: %.8f' % (epoch+1, EPOCH, train_l_sum))
                sys.stdout.flush()
            numpy.random.seed(100+epoch)
            mul = get_training_parameter(int(td.shape[0]/5))
            val_mu = self.scale_data.standardize(torch.from_numpy(
                    mul).float(), self.scale_data.points_scale)
            self.nn.eval()
            v_p, _ = self.nn(val_mu)
            loss = loss_f(v_p)
            if epoch%10 == 9 or epoch == 0 or epoch == 1:
                print('      %d/     - Loss: %.8f' % (epoch+1, loss))
                sys.stdout.flush()
            early_stopping(loss, self.nn)
            if early_stopping.early_stop:
                print("Early stopping at Epoch %d/%d - Loss: %.8f" % (epoch+1, EPOCH, train_l_sum))
                break
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'NNf_loss' + '.pickle'), 'wb') as file:
            pickle.dump(losses, file)
        plt.plot(losses, label='loss_f',lw=0.5)
        plt.legend(loc='best')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.savefig('loss_f.png')
        plt.close()
        end = time.time() - start
        print('The training time: %.5fs' % end)

        if not os.path.exists(os.path.join(os.path.abspath('.'), 'offline_data')):
            os.mkdir(os.path.join(os.path.abspath('.'), 'offline_data'))
        torch.save(self.nn.state_dict(), os.path.join(os.path.abspath('.'), 'offline_data', 'NNf_para.pkl'))
    def train(self, tdu, tdp, td, auxtdp, rb_data, fr=False):
        # semisupervised learning, corresponding to the first option in the paper
        learningrate = 0.00001
        derate = 0.95
        lrate = 1e-6
        num_workers = 4
        BATCH_SIZE = 40
        EPOCH = 3000
        Mc = torch.from_numpy(rb_data.Mc.transpose()).float()
        x_train = self.scale_data.standardize(torch.from_numpy(
            tdu.points).float(), self.scale_data.points_scale)
        u_train = torch.from_numpy(tdu.labels).float()
        p_train = torch.from_numpy(tdp.labels).float()
        unsup_train = self.scale_data.standardize(torch.from_numpy(
            td).float(), self.scale_data.points_scale)
        aux_x_train = self.scalep.standardize(torch.from_numpy(
            auxtdp.points).float(), self.scalep.points_scale)
        aux_p_train = torch.from_numpy(auxtdp.labels).float()
        datasetu = torch.utils.data.TensorDataset(x_train, u_train, p_train)
        train_size = int(0.8 * len(datasetu))
        valid_size = len(datasetu) - train_size
        train_datau, valid_datau = torch.utils.data.random_split(datasetu, [train_size, valid_size], generator=torch.manual_seed(0))
        datasetp = torch.utils.data.TensorDataset(unsup_train, torch.zeros(unsup_train.shape[0]))
        aux_datasetp = torch.utils.data.TensorDataset(aux_x_train, aux_p_train)
        aux_train_size = int(0.8 * len(aux_datasetp))
        aux_valid_size = len(aux_datasetp) - aux_train_size
        aux_train_datap, aux_valid_datap = torch.utils.data.random_split(aux_datasetp, [aux_train_size, aux_valid_size], generator=torch.manual_seed(0))
        dataloaderu = torch.utils.data.DataLoader(
            dataset=train_datau, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=num_workers
        )
        validloader = torch.utils.data.DataLoader(
            dataset=valid_datau, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=num_workers
        )
        dataloaderp = torch.utils.data.DataLoader(
            dataset=datasetp, batch_size=16*BATCH_SIZE, 
            shuffle=True, num_workers=num_workers
        )
        aux_dataloaderp = torch.utils.data.DataLoader(
            dataset=aux_train_datap, batch_size=4*BATCH_SIZE, 
            shuffle=True, num_workers=num_workers
        )
        aux_validloaderp = torch.utils.data.DataLoader(
            dataset=aux_valid_datap, batch_size=4*BATCH_SIZE, 
            shuffle=True, num_workers=num_workers
        )
        def loss_p(input):
            res = torch.mm(input, Mc)
            loss = (res**2).mean()*Mc.shape[1]
            return loss
        def loss_u(a,b):
            mse = torch.nn.MSELoss(reduction='sum')
            d = a.shape[0]
            return mse(a,b)/d
        optimizer = torch.optim.Adam(self.nn.parameters(), lr = learningrate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=derate, patience=300, verbose=False, 
            threshold=0.0001, threshold_mode='rel', cooldown=500, min_lr=0, eps=lrate
        )
        early_stopping = ES.EarlyStopping(patience=180)
        print(self.nn)
        if fr:
            i = 0
            for child in self.nn.children():
                if i == 1:
                    for param in child.parameters():
                        param.requires_grad = False
                i += 1
        else:
            i = 0
            for child in self.nn.children():
                if i == 1:
                    for param in child.parameters():
                        param.requires_grad = True
                i += 1
        sys.stdout.flush()
        start = time.time()
        losses = []
        for epoch in range(EPOCH):
            if epoch == 0:
                numpy.random.seed(99)
                mul = get_training_parameter(int(td.shape[0]/5))
                val_mu = self.scale_data.standardize(torch.from_numpy(
                        mul).float(), self.scale_data.points_scale)
                self.nn.eval()
                valid_l_sum, batch_step = 0, 0
                for v_x, v_u, v_p in iter(validloader):
                    p_pv, u_pv = self.nn(v_x)
                    loss = loss_u(u_pv, v_u) + loss_u(p_pv, v_p)
                    valid_l_sum += loss.item()
                    batch_step += 1
                valid_l_total = valid_l_sum / batch_step
                valid_l_sum, batch_step = 0, 0
                for v_x, v_p in iter(aux_validloaderp):
                    _, _, p_pv = self.nn(torch.broadcast_to(torch.tensor([0,0,0,1]).float(), (2,4)), v_x)
                    loss = loss_u(p_pv, v_p)
                    valid_l_sum += loss.item()
                    batch_step += 1
                valid_l_total2 = valid_l_sum / batch_step
                p_mu, _ = self.nn(val_mu)
                lp = loss_p(p_mu)
                early_stopping(valid_l_total, self.nn)
                if epoch%3 == 2 or epoch == 0 or epoch == 1:
                    print('-------%d-- Val Loss: %.8f    %.8f    %.8f    %.9f' % (epoch+1, 2*valid_l_total+valid_l_total2+10*lp, valid_l_total, valid_l_total2, lp))
                    sys.stdout.flush()
            self.nn.train()
            train_l_sum, batch_step = 0, 0
            dataloader_iterator = iter(dataloaderu)
            aux_dataloader_iterator = iter(aux_dataloaderp)
            for batch_uns, _ in dataloaderp:
                try:
                    batch_x, batch_u, batch_p = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(dataloaderu)
                    batch_x, batch_u, batch_p = next(dataloader_iterator)
                try:
                    aux_batch_x, aux_batch_p = next(aux_dataloader_iterator)
                except StopIteration:
                    aux_dataloader_iterator = iter(aux_dataloaderp)
                    aux_batch_x, aux_batch_p = next(aux_dataloader_iterator)
                b_x = batch_x
                b_u = batch_u
                b_p = batch_p
                b_uns = batch_uns
                aux_b_x = aux_batch_x
                aux_b_p = aux_batch_p
                BATCH0 = b_x.shape[0]
                _x = torch.cat((b_x, b_uns), dim=0)
                optimizer.zero_grad()
                p_pred, u_pred, aux_p_pred = self.nn(_x, aux_b_x)
                uns_pred = p_pred[BATCH0:,:]
                p_pred = p_pred[:BATCH0,:]
                u_pred = u_pred[:BATCH0,:]
                loss = 3*loss_u(u_pred, b_u) + 10*loss_p(uns_pred) + 2*loss_u(p_pred, b_p) + loss_u(aux_p_pred, aux_b_p)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                train_l_sum += loss.item()
                batch_step += 1
            train_l_sum /= batch_step
            losses.append(train_l_sum)
            if epoch%3 == 2 or epoch == 0 or epoch == 1:
                print('Epoch %d/%d - Loss: %.8f' % (epoch+1, EPOCH, train_l_sum))
                sys.stdout.flush()
            numpy.random.seed(100+epoch)
            mul = get_training_parameter(int(td.shape[0]/5))
            val_mu = self.scale_data.standardize(torch.from_numpy(
                    mul).float(), self.scale_data.points_scale)
            self.nn.eval()
            valid_l_sum, batch_step = 0, 0
            for v_x, v_u, v_p in iter(validloader):
                p_pv, u_pv = self.nn(v_x)
                loss = loss_u(u_pv, v_u) + loss_u(p_pv, v_p)
                valid_l_sum += loss.item()
                batch_step += 1
            valid_l_total = valid_l_sum / batch_step
            valid_l_sum, batch_step = 0, 0
            for v_x, v_p in iter(aux_validloaderp):
                _, _, p_pv = self.nn(torch.broadcast_to(torch.tensor([0,0,0,1]).float(), (2,4)), v_x)
                loss = loss_u(p_pv, v_p)
                valid_l_sum += loss.item()
                batch_step += 1
            valid_l_total2 = valid_l_sum / batch_step
            p_mu, _ = self.nn(val_mu)
            lp = loss_p(p_mu)
            early_stopping(valid_l_total, self.nn)
            if epoch%3 == 2 or epoch == 0 or epoch == 1:
                print('-------%d-- Val Loss: %.8f    %.8f    %.8f    %.9f' % (epoch+1, 2*valid_l_total+valid_l_total2+10*lp, valid_l_total, valid_l_total2, lp))
                sys.stdout.flush()
            if early_stopping.early_stop:
                print("Early stopping at Epoch %d/%d - Loss: %.8f" % (epoch+1, EPOCH, train_l_sum))
                break
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'NNfs_loss' + '.pickle'), 'wb') as file:
            pickle.dump(losses, file)
        plt.plot(losses, label='loss_fs',lw=0.5)
        plt.legend(loc='best')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.savefig('loss_fs.png')
        plt.close()
        end = time.time() - start
        print('The training time: %.5fs' % end)

        if not os.path.exists(os.path.join(os.path.abspath('.'), 'offline_data')):
            os.mkdir(os.path.join(os.path.abspath('.'), 'offline_data'))
        torch.save(self.nn.state_dict(), os.path.join(os.path.abspath('.'), 'offline_data', 'NNf_para.pkl'))
    def load_para(self):
        self.nn = NN_F.Net(self.n_input, self.n_output, self.u_hidden, self.p_hidden, self.u_middle, self.p_middle, self.scalep, self.scale_data)
        self.nn.load_state_dict(torch.load(os.path.join(os.path.abspath('.'), 'offline_data', 'NNf_para.pkl')))
        self.nn.eval()
    
    def predict(self, x):
        '''
        the prediction functions
        Parameters: x: 1D or 2D array   shape = (N_para,) (n, N_para)
                        the point to evaluate the prediction
        Returns:    y: 1D or 2D array   shape = (N_p,) (n, N_p)
                        the predicted label for regression of p at point x
                    u: 1D or 2D array   shape = (N_u,) (n, N_u)
                        the predicted label for regression of u at point x
        '''
        self.nn.eval()
        xi = x
        if xi.ndim == 1:
            x = x.reshape(1,-1)
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            x = self.scale_data.standardize(x, self.scale_data.points_scale)
            y, u = self.nn(x)
            #y = self.scale_data.inv_standardize(y, self.scale_data.q_labels_scale)
        y, u = y.numpy(), u.numpy()
        if xi.ndim == 1:
            y = y.flatten()
            u = u.flatten()
        return y, u
    def Jacobian(self, x):
        '''
        get the Jacobian at x
        Parameters: x: 1D array   shape = (N_para,)
                       the point to evaluate the prediction and Jacobian
        Returns:    y: 1D array   shape = (N_p,)
                       the predicted label for regression of p at point x
                    J: 2D array   shape = (N_p, N_para)
                       the Jacobian matrix for regression of p at point x
        '''
        self.nn.eval()
        n_x = self.scale_data.points_scale.shape[1]
        n_y = self.scalep.labels_scale.shape[1]
        J = torch.empty(n_y, n_x)
        x = torch.from_numpy(x.reshape(1,-1)).float()
        x.requires_grad_(True)
        xm = self.scale_data.standardize(x, self.scale_data.points_scale)
        y, u = self.nn(xm)
        #y = self.scale_data.inv_standardize(ym, self.scale_data.q_labels_scale)
        for i in range(n_y):
            y[0,i].backward(retain_graph=True)
            t = x.grad
            J[i,:] = t[0,:]
            x.grad.data.zero_()
        y = y.detach().numpy().flatten()
        return y, J.numpy()

    def load_from_up(self, nnu, nnp):
        # load parameter from subnetworks nnu and nnp, requires that nnf should be constituted by nnu and nnp
        udict = nnu.nn.state_dict().copy()
        pdict = nnp.nn.state_dict().copy()
        fdict = self.nn.state_dict().copy()
        ulist = list(udict.keys())
        plist = list(pdict.keys())
        flist = list(fdict.keys())
        Nu = len(ulist)
        Np = len(plist)
        Nf = len(flist)
        assert Nf == Nu+Np
        for i in range(Nu):
            fdict[flist[i]] = udict[ulist[i]]
        for i in range(Np):
            fdict[flist[i+Nu]] = pdict[plist[i]]
        self.nn.load_state_dict(fdict)

def offline(mesh_data, sample_size, tol):
    '''
    get the random sample of parameter
    Parameters: mesh_data: MeshData()
                        the mesh, the cellmarkers and the facetmarkers
                ss: int
                    the sample size for the parameter
                tol: list of float   0.0<tol<1.0
                     average projection error tolerace, the POD error tolerance = tol**2
    Returns:    mu: 2D array   shape = (ss, N_para)
    '''   
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    N_processes = comm.Get_size()
    

    # implementation of POD and generate the reduced basis, store in RB_Data() class and POD_vector()
    '''
    # get the snapshots
    start = time.time()
    numpy.random.seed(0)
    mul = get_training_parameter(sample_size)
    up_m = full_order_solution_m(mesh_data, mul)
    end = time.time() - start
    print('The computational time: %.5f s' % end)
    if my_rank == 0:
        u_m, p_m = up_m
        if not os.path.exists(os.path.join(os.path.abspath('.'), 'offline_data')):
            os.mkdir(os.path.join(os.path.abspath('.'), 'offline_data'))
        print("********** starting the POD training **********")
        sys.stdout.flush()
        # POD for unknown quantity u
        M_u, M_p = get_mass_matrix(mesh_data)
        print("*****first POD training:")
        sys.stdout.flush()
        start = time.time()
        u_list = POD_l(u_m, M_u, tol)
        end = time.time() - start
        print('The POD time: %.5f s' % end)
        mu_ml, p_ml = get_other_p(mesh_data, u_m)
        p_m_o = numpy.vstack((p_m, p_ml))
    else:
        get_other_p(mesh_data, None)
    if my_rank == 0:
        print("*****second POD training:")
        sys.stdout.flush()
        # POD for derived quantity P
        start = time.time()
        p_list = POD_l(p_m_o, M_p, tol)
        end = time.time() - start
        print('The POD time: %.5f s' % end)
        sys.stdout.flush()
        UU = fenics.VectorFunctionSpace(mesh_data.mesh, 'CG', 2, dim=2)
        UUU = fenics.TensorFunctionSpace(mesh_data.mesh, 'DG', 1, shape=(2,2))
        p = fenics.TrialFunction(UUU)
        u_ = fenics.TestFunction(UU)
        mc = fenics.inner(p, fenics.grad(u_))*fenics.dx
        M1 = fenics.assemble(mc)
        row,col,val = fenics.as_backend_type(M1).data()
        M1 = sp.csr_matrix((val, col, row))
        for i in range(len(u_list)):
            uli = sp.csr_matrix(u_list[i][0].transpose())
            m1t = sp.csr_matrix.dot(uli, M1).toarray()
            Mc = numpy.dot(m1t, p_list[i][0])
            pb = fenics.Function(UUU)
            p_tensor = []
            for j in range(p_list[i][1]):
                pb.vector().set_local(p_list[i][0][:,j])
                pb0, pb1, pb2, pb3 = pb.split()
                p0, p1 = fenics.assemble(pb0*fenics.dx), fenics.assemble(pb1*fenics.dx)
                p2, p3 = fenics.assemble(pb2*fenics.dx), fenics.assemble(pb3*fenics.dx)
                pt = numpy.array([[p0, p1], [p2, p3]])
                p_tensor.append(pt)
            rb_data = RB_Data(u_list[i], p_list[i], Mc, p_tensor)
            with open(os.path.join(os.path.abspath('.'), 'offline_data', 'RB_%d'%i + '.pickle'), 'wb') as file:
                pickle.dump(rb_data, file)
            p_rb = rb_data.project_p_rb(p_m)
            u_rb = rb_data.project_u_rb(u_m)    
            pod_vector = POD_vector(u_rb, p_rb, mul)
            with open(os.path.join(os.path.abspath('.'), 'offline_data', 'POD_vector_%d'%i + '.pickle'), 'wb') as file:
                pickle.dump(pod_vector, file)
            print("Tol:  %.8g     U:  %d     P:  %d" %(tol[i], u_list[i][1], p_list[i][1]))

        # get the initial training data at tol_ind, store in Training_Data() class
        for tol_ind in range(len(u_list)):
            with open(os.path.join(os.path.abspath('.'), 'offline_data', 'POD_vector_%d'%tol_ind + '.pickle'), 'rb') as file:
                pod_vector = pickle.load(file)
            training_data_u = Training_Data(pod_vector.mu, pod_vector.u_rb)
            assert training_data_u.points.shape[0] == training_data_u.labels.shape[0]
            with open(os.path.join(os.path.abspath('.'), 'offline_data', 'TDu_%d'%tol_ind + '.pickle'), 'wb') as file:
                pickle.dump(training_data_u, file)
            tp_rb = numpy.column_stack((pod_vector.mu, pod_vector.u_rb))
            training_data_p = Training_Data(tp_rb, pod_vector.p_rb)
            assert training_data_p.points.shape[0] == training_data_p.labels.shape[0]            
            assert training_data_p.points.shape[0] == training_data_u.labels.shape[0]
            with open(os.path.join(os.path.abspath('.'), 'offline_data', 'TDp_%d'%tol_ind + '.pickle'), 'wb') as file:
                pickle.dump(training_data_p, file)
        for tol_ind in range(len(u_list)):
            with open(os.path.join(os.path.abspath('.'), 'offline_data', 'RB_%d'%tol_ind + '.pickle'), 'rb') as file:
                rb_data = pickle.load(file)
            with open(os.path.join(os.path.abspath('.'), 'offline_data', 'POD_vector_%d'%tol_ind + '.pickle'), 'rb') as file:
                pod_vector = pickle.load(file)
            tp_rb = numpy.vstack((pod_vector.u_rb, pod_vector.u_rb))
            tp_mu = numpy.vstack((pod_vector.mu, mu_ml))
            tp_rb = numpy.column_stack((tp_mu, tp_rb))
            p_rb_o = rb_data.project_p_rb(p_m_o)
            aux_training_data_p = Training_Data(tp_rb, p_rb_o)
            assert aux_training_data_p.points.shape[0] == aux_training_data_p.labels.shape[0]
            with open(os.path.join(os.path.abspath('.'), 'offline_data', 'TDp_%d_aux'%tol_ind + '.pickle'), 'wb') as file:
               pickle.dump(aux_training_data_p, file)
    '''  
    # Generation the test data for validate the performance
    ''' 
    tol_ind = 3
    valid_size = 300
    start = time.time()
    # get the snapshots
    numpy.random.seed(10)
    mul = get_training_parameter(valid_size)
    up_m = full_order_solution_m(mesh_data, mul)
    end = time.time() - start
    print('The computational time: %.5f s' % end)
    if my_rank == 0:
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'RB_%d'%tol_ind + '.pickle'), 'rb') as file:
            rb_data = pickle.load(file)
        u_m, p_m = up_m
        u_rb, p_rb = rb_data.project_u_rb(u_m), rb_data.project_p_rb(p_m)
        up_rb = (mul, u_rb, p_rb)
        up_m = (mul, u_m, p_m)
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'Validation' + '.pickle'), 'wb') as file:
            pickle.dump(up_m, file)
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'Validation_rb_%d'%tol_ind + '.pickle'), 'wb') as file:
            pickle.dump(up_rb, file)
    '''  
    
    # implementation of TTDN training
    tol_ind = 3
    ''''''
    if my_rank == 0:
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'RB_%d'%tol_ind + '.pickle'), 'rb') as file:
            rb_data = pickle.load(file)
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'TDu_%d'%tol_ind + '.pickle'), 'rb') as file:
            training_data_u = pickle.load(file)
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'TDp_%d'%tol_ind + '.pickle'), 'rb') as file:
            training_data_p = pickle.load(file)
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'TDp_%d_aux'%tol_ind + '.pickle'), 'rb') as file:
            aux_training_data_p = pickle.load(file)
        nnu, nnp = NN_U(60), NN_P(100)


        trainux = training_data_u.points[:1500,:]
        trainuy = training_data_u.labels[:1500,:]
        trainpx = training_data_p.points[:1500,:]
        trainpy = training_data_p.labels[:1500,:]
        training_data_u0 = Training_Data(trainux, trainuy)
        training_data_p0 = Training_Data(trainpx, trainpy)
        auxx = aux_training_data_p.points[1500:3000,:]
        auxx = numpy.vstack((trainpx, auxx))
        auxy = aux_training_data_p.labels[1500:3000,:]
        auxy = numpy.vstack((trainpy, auxy))
        aux_training_data_p0 = Training_Data(auxx, auxy)


        nnu.get_scale(training_data_u0)
        nnp.get_scale(aux_training_data_p0)

        nnf = NN_F(nnu, nnp)
        print(nnu.nn)
        print(nnp.nn)
        print(nnf.nn)
        # pretrain stage for NNU
        nnu.train(training_data_u0, retrain=False)
        

    aux_ss = 15000
    with open(os.path.join(os.path.abspath('.'), 'offline_data', 'GB_%d'%tol_ind + '.pickle'), 'rb') as file:
        GB = pickle.load(file)
    start = time.time()
    aux = get_unsupervised_training_data(mesh_data, aux_ss, rb_data, nnu)
    end = time.time() - start
    print('The computational time: %.5f s' % end)    
    
    if my_rank == 0:
        # get constitutive data
        aux_tp, aux_tl = aux
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'TDp_%d_aux'%tol_ind + '.pickle'), 'rb') as file:
            aux_training_data_p = pickle.load(file)
        new_tpq = numpy.vstack((aux_training_data_p0.points, aux_tp))
        new_tlq = numpy.vstack((aux_training_data_p0.labels, aux_tl))
        aux_training_data_p0 = Training_Data(new_tpq, new_tlq)
        assert aux_training_data_p.points.shape[0] == aux_training_data_p.labels.shape[0]
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'TDp_%d_aux_'%tol_ind + '.pickle'), 'wb') as file:
            pickle.dump(aux_training_data_p0, file)
        ''''''
        nnp.get_scale(training_data_p0)
        nnp.load_para()
        with open(os.path.join(os.path.abspath('.'), 'offline_data', 'TDp_%d_aux_'%tol_ind + '.pickle'), 'rb') as file:
            aux_training_data_p0 = pickle.load(file)
        new_tpq = numpy.vstack((training_data_p0.points, aux_training_data_p0.points[1500:7500,:]))
        new_tlq = numpy.vstack((training_data_p0.labels, aux_training_data_p0.labels[1500:7500,:]))
        aux_training_data_p01 = Training_Data(new_tpq, new_tlq)
        new_tpq = aux_training_data_p0.points[:7500,:]
        new_tlq = aux_training_data_p0.labels[:7500,:]
        aux_training_data_p0 = Training_Data(new_tpq, new_tlq)
        print(aux_training_data_p0.points.shape[0], aux_training_data_p0.labels.shape[0])
        
        # pretrain NN_P
        nnp.train(aux_training_data_p0, retrain=False)
        nnp.load_para()
        nnp.train(aux_training_data_p01, retrain=True)
        
        
        ''''''
        smu = 10000
        nnf = NN_F(nnu, nnp)
        nnu.load_para()
        #nnf.load_para()
        #nnu.load_from_f(nnp, nnf)
        #nnf.load_from_up(nnu, nnp)
        numpy.random.seed(5)
        tdmu = get_training_parameter(smu)
        nnf.load_para()
        #nnf.train_unsupervised(tdmu, rb_data)
        #nnf.train(training_data_u0, training_data_p0, tdmu, aux_training_data_p01, rb_data, fr=True)
        nnf.train(training_data_u0, training_data_p0, tdmu, aux_training_data_p01, rb_data, fr=False)

def online():
    tol_ind = 3
    with open(os.path.join(os.path.abspath('.'), 'offline_data', 'RB_%d'%tol_ind + '.pickle'), 'rb') as file:
        rb_data = pickle.load(file)
    nnu, nnp = NN_U(60), NN_P(100)
    nnf = NN_F(nnu, nnp)
    nnf.load_para()
            
    mul = get_training_parameter(100)
    
    # reduced p and u by TTDN
    prep, preu = nnf.predict(mul)
    
    # stiffness tensor by TTDN
    KU = numpy.zeros(shape=(mul.shape[0],3,2,2))
    for i in range(mul.shape[0]):
        _, J2 = nnf.Jacobian(mul[i,:])
        for j in range(3):
            for k in range(J2.shape[0]):
                KU[i,j,:,:] += J2[k,j]*rb_data.p_tensor[k]

def get_training_parameter(ss):
    '''
    get the random sample of parameter
    Parameters: sample_size: int
                             the sample size for the parameter
    Returns:    mu: 2D array   shape = (ss, N_para)
    '''  
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    N_processes = comm.Get_size() 
    if my_rank == 0:
        Fml = -0.05
        Fmh = 0.05
        C2l = 0.2
        C2h = 5
        mu0 = numpy.random.uniform(low=Fml, high=Fmh, size=ss)
        mu1 = numpy.random.uniform(low=Fml, high=Fmh, size=ss)
        mu2 = numpy.random.uniform(low=Fml, high=Fmh, size=ss)
        mu3 = numpy.random.uniform(low=C2l, high=C2h, size=ss)
        mu = numpy.column_stack((mu0, mu1, mu2, mu3))
        return mu
    else:
        return numpy.array([])

def full_order_solution_m(mesh_data, mul):
    '''
    solve the microscale full order solution of mu
    parallel version
    Parameters: mesh_data: MeshData()
                        the mesh, the cellmarkers and the facetmarkers
                mul: 2D array   shape = (n, N_para)
                    the parameters to be solve
    Returns:    u_matrix: 2D array   shape = (n, N_full_u)
                        the collection of solutions u
                p_matrix: 2D array   shape = (n, N_full_p)
                        the collection of flux p
    '''
    fenics.parameters["form_compiler"]["cpp_optimize"] = True
    ffc_options = {
        "optimize": True, "eliminate_zeros": True, \
        "precompute_basis_const": True, "precompute_ip_const": True
    }
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    N_processes = comm.Get_size()

    # Create periodic boundary condition
    class PeriodicBoundary(fenics.SubDomain):
        def __init__(self, length = 1.0):
            fenics.SubDomain.__init__(self)
            self.length = length
        def inside(self, x, on_boundary):
            return bool(
                (fenics.near(x[0], 0) or fenics.near(x[1], 0)) and (
                    not ((fenics.near(x[0], 0) and fenics.near(x[1], self.length)) or
                    (fenics.near(x[0], self.length) and fenics.near(x[1], 0)))
                ) and on_boundary
            )
        def map(self, x, y):
            L = self.length
            if fenics.near(x[0], L) and fenics.near(x[1], L):
                y[0] = x[0] - L
                y[1] = x[1] - L
            elif fenics.near(x[0], L):
                y[0] = x[0] - L
                y[1] = x[1]
            elif fenics.near(x[1], L):  
                y[0] = x[0]
                y[1] = x[1] - L
            else:
                y[0] = -10000
                y[1] = -10000
    pbc = PeriodicBoundary()
    UUU = fenics.TensorFunctionSpace(mesh_data.mesh, 'DG', 1, shape=(2,2))
    ElemUU = fenics.VectorElement('CG', mesh_data.mesh.ufl_cell(), 2, dim=2)
    ElemLM = fenics.VectorElement('R', mesh_data.mesh.ufl_cell(), 0, dim=2)
    LU = fenics.FunctionSpace(mesh_data.mesh, fenics.MixedElement([ElemUU, ElemLM]), constrained_domain=pbc)
    U = fenics.VectorFunctionSpace(mesh_data.mesh, 'CG', 2, dim=2)
    '''
    def solve(mu):
        u, lamb = fenics.split(ulamb)
        FM = fenics.Constant(numpy.array([[mu[0]+1, mu[2]], [mu[2], mu[1]+1]]))
        Fm = FM + inveps*fenics.grad(u)
        C = Fm.T*Fm
        Ic = fenics.tr(C)
        J = fenics.det(Fm)
        psi = 1*(Ic-3-2*fenics.ln(J)) + mu[3]*(J-1)**2
        Pi = psi*fenics.dx + fenics.inner(u, lamb)*fenics.dx
        F = fenics.derivative(Pi, ulamb, ulamb_)
        J = fenics.derivative(F, ulamb, dulamb)
        #fenics.solve(F == 0, ulamb, J=J, form_compiler_parameters=ffc_options)
        fenics.solve(F == 0, ulamb, J=J, solver_parameters={"newton_solver": {"relative_tolerance": 1.e-12}})
        u = ulamb.split(deepcopy=True)[0]
        u.rename("displacement", "displacement")
        #u.vector()[:] *= inveps
        #file = fenics.File(fenics.MPI.comm_self, os.path.join(os.path.abspath('.'), 'fig_example', 'displacement%.3f.pvd'%mu[3]))
        #file << u
        #u.vector()[:] *= varepsilon
        def get_stress():
            Fm = FM + inveps*fenics.grad(u)
            Fm = fenics.variable(Fm)
            C = Fm.T*Fm
            Ic = fenics.tr(C)
            J = fenics.det(Fm)
            psi = 1*(Ic-3-2*fenics.ln(J)) + mu[3]*(J-1)**2
            dF = fenics.diff(psi, Fm)
            p = fenics.project (dF, UUU)
            return p
        p = get_stress()
        p.rename("stress", "stress")
        #file = fenics.File(fenics.MPI.comm_self, os.path.join(os.path.abspath('.'), 'fig_example', 'stress%.3f.pvd'%mu[3]))
        #file << p
        return u.vector().get_local(), p.vector().get_local()
    '''
    def solve(mu, *args):
        ulamb_ = fenics.TestFunction(LU)
        dulamb = fenics.TrialFunction(LU)
        ulamb = fenics.Function(LU) 
        if args:
            rou = args[0]
            if rou == 2:
                mus1 = mu.copy()
                mus1[0], mus1[0], mus1[0] = mus1[0]*2/3, mus1[0]*2/3, mus1[0]*2/3
                _, _, ulamb_ini = solve(mus1, 1)
                ulamb.vector().set_local(ulamb_ini)    
            elif rou == 1:
                mus0 = mu.copy()
                mus0[0], mus0[0], mus0[0] = mus0[0]*0.5, mus0[0]*0.5, mus0[0]*0.5
                _, _,ulamb_ini = solve(mus0, 0)
                ulamb.vector().set_local(ulamb_ini)
            else:
                ulamb_ini = numpy.zeros(LU.dim())
                ulamb.vector().set_local(ulamb_ini)
        else:
            mus2 = mu.copy()
            mus2[0], mus2[0], mus2[0] = mus2[0]*3/4, mus2[0]*3/4, mus2[0]*3/4
            _, _, ulamb_ini = solve(mus2, 2)
            ulamb.vector().set_local(ulamb_ini)
        u, lamb = fenics.split(ulamb)
        FM = fenics.Constant(numpy.array([[mu[0]+1, mu[2]], [mu[2], mu[1]+1]]))
        Fm = FM + inveps*fenics.grad(u)
        C = Fm.T*Fm
        Ic = fenics.tr(C)
        J = fenics.det(Fm)
        psi = 1*(Ic-3-2*fenics.ln(J)) + mu[3]*(J-1)**2
        Pi = psi*fenics.dx + fenics.inner(u, lamb)*fenics.dx
        F = fenics.derivative(Pi, ulamb, ulamb_)
        J = fenics.derivative(F, ulamb, dulamb)
        #fenics.solve(F == 0, ulamb, J=J, form_compiler_parameters=ffc_options)
        #fenics.solve(F == 0, ulamb, J=J, solver_parameters={"newton_solver": {"relative_tolerance": 1.e-12}})
        fenics.solve(F == 0, ulamb, J=J)
        u = ulamb.split(deepcopy=True)[0]
        u.rename("displacement", "displacement")
        #u.vector()[:] *= inveps
        #file = fenics.File(fenics.MPI.comm_self, os.path.join(os.path.abspath('.'), 'fig_example', 'displacement%.3f.pvd'%mu[3]))
        #file << u
        #u.vector()[:] *= varepsilon
        def get_stress():
            uv = u.vector().get_local().copy()
            ui = fenics.Function(U)
            ui.vector().set_local(uv)
            Fm = FM + inveps*fenics.grad(ui)
            Fm = fenics.variable(Fm)
            C = Fm.T*Fm
            Ic = fenics.tr(C)
            J = fenics.det(Fm)
            psi = 1*(Ic-3-2*fenics.ln(J)) + mu[3]*(J-1)**2
            dF = fenics.diff(psi, Fm)
            p = fenics.project(dF, UUU)
            return p
        p = get_stress()
        p.rename("stress", "stress")
        #file = fenics.File(fenics.MPI.comm_self, os.path.join(os.path.abspath('.'), 'fig_example', 'stress%.3f.pvd'%mu[3]))
        #file << p
        return u.vector().get_local(), p.vector().get_local(), ulamb.vector().get_local()
    def get_stress_p(mu, u_i):
        u = fenics.Function(U)
        u.vector().set_local(u_i)
        FM = fenics.Constant(numpy.array([[mu[0]+1, mu[2]], [mu[2], mu[1]+1]]))
        Fm = FM + inveps*fenics.grad(u)
        Fm = fenics.variable(Fm)
        C = Fm.T*Fm
        Ic = fenics.tr(C)
        J = fenics.det(Fm)
        psi = 1*(Ic-3-2*fenics.ln(J)) + mu[3]*(J-1)**2
        dF = fenics.diff(psi, Fm)
        p = fenics.project (dF, UUU)
        p.rename("stress", "stress")
        #file = fenics.File(fenics.MPI.comm_self, os.path.join(os.path.abspath('.'), 'fig_example', 'stress%.3f.pvd'%mu[3]))
        #file << p
        return p.vector().get_local()

    if my_rank == 0:
        def divide2chunk(n_task, n_process):
            chunksize = n_task // n_process + 1
            rest = n_task % n_process
            task_list = [chunksize]*rest
            task_list.extend([chunksize-1]*(n_process-rest))
            for i in range(1, n_process):
                task_list[i] += task_list[i-1]
            return task_list
        n_task = mul.shape[0]
        task_list = divide2chunk(n_task, N_processes)

        muc = mul[:task_list[0], :]
        print("Process %d distributes parameters..." %my_rank)
        sys.stdout.flush()
        for p in range(1, N_processes):
            data_send = mul[task_list[p-1]:task_list[p], :]
            comm.send(data_send, dest=p)
        print("Process %d done sending!" %my_rank)
        sys.stdout.flush()
    else:
        muc = comm.recv(source=0)
        print("Process %d received the parameters!" %my_rank)
        sys.stdout.flush()
    
    n_task_p = muc.shape[0]
    i = 0
    for mu in muc:
        i += 1
        print("********** Process %d is getting the %d/%d solution! **********" % (my_rank, i, n_task_p))
        print("Training parameter is          ", mu)
        sys.stdout.flush()
        u_i, p_i, _ = solve(mu)
        #p_i = get_stress_p(mu, u_i)

        if i == 1:
            u_matrix = u_i.reshape(1,-1)
            p_matrix = p_i.reshape(1,-1)
        else:
            u_matrix = numpy.append(u_matrix, u_i.reshape(1,-1), axis=0)
            p_matrix = numpy.append(p_matrix, p_i.reshape(1,-1), axis=0)
    print("Process %d finished the computation!" %my_rank)
    sys.stdout.flush()

    if my_rank == 0:
        for p in range(1, N_processes):
            u_matrix_p, p_matrix_p = comm.recv(source=p)
            u_matrix = numpy.append(u_matrix, u_matrix_p, axis=0)
            p_matrix = numpy.append(p_matrix, p_matrix_p, axis=0)
        print("Process %d done!" %my_rank)
        sys.stdout.flush()
        return u_matrix, p_matrix
    else:
        comm.send((u_matrix, p_matrix), dest=0)
        print("Process %d done!" %my_rank)
        sys.stdout.flush()

def get_unsupervised_training_data(mesh_data, sample_size, rb_data, nnu):
    '''
    get auxillary training data (constitutive data) for p
    Parameters: mesh_data: MeshData()
                           the mesh, the cellmarkers and the facetmarkers
                sample_size: int
                             the sample size for the parameter, can be large since it doesn't need the FO solutions
                rb_data: RB_Data()
                         the reduced basis of u and p
                nnu: NN()
                     neural network of u
    Returns:    tpl: 2D array   shape=(sample_size, N_para+N_u)
                     auxillary training points for p
                tpl: 2D array   shape=(sample_size, N_p)
                     auxillary training labels for p
    '''
    
    #p_ff = rb_data.p.Basis[:,0]

    UUU = fenics.TensorFunctionSpace(mesh_data.mesh, 'DG', 1, shape=(2,2))
    U = fenics.VectorFunctionSpace(mesh_data.mesh, 'CG', 2, dim=2)
    u = fenics.Function(U)
    #pvar = fenics.Function(UUU)
    mu3 = fenics.Constant(1)
    Fm = fenics.Function(UUU)
    #Fm1 = fenics.Function(UUU)
    #Fm2 = fenics.Function(UUU)
    FM = fenics.Constant(numpy.array([[1,0],[0,1]]))
    ieps = fenics.Constant(inveps)
    Fmv = fenics.variable(Fm)
    C = Fmv.T*Fmv
    Ic = fenics.tr(C)
    J = fenics.det(Fmv)
    psi = Ic-3-2*fenics.ln(J) + mu3*(J-1)**2
    dF = fenics.diff(psi, Fmv)

    '''
    for i in range(rb_data.u.N):
        u.vector().set_local(rb_data.u.Basis[:,i])
        gu = fenics.project(fenics.grad(u), UUU)
        if i == 0:
            GB = gu.vector().get_local().reshape(1,-1).copy()
        else:
            GB = numpy.append(GB, gu.vector().get_local().reshape(1,-1), axis=0)
    with open(os.path.join(os.path.abspath('.'), 'offline_data', 'GB_3' + '.pickle'), 'wb') as file:
        pickle.dump(GB, file)
    '''
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    N_processes = comm.Get_size()
    if my_rank == 0:
        def divide2chunk(n_task, n_process):
            chunksize = n_task // n_process + 1
            rest = n_task % n_process
            task_list = [chunksize]*rest
            task_list.extend([chunksize-1]*(n_process-rest))
            for i in range(1, n_process):
                task_list[i] += task_list[i-1]
            return task_list
        n_task = sample_size
        task_list = divide2chunk(n_task, N_processes)

        numpy.random.seed(2)
        mul = get_training_parameter(sample_size)
        print("Process %d distributes parameters..." %my_rank)
        sys.stdout.flush()
        for p in range(1, N_processes):
            data_send = mul[task_list[p-1]:task_list[p],:]
            comm.send(data_send, dest=p)
        print("Process %d done sending!" %my_rank)
        sys.stdout.flush()
        n_task_p = task_list[0]
        muc = mul[:task_list[0],:]
    else:
        muc = comm.recv(source=0)
        n_task_p = muc.shape[0]
        print("Process %d received the parameters!" %my_rank)
        sys.stdout.flush()

    uc = nnu.predict(muc)
    tpl = numpy.column_stack((muc, uc))

    tll = numpy.empty(shape=(n_task_p, rb_data.p.N))
    for i in range(n_task_p):
        mu = muc[i,:]
        u_f = rb_data.return_u_full_order(uc[i,:])
        u.vector().set_local(u_f)
        mu3.assign(fenics.Constant(mu[3]))
        FM.assign(fenics.Constant(numpy.array([[mu[0]+1, mu[2]], [mu[2], mu[1]+1]])))
        #Fm1.assign(fenics.project(FM, UUU))
        #Fm2.assign(fenics.project(fenics.grad(u), UUU))
        #Fm.assign(Fm1 + ieps*Fm2)
        Fm.assign(fenics.project(FM+ieps*fenics.grad(u), UUU))
        #Fmv = fenics.variable(Fm)
        #C = Fmv.T*Fmv
        #Ic = fenics.tr(C)
        #J = fenics.det(Fmv)
        #psi = Ic-3-2*fenics.ln(J) + mu3*(J-1)**2
        #dF = fenics.diff(psi, Fmv)
        #pvar.assign(fenics.project (dF, UUU))
        p_f = fenics.project (dF, UUU).vector().get_local()
        tll[i,:] = rb_data.project_p_rb(p_f)
        #if i%100 == 99:
        #    print('Process %d, finish %d/%d' % (my_rank ,i+1, n_task_p))
        #    sys.stdout.flush()
    print("Process %d finished the computation!" %my_rank)
    sys.stdout.flush()

    if my_rank == 0:
        for p in range(1, N_processes):
            tp_p, tl_p = comm.recv(source=p)
            tpl = numpy.append(tpl, tp_p, axis=0)
            tll = numpy.append(tll, tl_p, axis=0)
        print("Process %d done!" %my_rank)
        sys.stdout.flush()
        return tpl, tll
    else:
        comm.send((tpl, tll), dest=0)
        print("Process %d done!" %my_rank)
        sys.stdout.flush()

def get_p_label(mesh_data, rb_data, mul, ul):
    '''
    an implementation of the rb constitutive law
    Parameters: mesh_data: MeshData()
                           the mesh, the cellmarkers and the facetmarkers
                rb_data: RB_Data()
                         the reduced basis of u and p
                mul: 2D array   shape=(sample_size, N_para)
                     list of parameter to compute
                ul: 2D array   shape=(sample_size, N_u)
                     list of rb u to compute     
    Returns:    tll: 2D array   shape=(sample_size, N_p)
                     auxillary training labels for p
    '''
    UUU = fenics.TensorFunctionSpace(mesh_data.mesh, 'DG', 1, shape=(2,2))
    U = fenics.VectorFunctionSpace(mesh_data.mesh, 'CG', 2, dim=2)
    u = fenics.Function(U)
    mu3 = fenics.Constant(1)
    Fm = fenics.Function(UUU)
    FM = fenics.Constant(numpy.array([[1,0],[0,1]]))
    ieps = fenics.Constant(inveps)
    Fmv = fenics.variable(Fm)
    C = Fmv.T*Fmv
    Ic = fenics.tr(C)
    J = fenics.det(Fmv)
    psi = Ic-3-2*fenics.ln(J) + mu3*(J-1)**2
    dF = fenics.diff(psi, Fmv)

    tll = numpy.empty(shape=(mul.shape[0], rb_data.p.N))
    for i in range(mul.shape[0]):
        mu = mul[i,:]
        u_f = rb_data.return_u_full_order(ul[i,:])
        u.vector().set_local(u_f)
        mu3.assign(fenics.Constant(mu[3]))
        FM.assign(fenics.Constant(numpy.array([[mu[0]+1, mu[2]], [mu[2], mu[1]+1]])))
        Fm.assign(fenics.project(FM+ieps*fenics.grad(u), UUU))
        p_f = fenics.project (dF, UUU).vector().get_local()
        tll[i,:] = rb_data.project_p_rb(p_f)
    return tll

def get_other_p(mesh_data, um):
    # get un-consistent p with respect to given um and random mu
    UUU = fenics.TensorFunctionSpace(mesh_data.mesh, 'DG', 1, shape=(2,2))
    U = fenics.VectorFunctionSpace(mesh_data.mesh, 'CG', 2, dim=2)
    u = fenics.Function(U)
    pvar = fenics.Function(UUU)
    mu3 = fenics.Constant(1)
    Fm = fenics.Function(UUU)
    Fm1 = fenics.Function(UUU)
    Fm2 = fenics.Function(UUU)
    FM = fenics.Constant(numpy.array([[1,0],[0,1]]))
    ieps = fenics.Constant(inveps)
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    N_processes = comm.Get_size()
    if my_rank == 0:
        u_f = um
        numpy.random.seed(7)
        mu_f = get_training_parameter(u_f.shape[0])

        def divide2chunk(n_task, n_process):
            chunksize = n_task // n_process + 1
            rest = n_task % n_process
            task_list = [chunksize]*rest
            task_list.extend([chunksize-1]*(n_process-rest))
            for i in range(1, n_process):
                task_list[i] += task_list[i-1]
            return task_list
        n_task = u_f.shape[0]
        task_list = divide2chunk(n_task, N_processes)

        print("Process %d distributes parameters..." %my_rank)
        sys.stdout.flush()
        for p in range(1, N_processes):
            data_send = (mu_f[task_list[p-1]:task_list[p],:], u_f[task_list[p-1]:task_list[p],:])
            comm.send(data_send, dest=p)
        print("Process %d done sending!" %my_rank)
        sys.stdout.flush()
        n_task_p = task_list[0]
        muc = mu_f[:task_list[0],:]
        ufc = u_f[:task_list[0],:]
    else:
        muc, ufc = comm.recv(source=0)
        n_task_p = muc.shape[0]
        print("Process %d received the parameters!" %my_rank)
        sys.stdout.flush()
    
    for i in range(n_task_p):
        mu, uf = muc[i,:], ufc[i,:]
        u.vector().set_local(uf)
        mu3.assign(fenics.Constant(mu[3]))
        FM.assign(fenics.Constant(numpy.array([[mu[0]+1, mu[2]], [mu[2], mu[1]+1]])))
        Fm1.assign(fenics.project(FM, UUU))
        Fm2.assign(fenics.project(fenics.grad(u), UUU))
        Fm.assign(Fm1 + ieps*Fm2)
        Fmv = fenics.variable(Fm)
        C = Fmv.T*Fmv
        Ic = fenics.tr(C)
        J = fenics.det(Fmv)
        psi = 1*(Ic-3-2*fenics.ln(J)) + mu3*(J-1)**2
        dF = fenics.diff(psi, Fmv)
        pvar.assign(fenics.project (dF, UUU))
        pv = pvar.vector().get_local()
        if i == 0:
            tll = pv.reshape(1,-1).copy()
        else:
            tll = numpy.append(tll, pv.reshape(1,-1), axis=0)
        if i%100 == 99:
            print('Process n%d, finish %d/%d' % (my_rank ,i+1, n_task_p))
            sys.stdout.flush()
    print("Process %d finished the computation!" %my_rank)
    sys.stdout.flush()

    if my_rank == 0:
        for p in range(1, N_processes):
            tl_p = comm.recv(source=p)
            tll = numpy.append(tll, tl_p, axis=0)
        print("Process %d done!" %my_rank)
        sys.stdout.flush()
        return mu_f, tll
    else:
        comm.send(tll, dest=0)
        print("Process %d done!" %my_rank)
        sys.stdout.flush()

def get_mass_matrix(mesh_data):
    '''
    get the full order L2 mass matrix for u and p
    Parameters: mesh_data: MeshData()
                           the mesh, the cellmarkers and the facetmarkers
    Returns:    M1: 2D array   shape = (N_full_u, N_full_u)
                    the L2 mass matrix of u within Omega
                M2: 2D array   shape = (N_full_p, N_full_p)
                    the L2 mass matrix of p within Omega
    '''
    fenics.parameters['linear_algebra_backend'] = 'Eigen'
    UU = fenics.VectorFunctionSpace(mesh_data.mesh, 'CG', 2, dim=2)
    UUU = fenics.TensorFunctionSpace(mesh_data.mesh, 'DG', 1, shape=(2,2))
    u = fenics.TrialFunction(UU)
    u_ = fenics.TestFunction(UU)
    p = fenics.TrialFunction(UUU)
    p_ = fenics.TestFunction(UUU)
    m1 =  fenics.inner(u,u_)*fenics.dx + fenics.inner(fenics.grad(u), fenics.grad(u_))*fenics.dx
    m2 = fenics.inner(p, p_)*fenics.dx
    m1 = fenics.assemble(m1)
    m2 = fenics.assemble(m2)
    row,col,val = fenics.as_backend_type(m1).data()
    M1 = sp.csr_matrix((val, col, row))
    row,col,val = fenics.as_backend_type(m2).data()
    M2 = sp.csr_matrix((val, col, row))
    return M1, M2
    
def POD_l(W, M, tol): 
    '''
    Implement POD for W, M is the mass matrix, tol is the square of POD error tolerance 
    Parameters: W: 2D array   shape = (n, N) #shape = (sample_size, N_full)
                   the collection of vectors for 
                M: 2D array   shape = (N, N) #shape = (N_full, N_full)
                   the mass matrix of W
                tol: list of float   0.0<tol<1.0
                     average projection error tolerace, the POD error tolerance = tol**2
    Returns:    B: 2D array   shape = (N_full, N_rb) #N_rb = N_u, N_q, N_r, N_full = N_full_u, N_full_q, N_full_r
                   the reduced basis, the later the more significant
                i: int
                   the reduced basis size, = N_rb
                D: 1D array   shape = (n,) #shape = (sample_size,)
                   the eigenvalues in ascending order
                B_proj: 2D array   shape = (N_full, N_rb)
                        the projector matrix, = M*B
    '''
    tol2 = [ele*ele for ele in tol]
    W = sp.csr_matrix(W)
    C = sp.csr_matrix.dot(M, W.transpose()).toarray()
    W = W.toarray()
    C = numpy.dot(W, C)
    D, V = numpy.linalg.eigh(C)
    S = D.sum()
    output = []
    for ele in tol2:
        err, i = 1.0, 0
        while err > ele:
            i += 1
            err -= D[D.shape[0]-i]/S
        Vl = V[:,(D.shape[0]-i):]
        B = numpy.dot(numpy.dot(W.transpose(), Vl), numpy.diag(1.0/numpy.sqrt(D[(D.shape[0]-i):])))
        Bs = sp.csr_matrix(B)
        B_proj = sp.csr_matrix.dot(M, Bs)
        B_proj = B_proj.toarray()
        output.append((B, i, D, B_proj))
    return output

def get_Ku(rb_data, J):
    '''
    get the stiffness tensor wrt U by given the derivative of the TTDN
    Parameters: rb_data: RB_Data()
                         the reduced basis of u and p
                J: 2D array   shape=(N_p, N_para)
                   the Jacobian matrix for regression of p wrt para by NNF
    Returns:    Ku: 4D array   shape=(2,2,2,2)
                    stiffness tensor wrt U
    '''
    dim = rb_data.p_tensor[0].shape[0]
    Ku = numpy.zeros((dim,dim,dim,dim))
    for i in range(J.shape[0]):
        Ku[0,0,0,0] += J[i,0]*rb_data.p_tensor[i][0,0]
        Ku[0,1,0,0] += J[i,0]*rb_data.p_tensor[i][0,1]
        Ku[1,0,0,0] += J[i,0]*rb_data.p_tensor[i][1,0]
        Ku[1,1,0,0] += J[i,0]*rb_data.p_tensor[i][1,1]

        Ku[0,0,0,1] += J[i,2]*rb_data.p_tensor[i][0,0]
        Ku[0,1,0,1] += J[i,2]*rb_data.p_tensor[i][0,1]
        Ku[1,0,0,1] += J[i,2]*rb_data.p_tensor[i][1,0]
        Ku[1,1,0,1] += J[i,2]*rb_data.p_tensor[i][1,1]
        
        Ku[0,0,1,0] += J[i,2]*rb_data.p_tensor[i][0,0]
        Ku[0,1,1,0] += J[i,2]*rb_data.p_tensor[i][0,1]
        Ku[1,0,1,0] += J[i,2]*rb_data.p_tensor[i][1,0]
        Ku[1,1,1,0] += J[i,2]*rb_data.p_tensor[i][1,1]

        Ku[0,0,1,1] += J[i,1]*rb_data.p_tensor[i][0,0]
        Ku[0,1,1,1] += J[i,1]*rb_data.p_tensor[i][0,1]
        Ku[1,0,1,1] += J[i,1]*rb_data.p_tensor[i][1,0]
        Ku[1,1,1,1] += J[i,1]*rb_data.p_tensor[i][1,1]
    return Ku

