from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.dot(X, self.W) + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = self.input.shape[0]
        dX = np.dot(grad, self.W.T)
        dW = np.dot(self.input.T, grad)
        db = np.sum(grad, axis=0, keepdims=True)

        if self.weight_decay:
            dW += 2 * self.weight_decay_lambda * self.W

        self.grads['W'] = dW
        self.grads['b'] = db
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(out_channels, 1, 1))
        self.grads = {'W': None, 'b': None}
        self.input = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X
        batch_size, in_channels, H, W = X.shape
        out_H = int((H + 2 * self.padding - self.kernel_size) / self.stride + 1)
        out_W = int((W + 2 * self.padding - self.kernel_size) / self.stride + 1)
        output = np.zeros((batch_size, self.out_channels, out_H, out_W))

        X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                          mode='constant')

        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h_out in range(out_H):
                    for w_out in range(out_W):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size
                        output[b, c_out, h_out, w_out] = np.sum(
                            X_padded[b, :, h_start:h_end, w_start:w_end] * self.W[c_out, :, :, :]) + self.b[c_out, 0, 0]

        return output


    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, out_channels, new_H, new_W = grads.shape
        _, in_channels, H, W = self.input.shape

        X_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                          mode='constant')
        dX_padded = np.zeros_like(X_padded)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(new_H):
                    for w_out in range(new_W):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size

                        dX_padded[b, :, h_start:h_end, w_start:w_end] += grads[b, c_out, h_out, w_out] * self.W[c_out,
                                                                                                         :, :, :]
                        dW[c_out, :, :, :] += grads[b, c_out, h_out, w_out] * X_padded[b, :, h_start:h_end,
                                                                              w_start:w_end]
                        db[c_out, 0, 0] += grads[b, c_out, h_out, w_out]

        if self.weight_decay:
            dW += 2 * self.weight_decay_lambda * self.W

        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        self.grads['W'] = dW
        self.grads['b'] = db
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output

    def backward(self, grads):
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels
        batch_size = predicts.shape[0]
        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts
        epsilon = 1e-10
        log_probs = -np.log(probs[np.arange(batch_size), labels] +epsilon)
        loss = np.sum(log_probs) / batch_size
        return loss

    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]
        if self.has_softmax:
            probs = softmax(self.predicts)
            grads = probs.copy()
            grads[np.arange(batch_size), self.labels] -= 1
            grads /= batch_size
        else:
            # 如果没有softmax，这里需要根据具体情况计算梯度
            pass

        self.grads = grads
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """

    def __init__(self, model, lambda_reg=1e-8):
        self.model = model
        self.lambda_reg = lambda_reg

    def forward(self):
        total_loss = 0
        for layer in self.model.layers:
            if hasattr(layer, 'params'):
                for param in layer.params.values():
                    total_loss += np.sum(np.square(param))
        return 0.5 * self.lambda_reg * total_loss

    def backward(self):
        for layer in self.model.layers:
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key, param in layer.params.items():
                    layer.grads[key] += self.lambda_reg * param

def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition