from keras import backend as K

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=-1, keepdims=True))
    
def eucl_dist_output_shape(shapes):
    s1, s2 = shapes
    return (s1[0], 1)

def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    s1, s2 = shapes
    return (s1[0], 1)

def contrastive_loss(margin=10): 
    def _contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    return _contrastive_loss

def triplet_loss(margin=100): 
    def _triplet_loss(y_true, y_pred):
        positive_distance, negative_distance = y_pred[:,0], y_pred[:,1]
        return K.mean(K.maximum(0.0, positive_distance - negative_distance + margin)) 

    # def _triplet_loss(y_true, y_pred):
    #     positive_distance, negative_distance = y_pred[:,0], y_pred[:,1]
    #     dist = K.sum(K.square(positive_distance) - K.square(negative_distance), axis=-1)
    #     return K.mean(K.maximum(0.0, dist + margin))
    
    return _triplet_loss

def triplet_bpr_loss(): 
    def _triplet_loss(X):
            user, positive, negative = X
            loss = 1.0 - K.sigmoid(
                K.sum(user * positive, axis=-1, keepdims=True) -
                K.sum(user * negative, axis=-1, keepdims=True))
            return loss
    return _triplet_loss

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)
