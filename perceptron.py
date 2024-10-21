import numpy as np
import sys
STUDENT_NAME = ’Anand Murugan , Karthick , Shyam ’
STUDENT_ID = ’20985901 , 20891385 , 20891384 ’

# Default error tolerance , this is the upper limit for the error in our model

DEFAULT_ERR_TOLERANCE = 0.0001
# Learning rate : this can be any value between 0 and 1
LEARNING_RATE = 0.5
WEIGHT_RANGE = 0.1
DEFAULT_EPOCS = 2500
NODES_HIDDEN_LAYER = 4
OUTPUT_FILE_NAME = ’ model_weight .npy ’
MAX_ITERATION_OVER_FITTING = 5

class NNUtils :
    @staticmethod
    def sigmoid (x) -> float :
    """
    Classic activation function for non - linearity in a the learning model
    .
    """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def split_train_test ( input_data =None , train =70 , validation =15 , test =15) :
        if input_data is None :
            print (" train_test array is None , please retry with correct data ", file = sys. stderr )
            sys . exit (2)

        num_rows , num_cols = input_data . shape
        print (" There are total {0} rows and {1} columns in input array ".format ( num_rows , num_cols ))
        train_idx = int (( train / 100) * num_rows )
        validation_idx = int (( validation / 100) * num_rows )
        test_idx = int (( test / 100) * num_rows )
        idx = train_idx
        train_data = input_data [: idx ]
        idx += validation_idx
        validation_data = input_data [ train_idx : idx]
        test_data = input_data [idx :]
        print (" Input data is divided train {0} , validation {1} , test {2}".format (len( train_data ), len( validation_data ),
        len ( test_data )))
        return [ train_data , validation_data , test_data ]

    @staticmethod
    def softmax (x):
    # https :// towardsdatascience .com/ softmax - function - simplified -714068bf8156
    """Softmax function will normalize the output values .It will convert weighted sum values into probabilities that sum to
    one ."""
        exponential = np. exp(x)
        return exponential / np.sum( exponential , axis =1, keepdims = True )

    @staticmethod
    def loss (y_true , y_pred , eps =1e-15) :
        num_rows , num_cols = y_true . shape
        predictions = np. clip (y_true , eps , 1 - eps)
        return (-1 / num_rows ) * (np.sum ( y_pred * np.log ( predictions )))

    @staticmethod
    def accuracy (y_true , y_pred ):
        if not (len( y_true ) == len( y_pred )):
            print (’Size of predicted and true labels not equal .’)
            return 0.0

        corr = 0
        for i in range (0, len( y_true )):
            corr += 1 if ( y_true [i] == y_pred [i]).all () else 0

        return corr / len( y_true )


class BPLImplementation :
    # We will initialize the weight and the bias for the hidden layer (w1 ,b1) and the output layer (w2 ,b2)
    def __init__ (self , total_features , output_category , load_from_file = False ):
        self .w1 = np. random . randn ( total_features , NODES_HIDDEN_LAYER ) * WEIGHT_RANGE
        self .b1 = np. zeros ( NODES_HIDDEN_LAYER )
        self .w2 = np. random . randn ( NODES_HIDDEN_LAYER , output_category ) * WEIGHT_RANGE
        self .b2 = np. zeros ( output_category )
        self . total_loss = list ()

        # if load_from_file variable is True , then we need to load the weights and bias information from the file .
        # other words , no need to train the model .

        if load_from_file :
            dic = np. load ( OUTPUT_FILE_NAME , allow_pickle = True )
            self .w1 = np. array ( dic. item ().get(’w1 ’))
            self .b1 = np. array ( dic. item ().get(’b1 ’))
            self .w2 = np. array ( dic. item ().get(’w2 ’))
            self .b2 = np. array ( dic. item ().get(’b2 ’))

    def predict (self , test_data ):
        """here we are calculating the class distribution for the input ( feed
        forward )"""
        result = list ()
        o1 = np.dot( test_data , self .w1) + self .b1
        o2 = NNUtils . sigmoid (o1)
        o3 = np.dot(o2 , self .w2) + self .b2
        o4 = NNUtils . softmax (o3)
        for output in o4:
        # Initially probability for all the labels is zero .
        current_label = [0] * 4
        # finding the class with the largest predicted probability and updating it to 1.
        current_label [np. argmax ( output )] = 1
        result . append ( current_label )

        return np. array ( result )

    def fit(self , train , label , validate_x , validate_y ):
        """
        We will work in the following way , we will train our NN and validate the accumulative error in the
        output . If the error is below the tolerance we will check , if this happened 5 times , that means we can break the training loop /
        else we will continue and update the weights in a back propagation for all the required layers .
        """
        counter_no_updates = 0
        self . total_loss . append (0)

        print (" Before training variables are \n w1 : {0}\ nw2 : {1}\ nb1 :{2}\ nb2 : {3}". format ( self .w1 , self .w2 , self .b1 , self .b2))
        for epoch in range ( DEFAULT_EPOCS ):
        # o1 = x1.w1 + b1
        o1 = np.dot(train , self .w1) + self .b1
        # Activation function - non - linearity in a the learning model .
        o2 = NNUtils . sigmoid (o1)
        # o3 = o2.w2 + b2
        o3 = np.dot(o2 , self .w2) + self .b2
        # Softmax will maps the output from the hidden layer (o3) to 4 classes whose total sum = 1.
        o4 = NNUtils . softmax (o3)

        """
        Here we will do the validation to avoid over fitting . If the accuracy stops increasing after MAX_ITERATION_OVER_FITTING
        iterations . we will break the loop and finish the training . We will do the comparison against DEFAULT_ERR_TOLERANCE .
        """
        # o1 = x1.w1 + b1
        validation_o1 = np.dot( validate_x , self .w1) + self .b1
        # Activation
        validation_o2 = NNUtils . sigmoid ( validation_o1 )
        # o3 = o2.w2 + b2
        validation_o3 = np.dot( validation_o2 , self .w2) + self .b2
        # Softmax --> We are using softmax because we have more than 2 labels to classify .
        # We will normalise the output sum values into probabilities that sum to one.
        validation_o4 = NNUtils . softmax ( validation_o3 )
        # loss
        current_iteration_loss = NNUtils . loss ( validation_o4 , validate_y )
        print ("The current epoch is {0} and current_iteration_loss is {1}". format (epoch , current_iteration_loss ))
        if self . total_loss [ -1] - current_iteration_loss < DEFAULT_ERR_TOLERANCE :
        counter_no_updates += 1
        # If the total number of iterations happened where nothing changed on validation set ,
        # then over fitting is happening We should break from the training loop .
        if counter_no_updates > MAX_ITERATION_OVER_FITTING :
        break
        self . total_loss . append ( current_iteration_loss )

        # If we reached here , that means training didn ’t end and we need to back propagate
        # and update the weights in the desired layers .

        de_wrt_do = o4 - label
        de_wrt_hidden_layer = de_wrt_do .dot( self .w2.T) * o2 * (1 - o2)
        de_wrt_w2 = o2.T.dot( de_wrt_do ) / train . shape [0]
        de_wrt_w1 = train .T.dot( de_wrt_hidden_layer ) / train . shape [0]
        de_wrt_loss_b1 = np. sum( de_wrt_hidden_layer ) / train . shape [0]
        de_wrt_loss_b2 = np. sum( de_wrt_do ) / train . shape [0]

        self .w1 = self .w1 - LEARNING_RATE * de_wrt_w1
        self .b1 = self .b1 - LEARNING_RATE * de_wrt_loss_b1

        self .w2 = self .w2 - LEARNING_RATE * de_wrt_w2
        self .b2 = self .b2 - LEARNING_RATE * de_wrt_loss_b2

        print (" After training ended variables are \n w1 : {0}\ nw2 : {1}\ nb1 : {2}\ nb2 : {3}". format ( self .w1 , self .w2 , self .b1 , self .b2))
        self . total_loss . clear ()
        model = {’w1 ’: self .w1 , ’b1 ’: self .b1 , ’w2 ’: self .w2 , ’b2 ’: self .b2}
        np. save ( OUTPUT_FILE_NAME , model )


def test_mlp ( data_file ):
    # Load the test set
    # START
    test_data = np. loadtxt ( data_file , delimiter =’,’)
    # END

    # Load your network
    # START
    obj_bpl = BPLImplementation ( total_features = test_data . shape [1] , output_category =4, load_from_file = True )
    # END

    # Predict test set - one - hot encoded
    return obj_bpl . predict ( test_data )


’’’
How we will test your code :

from test_mlp import test_mlp , STUDENT_NAME , STUDENT_ID
from acc_calc import accuracy

y_pred = test_mlp ( ’./ test_data .csv ’)

test_labels = ...

test_accuracy = accuracy ( test_labels , y_pred ) *100
’’’
# This code will be called when main file is test_mlp .py , this will train the NN and output the information in
# OUTPUT_FILE_NAME file .
if __name__ == ’__main__ ’:
 try :
    # Load the data and split the data into train , validation and test .
    # Default split is 70, 15, 15 and can be changes by sending params to NNUtils . split_train_test
    train_data_input = np. loadtxt (’train_data .csv ’, delimiter =’,’)
    data_train , data_validation , data_test = NNUtils . split_train_test ( train_data_input )
    train_labels_input = np. loadtxt (’ train_labels .csv ’, delimiter =’,’)
    labels_train , labels_validation , labels_test = NNUtils . split_train_test ( train_labels_input )

    # Object from back propagation Algorithm implementation .
    obj_BPL = BPLImplementation ( total_features = train_data_input . shape [1] , output_category =4, load_from_file = False )
    # Use the training data to fit the NN
    obj_BPL . fit( data_train , labels_train , data_validation , labels_validation )

    # Test the prediction against the test data .
    print (" Accuracy -> {0}". format ( NNUtils . accuracy ( labels_test , obj_BPL . predict ( data_test ))))

    except IOError as e:
    print (f" This execution failed : {e}", file =sys. stderr )
    sys . exit (1)