import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU

import tensorflow as tf
import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import cv2
import os

import tensorflow as tf
import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse
import pandas as pd




class SMnistData:

    def __init__(self):
        self.labels = ['PNEUMONIA', 'NORMAL']
        self.img_size = 256

        train = self.get_training_data('chest_xray/train')
        test = self.get_training_data('chest_xray/test')
        val = self.get_training_data('chest_xray/val')
        self.x_train = []
        self.y_train = []

        self.x_val = []
        self.y_val = []

        self.x_test = []
        self.y_test = []
        

        for feature, label in train:
            self.x_train.append(feature)
            self.y_train.append(label)

        for feature, label in test:
            self.x_test.append(feature)
            self.y_test.append(label)
            
        for feature, label in val:
            self.x_val.append(feature)
            self.y_val.append(label)
        print("Length of x_train:", len(self.x_train))
        print("Length of y_train:", len(self.y_train))
        self.x_train = np.array(self.x_train)  # Check if shapes are consistent
        self.y_train = np.array(self.y_train)
        # self.x_train = np.stack( self.x_train, axis=1)
        # self.y_train = np.stack( self.y_train, axis=0)
        self.x_val = np.array(self.x_val)  # Check if shapes are consistent
        self.y_val = np.array(self.y_val)
        self.x_test = np.array(self.x_test)  # Check if shapes are consistent
        self.y_test = np.array(self.y_test)   
        # self.x_val = np.stack( self.x_val, axis=1)
        # self.y_val = np.stack( self.y_val, axis=0)

        # self.x_test = np.stack( self.x_test, axis=1)
        # self.y_test = np.stack(  self.y_test, axis=0)
        # self.x_train = np.array(self.x_train) / 255
        # self.x_val = np.array(self.x_val) / 255
        # self.x_test = np.array(self.x_test) / 255

    def get_training_data(self, data_dir):
        data = [] 
        for label in self.labels: 
            path = os.path.join(data_dir, label)
            class_num = self.labels.index(label)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    resized_arr = cv2.resize(img_arr, (self.img_size, self.img_size)) # Reshaping images to preferred size
                    data.append([resized_arr, class_num])
                except Exception as e:
                    print(e)
        return np.array(data)
    def iterate_train(self,batch_size=16):
        total_seqs = self.x_train.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = self.x_train[permutation[start:end]]
            batch_y = self.y_train[permutation[start:end]]
            yield (batch_x,batch_y)
    
        


class SMnistModel:

    def __init__(self,model_type,model_size,sparsity_level=0.0,learning_rate = 0.001):
        self.model_type = model_type
        self.learning_rate = 0.001
        self.constrain_op = []
        self.sparsity_level = sparsity_level
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,None,150])
        self.target_y = tf.placeholder(dtype=tf.int32,shape=[None])
        print(self.target_y.get_shape())
        self.model_size = model_size
        head = self.x
        if(model_type == "lstm"):
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)

            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type.startswith("ltc")):
            learning_rate = 0.005 # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
            if(model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif(model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head,_ = tf.nn.dynamic_rnn(self.wm,head,dtype=tf.float32,time_major=True)
            self.constrain_op.extend(self.wm.get_param_constrain_op())
        elif(model_type == "node"):
            self.fused_cell = NODE(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctgru"):
            self.fused_cell = CTGRU(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctrnn"):
            self.fused_cell = CTRNN(model_size,cell_clip=-1,global_feedback=True)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))
        if(self.sparsity_level > 0):
            self.constrain_op.extend(self.get_sparsity_ops())
        print("head.shape: ",str(head.shape))
        self.y = tf.layers.Dense(1,activation=None)(head)
        print("logit y shape: ",str(self.y.shape))
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels = self.target_y,
            logits = self.y,
        ))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(model_prediction, tf.cast(self.target_y,tf.int64)), tf.float32))

        self.predict_percent = tf.argsort(self.y, axis=2)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        self.result_file = os.path.join("results","lung","{}_{}.csv".format(model_type,model_size))
        if(not os.path.exists("results/lung")):
            os.makedirs("results/lung")
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

        self.checkpoint_path = os.path.join("tf_sessions","lung","{}".format(model_type))
        if(not os.path.exists("tf_sessions/lung")):
            os.makedirs("tf_sessions/lung")
            
        self.saver = tf.train.Saver()
    def get_sparsity_ops(self):
        tf_vars = tf.trainable_variables()
        op_list = []
        for v in tf_vars:
            # print("Variable {}".format(str(v)))
            if(v.name.startswith("rnn")):
                if(len(v.shape)<2):
                    # Don't sparsity biases
                    continue
                if("ltc" in v.name and (not "W:0" in v.name)):
                    # LTC can be sparsified by only setting w[i,j] to 0
                    # both input and recurrent matrix will be sparsified
                    continue
                op_list.append(self.sparse_var(v,self.sparsity_level))
                
        return op_list
        
    def sparse_var(self,v,sparsity_level):
        mask = np.random.choice([0, 1], size=v.shape, p=[sparsity_level,1-sparsity_level]).astype(np.float32)
        v_assign_op = tf.assign(v,v*mask)
        print("Var[{}] will be sparsified with {:0.2f} sparsity level".format(
            v.name,sparsity_level
        ))
        return v_assign_op
    
    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        
        self.saver.restore(self.sess, self.checkpoint_path)


    def fit(self,smnist_data,epochs,verbose=True,log_period=50):

        best_valid_accuracy = 0
        best_valid_stats = (0,0,0,0,0,0,0)
        self.save()
        for e in range(epochs):
            if(verbose and e%log_period == 0):
                test_acc,test_loss = self.sess.run([self.accuracy,self.loss],{self.x: smnist_data.x_test,self.target_y: smnist_data.y_test})
                valid_acc,valid_loss = self.sess.run([self.accuracy,self.loss],{self.x:smnist_data.x_val,self.target_y: smnist_data.y_val})
                # Accuracy metric -> higher is better
                if(valid_acc > best_valid_accuracy and e > 0):
                    best_valid_accuracy = valid_acc
                    best_valid_stats = (
                        e,
                        np.mean(losses),np.mean(accs)*100,
                        valid_loss,valid_acc*100,
                        test_loss,test_acc*100
                    )
                    self.save()

            losses = []
            accs = []
            for batch_x,batch_y in smnist_data.iterate_train(batch_size=16):
                acc,loss,_ = self.sess.run([self.accuracy,self.loss,self.train_step],{self.x : batch_x,self.target_y: batch_y})
                if(len(self.constrain_op) > 0):
                    self.sess.run(self.constrain_op)
                losses.append(loss)
                accs.append(acc)

            if(verbose and e%log_period == 0):
                print("Epochs {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
                    e,
                    np.mean(losses),np.mean(accs)*100,
                    valid_loss,valid_acc*100,
                    test_loss,test_acc*100
                ))
                with open(self.result_file,"a") as f:
                    f.write("{:03d},{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f}\n".format(
                    e,
                    np.mean(losses),np.mean(accs)*100,
                    valid_loss,valid_acc*100,
                    test_loss,test_acc*100
                ))
            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc = best_valid_stats
        print("Best epoch {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))
        with open(self.result_file,"a") as f:
            f.write("{:03d}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))
    def save_model(self, filepath):
        self.saver.save(self.sess, filepath)
        print("Model saved to", filepath)

    @classmethod
    def load_model(cls, model_type, model_size, filepath):
        loaded_model = cls(model_type, model_size)  # Create an instance of SMnistModel
        loaded_model.restore_from_file(filepath)
        return loaded_model

    def restore_from_file(self, filepath):
        self.restore()
        self.saver.restore(self.sess, filepath)
        print("Model restored from", filepath)

    def predict(self, input_data):
        predicted_labels = self.sess.run(self.y, {self.x: input_data})
        return np.argmax(predicted_labels, axis=1)
    def evaluate(self, input_data, true_labels):
        predicted_labels = self.predict(input_data)
        precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
        return precision, recall, f1_score

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="lstm")
    parser.add_argument('--log',default=1,type=int)
    parser.add_argument('--size',default=128,type=int)
    parser.add_argument('--epochs',default=200,type=int)
    parser.add_argument('--save_path', default="saved_model/model.ckpt", type=str)  # Path to save the model
    parser.add_argument('--load_path', default=None, type=str)  # Path to load a saved model
    parser.add_argument('--sparsity',default=0.0,type=float)

    args = parser.parse_args()


    occ_data = SMnistData()
    model = SMnistModel(model_type = args.model,model_size=args.size,sparsity_level=args.sparsity,learning_rate=0.001)

    model.fit(occ_data,epochs=args.epochs,log_period=args.log)

    precision, recall, f1_score = model.evaluate(occ_data.x_test, occ_data.y_test)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1_score)

