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



class SMnistData:

    def __init__(self):
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

        train_x = train_x.astype(np.float32)/255.0
        test_x = test_x.astype(np.float32)/255.0

        train_split = int(0.9*train_x.shape[0])
        valid_x = train_x[train_split:]
        train_x = train_x[:train_split]
        valid_y = train_y[train_split:]
        train_y = train_y[:train_split]

        train_x = train_x.reshape([-1,28,28])
        test_x = test_x.reshape([-1,28,28])
        valid_x = valid_x.reshape([-1,28,28])


        self.valid_x = np.transpose(valid_x,(1,0,2))
        self.train_x = np.transpose(train_x,(1,0,2))
        self.test_x = np.transpose(test_x,(1,0,2))
        self.valid_y = valid_y
        self.train_y = train_y
        self.test_y = test_y

        print("Total number of training sequences: {}".format(train_x.shape[0]))
        print("Total number of validation sequences: {}".format(self.valid_x.shape[0]))
        print("Total number of test sequences: {}".format(self.test_x.shape[0]))

        
    def iterate_train(self,batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = self.train_x[:,permutation[start:end]]
            batch_y = self.train_y[permutation[start:end]]
            yield (batch_x,batch_y)

class SMnistModel:

    def __init__(self,model_type,model_size,learning_rate = 0.001):
        self.model_type = model_type
        self.constrain_op = None
        self.x = tf.placeholder(dtype=tf.float32,shape=[28,None,28])
        self.target_y = tf.placeholder(dtype=tf.int32,shape=[None])

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
            self.constrain_op = self.wm.get_param_constrain_op()
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
        
        print("head.shape: ",str(head.shape))
        unstack_head = tf.unstack(head,axis=0)
        head = unstack_head[-1]
        self.y = tf.layers.Dense(10,activation=None)(head)
        print("logit shape: ",str(self.y.shape))
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels = self.target_y,
            logits = self.y,
        ))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(model_prediction, tf.cast(self.target_y,tf.int64)), tf.float32))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join("results","smnist","{}_{}.csv".format(model_type,model_size))
        if(not os.path.exists("results/smnist")):
            os.makedirs("results/smnist")
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

        self.checkpoint_path = os.path.join("tf_sessions","smnist","{}".format(model_type))
        if(not os.path.exists("tf_sessions/smnist")):
            os.makedirs("tf_sessions/smnist")
            
        self.saver = tf.train.Saver()

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
                test_acc,test_loss = self.sess.run([self.accuracy,self.loss],{self.x: smnist_data.test_x,self.target_y: smnist_data.test_y})
                valid_acc,valid_loss = self.sess.run([self.accuracy,self.loss],{self.x:smnist_data.valid_x,self.target_y: smnist_data.valid_y})
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
                acc,loss,_ = self.sess.run([self.accuracy,self.loss,self.train_step],{self.x: batch_x,self.target_y: batch_y})
                if(not self.constrain_op is None):
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
    parser.add_argument('--size',default=30,type=int)
    parser.add_argument('--epochs',default=200,type=int)
    parser.add_argument('--save_path', default="saved_model/model.ckpt", type=str)  # Path to save the model
    parser.add_argument('--load_path', default=None, type=str)  # Path to load a saved model
    args = parser.parse_args()


    occ_data = SMnistData()
    #model = SMnistModel(model_type = args.model,model_size=args.size)

    #model.fit(occ_data,epochs=args.epochs,log_period=args.log)
    if args.load_path:
        # Load a saved model
        model = SMnistModel.load_model(args.model, args.size, args.load_path)
    else:
        model = SMnistModel(model_type=args.model, model_size=args.size)
        model.fit(occ_data, epochs=args.epochs, log_period=args.log)
        # Save the trained model
        model.save_model(args.save_path)

    # Evaluate the model
    precision, recall, f1_score = model.evaluate(occ_data.test_x, occ_data.test_y)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1_score)

