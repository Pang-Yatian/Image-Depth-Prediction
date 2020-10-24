import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import models

cap = cv2.VideoCapture(0)
height = 228
width = 304
dim = (width, height)
channels = 3
batch_size = 1

def main():


         # Read image

         input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

         # Construct the network
         net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

         with tf.Session() as sess:
             # Load the converted parameters
             print('Loading the model')

             # Use to load from ckpt file
             # saver = tf.train.Saver()
             # saver.restore(sess, model_data_path)
             plt.ion()
             # Use to load from npy file
             net.load("D:\\ME5001_project\\FCRN-DepthPrediction-master\\image_depth_prediction\\NYU_ResNet-UpProj.npy", sess)
             while(True):
                ret, img = cap.read()
                cv2.imshow('frame1', img)
             # print('Original Dimensions : ', img.shape) resize image
                resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                img = np.expand_dims(np.asarray(resized), axis=0)
                # Create a placeholder for the input imageqqqqqqQq

             # Evalute the network for the given image
                pred = sess.run(net.get_output(), feed_dict={input_node: img})
                print('Plot')

             # Plot result
                fig = plt.figure()
                ii = plt.imshow(pred[0, :, :, 0], interpolation='nearest')
                fig.colorbar(ii)
                fig.canvas.draw()
                data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                #final_img = Image.fromarray(data, 'RGB')
                plt.close()
                #gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
                #ret,imggray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
                cv2.imshow('frame',data)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                  break

             os._exit(0)


if __name__ == '__main__':
    main()

        



