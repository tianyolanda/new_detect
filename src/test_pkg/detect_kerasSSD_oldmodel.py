#!/usr/bin/env python3

from __future__ import print_function
import roslib
import sys
import pyzed.camera as zcam
import pyzed.defines as sl
import pyzed.types as tp
import pyzed.core as core
import math
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image as keras_image
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.misc import imread
import tensorflow as tf
from keras import backend as K
import math
import time

from ssd_v2 import SSD300v2
#from ssd_utils import BBoxUtility
import cv2
import rospy
import std_msgs.msg
from detect_pkg.msg import OBJINFO
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import test_pkg.point_cloud2 as pcl2
from cv_bridge import CvBridge, CvBridgeError


class image_converter: 

#zed_image = 0

    def __init__(self):

        self.image_pub = rospy.Publisher("image_topic_2",Image,queue_size=10)
        self.zed_image = None
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/zed/rgb/image_rect_color",Image,self.callback)

    def callback(self,data):
        #global zed_image
        cv_image= self.bridge.imgmsg_to_cv2(data)
        self.zed_image = cv_image

class image_converter_pointcloud:

    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic_3",PointCloud2,queue_size=10)
        self.zed_image_pointcloud = None
        self.bridge = CvBridge()
        self.dict_1 = 0
        self.dict_2 = 0
        self.image_sub2 = rospy.Subscriber("/zed/point_cloud/cloud_registered",PointCloud2,self.callback_pointcloud)
        self.rate = rospy.Rate(1);
        # rospy.spin();

    def callback_pointcloud(self,data):
        # print('33333333333:icp.dict_1 = :',self.dict_1)    
        # print('33333333333:icp.dict_2 = :',self.dict_2) 
        data_out = pcl2.read_points(data,field_names = ("x", "y", "z"),skip_nans=False,uvs=[[self.dict_1,self.dict_2]])
        self.zed_image_pointcloud = data_out




class BBoxUtility(object):
    """Utility class to do some stuff with bounding boxes and priors.

    # Arguments
        num_classes: Number of classes including background.
        priors: Priors and variances, numpy tensor of shape (num_priors, 8),
            priors[i] = [xmin, ymin, xmax, ymax, varxc, varyc, varw, varh].
        overlap_threshold: Threshold to assign box to a prior.
        nms_thresh: Nms threshold.
        top_k: Number of total bboxes to be kept per image after nms step.

    # References
        https://arxiv.org/abs/1512.02325
    """
    # TODO add setter methods for nms_thresh and top_K
    def __init__(self, num_classes, priors=None, overlap_threshold=0.5,
                 nms_thresh=0.45, top_k=400):
        self.num_classes = num_classes
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

    @property
    def nms_thresh(self):
        return self._nms_thresh

    @nms_thresh.setter
    def nms_thresh(self, value):
        self._nms_thresh = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, value):
        self._top_k = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    def iou(self, box):
        """Compute intersection over union for the box with all priors.

        # Arguments
            box: Box, numpy tensor of shape (4,).

        # Return
            iou: Intersection over union,
                numpy tensor of shape (num_priors).
        """
        # compute intersection
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # compute union
        area_pred = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.priors[:, 2] - self.priors[:, 0])
        area_gt *= (self.priors[:, 3] - self.priors[:, 1])
        union = area_pred + area_gt - inter
        # compute iou
        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        """Encode box for training, do it only for assigned priors.

        # Arguments
            box: Box, numpy tensor of shape (4,).
            return_iou: Whether to concat iou to encoded values.

        # Return
            encoded_box: Tensor with encoded box
                numpy tensor of shape (num_priors, 4 + int(return_iou)).
        """
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        assigned_priors = self.priors[assign_mask]
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])
        # we encode variance
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh /
                                                  assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        """Assign boxes to priors for training.

        # Arguments
            boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes),
                num_classes without background.

        # Return
            assignment: Tensor with assigned boxes,
                numpy tensor of shape (num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                assignment[:, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                assignment[:, -7:] are all 0. See loss for more details.
        """
        assignment = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,
                                                         np.arange(assign_num),
                                                         :4]
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        """Convert bboxes from local predictions to shifted priors.

        # Arguments
            mbox_loc: Numpy array of predicted locations.
            mbox_priorbox: Numpy array of prior boxes.
            variances: Numpy array of variances.

        # Return
            decode_bbox: Shifted priors.
        """
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_width * variances[:, 1]
        decode_bbox_center_y += prior_center_y
        decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_height
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, background_label_id=0, keep_top_k=10,
                      confidence_threshold=0.01):
        """Do non maximum suppression (nms) on prediction results.

        # Arguments
            predictions: Numpy array of predicted values.
            num_classes: Number of classes for prediction.
            background_label_id: Label of background class.
            keep_top_k: Number of total bboxes to be kept per image
                after nms step.
            confidence_threshold: Only consider detections,
                whose confidences are larger than a threshold.

        # Return
            results: List of predictions for every picture. Each prediction is:
                [label, confidence, xmin, ymin, xmax, ymax]
        """
        mbox_loc = predictions[:, :, :4]
        variances = predictions[:, :, -4:]
        mbox_priorbox = predictions[:, :, -8:-4]
        mbox_conf = predictions[:, :, 4:-8]
        results = []
        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i],
                                            mbox_priorbox[i], variances[i])
            for c in range(self.num_classes):
                if c == background_label_id:
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    feed_dict = {self.boxes: boxes_to_process,
                                 self.scores: confs_to_process}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict)
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes),
                                            axis=1)
                    results[-1].extend(c_pred)
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                results[-1] = results[-1][:keep_top_k]
        return results




def main():

    rospy.init_node('detect_pkg', anonymous=True)
    rospy.sleep(0.5)
    publisher()
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8) 
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)
    K.set_session(sess)


    np.set_printoptions(suppress=True)

    voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                   'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                   'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
                   'Sheep', 'Sofa', 'Train', 'Tvmonitor']
    NUM_CLASSES = len(voc_classes) + 1
    class_colors = []
    for i in range(0, len(voc_classes)):
        # This can probably be written in a more elegant manner
        hue = 255*i/len(voc_classes)
        col = np.zeros((1,1,3)).astype("uint8")
        col[0][0][0] = hue
        col[0][0][1] = 128 # Saturation
        col[0][0][2] = 255 # Value
        cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
        col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
        class_colors.append(col)


    # global dict_1,dict_2
    # dict_1 = 0
    # dict_2 = 0

    ic = image_converter()
    icp = image_converter_pointcloud()
    r = rospy.Rate(50)
    rospy.sleep(0.2)
    img_shape = ic.zed_image.shape
    img_shape = (360, 640, 3)

    input_shape = img_shape
    print('---------initialization model...please wait----------')
    model = SSD300v2(input_shape, num_classes=NUM_CLASSES)
    model.load_weights('/home/ogai1234/catkin_ws/src/detect_pkg/bin/weights_SSD300.hdf5', by_name=True)
    print('---------model done----------')
    bbox_util = BBoxUtility(NUM_CLASSES)
    print('after loading')
    key = ''
    counter = 0
    t0 = time.time()
    frame_code = 1

    while (key != 113) and (not rospy.is_shutdown()):


        frame_code = frame_code + 1
        print("_________________________Frame:",frame_code)
        counter = counter + 1
        image = ic.zed_image
        frame = image
        frame = frame[:,:,0:3]
        res=cv2.resize(frame,(640,360)) 
        img_old = res

        img = keras_image.img_to_array(res)
        img = img[np.newaxis, :,:,:]
        #inputs.append(img.copy())
        #inputs = preprocess_input(np.array(inputs))
        inputs = preprocess_input(img)
        t1 = time.time()
        preds = model.predict(inputs)

        results = bbox_util.detection_out(preds)
        #results = bbox_util.detection_out(preds,keep_top_k=200,confidence_threshold=0.8)
        #results = detection_out(preds)

        #print(results)
        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin = results[0][:, 2]
        det_ymin = results[0][:, 3]
        det_xmax = results[0][:, 4]
        det_ymax = results[0][:, 5]


        global x,y,z,top_conf,type_code,distance,confidence,objPerFrame

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.8]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        objPerFrame=0
        global type_code_list,confidence_list,distance_list,x_list,y_list,z_list
        type_code_list=[0,0,0,0,0,0,0,0,0,0,0]
        confidence_list=[0,0,0,0,0,0,0,0,0,0,0]
        distance_list=[0,0,0,0,0,0,0,0,0,0,0]
        x_list=[0,0,0,0,0,0,0,0,0,0,0]
        y_list=[0,0,0,0,0,0,0,0,0,0,0]
        z_list=[0,0,0,0,0,0,0,0,0,0,0]




        # list store information of each object in one frame



        for i in top_indices:
            xmin = int(round(top_xmin[i] * img_old.shape[1]))
            ymin = int(round(top_xmin[i] * img_old.shape[0]))
            xmax = int(round(top_xmax[i] * img_old.shape[1]))
            ymax = int(round(top_ymax[i] * img_old.shape[0]))
            class_num = int(top_label_indices[i])
            x_center = round((xmin+xmax) / 2)
            y_center = round((ymin+ymax) / 2)
            # print('11111111111:icp.dict_1 = :',icp.dict_1)    
            # print('11111111111:icp.dict_2 = :',icp.dict_2) 
            icp.dict_1=x_center
            icp.dict_2=y_center
            # print('22222222222:icp.dict_1 = :',icp.dict_1)    
            # print('22222222222:icp.dict_2 = :',icp.dict_2) 
            rospy.sleep(0.04)
            # print('4444444444:icp.dict_1 = :',icp.dict_1)    
            # print('4444444444:icp.dict_2 = :',icp.dict_2) 
            point_cloud = icp.zed_image_pointcloud
            # print('5555555555:icp.dict_1 = :',icp.dict_1)    
            # print('5555555555:icp.dict_2 = :',icp.dict_2) 
            # print(point_cloud)
            x,y,z = 0,0,0
            #x,y,z is the position obtained from pointcloud2
            for p in point_cloud:
                x,y,z = p
                break

            distance = math.sqrt(x*x+y*y+z*z)

            if voc_classes[class_num-1] == "Person": 
                type_code = 1
                #'%.2f' % Is to retain 2 decimal places
                confidence = round(top_conf[i],2)
                objPerFrame = objPerFrame + 1


                #stuck into list for ROS transfer
                type_code_list[objPerFrame]=type_code
                confidence_list[objPerFrame]=confidence
                distance_list[objPerFrame]=round(distance,2)
                x_list[objPerFrame]=round(x,2)
                y_list[objPerFrame]=round(y,2)
                z_list[objPerFrame]=round(z,2)               
               
                cv2.rectangle(img_old, (xmin, ymin), (xmax, ymax), class_colors[class_num-1], 4)

                text = voc_classes[class_num-1] + " " + ('%.2f' % top_conf[i])+ " "+("distance: %.2f" %(distance))+ " "+("x: %.2f" %(x)) + " "+("y: %.2f" %(y))
                print("Person")
                print('confidence:',confidence)
                print('distance',distance_list[objPerFrame])
                print('x:',x_list[objPerFrame])
                print('y:',y_list[objPerFrame])
                print('z:',z_list[objPerFrame])

                # talker()
                text_top = (xmin, ymin - 10)
                text_bot = (xmin + 280, ymin + 5)
                text_pos = (xmin + 5, ymin)

                cv2.rectangle(img_old, text_top, text_bot, class_colors[class_num-1], -1)
                cv2.putText(img_old, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

            elif voc_classes[class_num-1] == "Car":
                type_code = 2
                confidence = round(top_conf[i],2)
                objPerFrame = objPerFrame + 1

                #stuck into list for ROS transfer
                type_code_list[objPerFrame]=type_code
                confidence_list[objPerFrame]=confidence
                distance_list[objPerFrame]=round(distance,2)
                x_list[objPerFrame]=round(x,2)
                y_list[objPerFrame]=round(y,2)
                z_list[objPerFrame]=round(z,2)

                cv2.rectangle(img_old, (xmin, ymin), (xmax, ymax), class_colors[class_num-1], 4)

                text = voc_classes[class_num-1] + " " + ('%.2f' % top_conf[i])+ " "+("distance: %.2f" %(distance))+ " "+("x: %.2f" %(x)) + " "+("y: %.2f" %(y))

                print("Car")
                print('confidence:',confidence)
                print('distance',distance_list[objPerFrame])
                print('x:',x_list[objPerFrame])
                print('y:',y_list[objPerFrame])
                print('z:',z_list[objPerFrame])

                # talker()
                text_top = (xmin, ymin - 10)
                text_bot = (xmin + 280, ymin + 5)
                text_pos = (xmin + 5, ymin)

                cv2.rectangle(img_old, text_top, text_bot, class_colors[class_num-1], -1)
                cv2.putText(img_old, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
      
        cv2.imshow("success!", img_old)
        key  = cv2.waitKey(1)
        print("There are",objPerFrame,"objects in this frame")
        t21 = time.time()
        print('fps {:f}'.format( 1 / (t21 - t1)))
        talker()

    


def talker():
    # type_ros=['type1']
    #rospy.init_node('detect_pkg', anonymous=True)
    r = rospy.Rate(10) #10hz
    objinfo = OBJINFO()
    objinfo.objnum=objPerFrame
    objinfo.type1=type_code_list[1]
    objinfo.type2=type_code_list[2]
    objinfo.type3=type_code_list[3]
    objinfo.type4=type_code_list[4]
    objinfo.type5=type_code_list[5]
    objinfo.type6=type_code_list[6]
    objinfo.type7=type_code_list[7]
    objinfo.type8=type_code_list[8]
    objinfo.type9=type_code_list[9]
    objinfo.type10=type_code_list[10]

    objinfo.confidence1=confidence_list[1]
    objinfo.confidence2=confidence_list[2]
    objinfo.confidence3=confidence_list[3]
    objinfo.confidence4=confidence_list[4]
    objinfo.confidence5=confidence_list[5]
    objinfo.confidence6=confidence_list[6]
    objinfo.confidence7=confidence_list[7]
    objinfo.confidence8=confidence_list[8]
    objinfo.confidence9=confidence_list[9]
    objinfo.confidence10=confidence_list[10]

    objinfo.distance1=distance_list[1]
    objinfo.distance2=distance_list[2]
    objinfo.distance3=distance_list[3]
    objinfo.distance4=distance_list[4]
    objinfo.distance5=distance_list[5]
    objinfo.distance6=distance_list[6]
    objinfo.distance7=distance_list[7]
    objinfo.distance8=distance_list[8]
    objinfo.distance9=distance_list[9]
    objinfo.distance10=distance_list[10]

    objinfo.x1=x_list[1]
    objinfo.x2=x_list[2]
    objinfo.x3=x_list[3]
    objinfo.x4=x_list[4]
    objinfo.x5=x_list[5]
    objinfo.x6=x_list[6]
    objinfo.x7=x_list[7]
    objinfo.x8=x_list[8]
    objinfo.x9=x_list[9]
    objinfo.x10=x_list[10]
    
    objinfo.y1=y_list[1]
    objinfo.y2=y_list[2]
    objinfo.y3=y_list[3]
    objinfo.y4=y_list[4]
    objinfo.y5=y_list[5]
    objinfo.y6=y_list[6]
    objinfo.y7=y_list[7]
    objinfo.y8=y_list[8]
    objinfo.y9=y_list[9]
    objinfo.y10=y_list[10]

    objinfo.z1=z_list[1]
    objinfo.z2=z_list[2]
    objinfo.z3=z_list[3]
    objinfo.z4=z_list[4]
    objinfo.z5=z_list[5]
    objinfo.z6=z_list[6]
    objinfo.z7=z_list[7]
    objinfo.z8=z_list[8]
    objinfo.z9=z_list[9]
    objinfo.z10=z_list[10]

    # objinfo.type_code_list_ros = type_code_list
    # objinfo.confidence_list_ros= confidence_list
    # objinfo.distance_list_ros = distance_list
    # objinfo.x_list_ros = x_list
    # objinfo.y_list_ros = y_list
    # objinfo.z_list_ros = z_list
    obj_info_pub.publish(objinfo)


def publisher():
    global obj_info_pub
    obj_info_pub = rospy.Publisher('obj_info', OBJINFO,queue_size=10)


if __name__ == "__main__":
    main()
