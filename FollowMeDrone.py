import cv2
import cv2 as cv
import mediapipe as mp
import numpy as np
import argparse
import socket
import struct
import time
import copy
import math

######################################################################## DRONE
import logging
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.utils.multiranger import Multiranger
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

if len(sys.argv) > 1:
    URI = sys.argv[1]

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)
######################################################################## DRONE
parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer example')
parser.add_argument("-n", default="192.168.4.1", metavar="ip", help="AI-deck IP")
parser.add_argument("-p", type=int, default=5000, metavar="port", help="AI-deck port")
parser.add_argument('--save', action='store_true', help="Save streamed images")
args = parser.parse_args()

deck_ip = args.n
deck_port = args.p

print("Connecting to socket on {}:{}".format(deck_ip, deck_port))
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((deck_ip, deck_port))
print("Socket connected")

imgdata = None
data_buffer = bytearray()

def rx_bytes(size):
    data = bytearray()
    while len(data) < size:
        packet_chunk = client_socket.recv(size-len(data))
        if not packet_chunk:
            raise ConnectionError("Socket connection lost")
        data.extend(packet_chunk)
    return data

start = time.time()
count = 0

def main():

    readings_avg = []
    readings_count = 0
    shoulder_count = 0
    
    cflib.crtp.init_drivers()

    global count
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # Initialize mediapipe pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5)

    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=cf) as scf:
        with MotionCommander(scf, default_height = 0.8) as motion_commander:
            with Multiranger(scf) as multiranger:
                keep_flying = True

                while keep_flying:
                    
                    VELOCITY = 0.5
                    velocity_x = 0.0
                    velocity_y = 0.0

                    if is_close(multiranger.front):
                        velocity_x -= VELOCITY
                        # print("Doing action F")
                    if is_close(multiranger.back):
                        velocity_x += VELOCITY
                        # print("Doing action B")
                    if is_close(multiranger.left):
                        velocity_y -= VELOCITY
                        # print("Doing action L")
                    if is_close(multiranger.right):
                        velocity_y += VELOCITY
                        # print("Doing action R")

                    if is_close(multiranger.up):
                        keep_flying = False
                    # time.sleep(0.1)
                    motion_commander.start_linear_motion(
                        velocity_x, velocity_y, 0)


                    # Get the image packet from the AI-deck
                    packetInfoRaw = rx_bytes(4)
                    [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)

                    imgHeader = rx_bytes(length - 2)
                    [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)

                    if magic == 0xBC:
                        img_stream = bytearray()
                        while len(img_stream) < size:
                            packet_info_raw = rx_bytes(4)
                            length, dst, src = struct.unpack('<HBB', packet_info_raw)
                            chunk = rx_bytes(length - 2)
                            img_stream.extend(chunk)

                        count = count + 1
                        # mean_time_per_image = (time.time() - start) / count
                        # print("Mean time per image: {:.2f}s".format(mean_time_per_image))

                        if format == 0:
                            # Assuming this is a raw Bayer image
                            bayer_img = np.frombuffer(img_stream, dtype=np.uint8)
                            bayer_img.shape = (244, 324)  # Adjust this if needed
                            color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGRA)
                            cv2.imshow('AI-Deck Stream - Raw', bayer_img)
                            cv2.imshow('AI-Deck Stream - Color', color_img)
                            if args.save:
                                cv2.imwrite(f"stream_out/raw/img_{count:06d}.png", bayer_img)
                                cv2.imwrite(f"stream_out/debayer/img_{count:06d}.png", color_img)
                            cv2.waitKey(1)
                        elif format == 1:
                            with open("img.jpeg", "wb") as f:
                                f.write(img_stream)
                            nparr = np.frombuffer(img_stream, np.uint8)
                            decoded = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
                    
                    image = decoded
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    image = cv.flip(image, 1)
                    scale_factor = 1.5
                    image = cv.resize(image, None, fx=scale_factor, fy=scale_factor)
                    # print("image recieved")
                    image.flags.writeable = False
                    
                    results = pose.process(image) ##send image to pose model
                    
                    image.flags.writeable = True
                    # print(results)

                    # cv2.imshow("decoded", decoded)
                    if results.pose_landmarks:
                        annotated_image = image.copy()
                        # print("annotated")
                        mp_drawing.draw_landmarks(
                            annotated_image,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        cv2.imshow('AI-Deck Stream - Pose', annotated_image)
                        # right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                        # left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

                        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        # missing_landmark_x = ((left_hip.x - right_hip.x) / 2) + right_hip.x
                        missing_landmark_x = ((left_shoulder.x - right_shoulder.x) / 2) + right_shoulder.x
                        # missing_landmark_y = left_hip.y
                        # print("missing landmark x:", missing_landmark_x, "\nmissing landmark y:", missing_landmark_y)

                        readings_avg.append(missing_landmark_x)
                        readings_count += 1

                        ## yaw follow me 
                        if(readings_count == 10):

                            average_x = sum(readings_avg) / len(readings_avg)
                            if average_x < 0.55 and average_x > 0.45:
                                print("Stay still")
                            elif average_x > 0.60:
                                print("Turn Right")
                                motion_commander.turn_left(15)
                            elif average_x < 0.40:
                                print("Turn Left")
                                motion_commander.turn_right(15)
                            
                            readings_avg = []
                            readings_count = 0
                        
                        ## Move forward follow me 
                        ##find euclidean distance, closer to cam == greater distance, farther from cam == less distance
                        distance_shoulders = math.sqrt((left_shoulder.x - right_shoulder.x)**2 + (left_shoulder.y - right_shoulder.y)**2)
                        shoulder_count += 1

                        if (shoulder_count == 10):
                            if(distance_shoulders < 0.1):
                                motion_commander.forward(0.8)
                                print("move forward")
                            
                            shoulder_count = 0

                    # Exit on ESC key
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

    # Clean up
    client_socket.close()
    cv2.destroyAllWindows()

def is_close(range):
    MIN_DISTANCE = 0.2  # m

    if range is None:
        return False
    else:
        return range < MIN_DISTANCE

if __name__ == '__main__':
    main()