# Gesture and Body Position Drone  
### **By Amna Hayat, Calvin Li, Nicholas Prakoso, Renzheng Zheng**  

![Crazyflie Drone](https://bitcraze.io/wp-content/uploads/2021/04/crazyflie-2.1-800x600.jpg)  

## **Overview**  
This project showcases one of the most challenging things I have ever built—a **gesture-controlled and "follow-me" drone** using machine learning and embedded systems. The project integrates **computer vision, drone flight control, and real-time gesture recognition** to enable a **Crazyflie 2.1 drone** to respond to hand gestures and body movements.

## **Project Objective**  
- Control a **Crazyflie drone** using **hand gestures**.  
- Enable **body movement-based "follow-me" mode**.  
- Implement **active object avoidance**.  
- Integrate **machine learning (ML) for gesture and body movement recognition**.  
- Utilize onboard **camera and sensor data** for accurate flight control.  

---

## **Hardware Components**  
| Component          | Purpose |
|-------------------|---------|
| **Crazyflie 2.1** | Small programmable drone |
| **AI Deck**      | Runs onboard machine learning models, includes camera & WiFi |
| **Multiranger Deck** | Detects objects around the drone (up to 4m) |
| **Flow Deck**     | Measures drone movement relative to the ground |
| **CrazyRadio PA** | Enables communication between the drone and the computer |

---

## **Software & Libraries**  
The software stack is primarily **Python-based**, utilizing the following key libraries:  
- **OpenCV** → Image processing  
- **NumPy** → Mathematical operations  
- **MediaPipe** → Body & hand landmarks detection  
- **TensorFlow** → Gesture classification using a Neural Network  
- **Bitcraze firmware & cfclient** → Drone control and communication  

### **System Diagram**
