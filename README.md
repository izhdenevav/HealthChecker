# Branch Arina

This function estimates the **head rotation angles** (yaw, pitch, roll) for each detected face and returns True or False.

## Installation & Run

To install the required dependencies and run the project, use the following commands:

```
bash
pip install -r requirements.txt
python angle_fun.py
```  

## Using in code  
You need to define the following buffers:  
```
N = 5  
landmark_buffers = [deque(maxlen=N) for _ in range(6)]  
yaw_buffer = deque(maxlen=N)  
pitch_buffer = deque(maxlen=N)  
roll_buffer = deque(maxlen=N)
```
